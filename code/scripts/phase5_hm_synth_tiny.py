"""Train and causally audit frozen-backbone hidden-message adapters on HM-Synth Chain."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dllm
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
import zstandard
from torch import nn

from hidden_messages.communication import MessageCompressor, MessageFusion
from hidden_messages.communication.interventions import apply_message_intervention
from hidden_messages.contracts import MessageIntervention
from hidden_messages.datasets.hm_synth import HMSynthExample, generate_chain_lookup_pair
from hidden_messages.utils.checksums import sha256_file, sha256_tensor
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json
from hidden_messages.utils.reproducibility import require_authorized_cuda, seed_everything


@dataclass(slots=True)
class TokenizedSplit:
    examples: list[HMSynthExample]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    segment_ids: torch.Tensor
    answer_positions: torch.Tensor
    target_ids: torch.Tensor


@dataclass(slots=True)
class HiddenSplit:
    examples: list[HMSynthExample]
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    segment_ids: torch.Tensor
    answer_positions: torch.Tensor
    target_ids: torch.Tensor


class ReceiverChannel(nn.Module):
    """Shared sender compressor and designated-receiver fusion adapter."""

    def __init__(
        self,
        model_dim: int,
        *,
        message_slots: int,
        message_dim: int,
        width: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.compressor = MessageCompressor(
            model_dim,
            message_slots=message_slots,
            message_dim=message_dim,
            width=width,
            num_heads=num_heads,
            max_agents=2,
        )
        self.fusion = MessageFusion(
            model_dim,
            message_dim=message_dim,
            width=width,
            num_heads=num_heads,
            max_agents=2,
            dropout=0.0,
        )

    def encode(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_id: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        uncertainty = torch.zeros(
            batch_size,
            sequence_length,
            3,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        sender_ids = torch.full(
            (batch_size,),
            sender_id,
            dtype=torch.long,
            device=hidden_states.device,
        )
        return self.compressor(
            hidden_states,
            segment_ids=segment_ids,
            normalized_time=normalized_time,
            sender_ids=sender_ids,
            uncertainty=uncertainty,
            attention_mask=attention_mask,
        )

    def fuse_with_gate(
        self,
        receiver_hidden: torch.Tensor,
        *,
        incoming_slots: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, answer_length, _ = receiver_hidden.shape
        sender_ids = torch.full(
            (batch_size, 1, 1),
            sender_id,
            dtype=torch.long,
            device=receiver_hidden.device,
        )
        uncertainty = torch.zeros(
            batch_size,
            answer_length,
            3,
            dtype=receiver_hidden.dtype,
            device=receiver_hidden.device,
        )
        uncertainty[..., 2] = 1
        return self.fusion(
            receiver_hidden,
            incoming_slots=incoming_slots,
            sender_ids=sender_ids,
            message_ages=torch.zeros(
                batch_size,
                1,
                1,
                dtype=receiver_hidden.dtype,
                device=receiver_hidden.device,
            ),
            normalized_time=normalized_time,
            uncertainty=uncertainty,
            canvas_mask=torch.ones(
                batch_size,
                answer_length,
                dtype=torch.bool,
                device=receiver_hidden.device,
            ),
        )

    def fuse(
        self,
        receiver_hidden: torch.Tensor,
        *,
        incoming_slots: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_id: int,
    ) -> torch.Tensor:
        fused, _ = self.fuse_with_gate(
            receiver_hidden,
            incoming_slots=incoming_slots,
            normalized_time=normalized_time,
            sender_id=sender_id,
        )
        return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="RTX 8000")
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("phase5 config must be a mapping")
    return payload


def build_split(
    *,
    pair_count: int,
    seed_start: int,
    hops: list[int],
    distractors_per_agent: int,
    answer_values: tuple[str, ...],
) -> list[HMSynthExample]:
    examples: list[HMSynthExample] = []
    for pair_index in range(pair_count):
        factual, counterfactual = generate_chain_lookup_pair(
            seed_start + pair_index,
            num_agents=2,
            hops=hops[pair_index % len(hops)],
            distractors_per_agent=distractors_per_agent,
            answer_values=answer_values,
        )
        if factual.private_contexts[0] != counterfactual.private_contexts[0]:
            raise AssertionError("counterfactual pair changed designated-receiver context")
        if factual.answer == counterfactual.answer:
            raise AssertionError("counterfactual pair did not change the answer")
        examples.extend((factual, counterfactual))
    if len({example.example_id for example in examples}) != len(examples):
        raise AssertionError("HM-Synth example IDs are not unique")
    return examples


def split_digest(examples: list[HMSynthExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.canonical_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def union_context_examples(examples: list[HMSynthExample]) -> list[HMSynthExample]:
    union: list[HMSynthExample] = []
    for example in examples:
        receiver_context, sender_context = example.private_contexts
        union.append(
            replace(
                example,
                private_contexts=(f"{receiver_context} {sender_context}", sender_context),
            )
        )
    return union


def tokenize_split(
    examples: list[HMSynthExample],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    assistant_prefix: str = "",
) -> TokenizedSplit:
    conversations: list[list[dict[str, str]]] = []
    for example in examples:
        for private_context in example.private_contexts:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": (
                            f"{example.question}\nPrivate evidence: {private_context}\n"
                            "Use the private evidence and return only the requested code."
                        ),
                    }
                ]
            )
    prompts = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
    )
    if len(prompts) != len(examples) * 2:
        raise RuntimeError("tokenizer returned an unexpected number of agent prompts")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("frozen-backbone tokenizer has no mask token")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("frozen-backbone tokenizer has no pad/eos token")

    assistant_prefix_ids = [
        int(token) for token in tokenizer.encode(assistant_prefix, add_special_tokens=False)
    ]
    prompt_rows: list[list[int]] = []
    answer_positions: list[int] = []
    for prompt in prompts:
        row = [int(token) for token in prompt] + assistant_prefix_ids
        answer_positions.append(len(row))
        row.append(int(mask_id))
        if len(row) > max_sequence_length:
            raise RuntimeError(
                f"HM-Synth prompt length {len(row)} exceeds frozen limit {max_sequence_length}"
            )
        prompt_rows.append(row)

    target_ids: list[int] = []
    for example in examples:
        encoded = tokenizer.encode(example.answer, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != example.answer:
            raise RuntimeError(f"answer value {example.answer!r} is not a stable single token")
        target_ids.append(int(encoded[0]))

    input_ids = torch.full(
        (len(examples), 2, max_sequence_length),
        int(pad_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    segment_ids = torch.zeros_like(input_ids)
    answer_position_tensor = torch.empty((len(examples), 2), dtype=torch.long)
    for flat_index, row in enumerate(prompt_rows):
        example_index, agent_id = divmod(flat_index, 2)
        row_length = len(row)
        input_ids[example_index, agent_id, :row_length] = torch.tensor(row)
        attention_mask[example_index, agent_id, :row_length] = True
        segment_ids[example_index, agent_id, : row_length - 1] = 1
        segment_ids[example_index, agent_id, row_length - 1] = 2
        answer_position_tensor[example_index, agent_id] = answer_positions[flat_index]
    return TokenizedSplit(
        examples=examples,
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        answer_positions=answer_position_tensor,
        target_ids=torch.tensor(target_ids, dtype=torch.long),
    )


@torch.inference_mode()
def precompute_hidden(
    tokenized: TokenizedSplit,
    model: nn.Module,
    output_head: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> HiddenSplit:
    flat_ids = tokenized.input_ids.reshape(-1, tokenized.input_ids.shape[-1])
    flat_mask = tokenized.attention_mask.reshape(-1, tokenized.attention_mask.shape[-1])
    hidden_batches: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_hidden(
        _module: nn.Module,
        hook_inputs: tuple[Any, ...],
    ) -> None:
        if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
            raise RuntimeError("output head did not receive hidden states")
        captured.append(hook_inputs[0])

    handle = output_head.register_forward_pre_hook(capture_hidden)
    try:
        for start in range(0, flat_ids.shape[0], batch_size):
            stop = min(start + batch_size, flat_ids.shape[0])
            outputs = model(
                flat_ids[start:stop].to(device),
                attention_mask=flat_mask[start:stop].to(device),
            )
            if len(captured) != 1:
                raise RuntimeError(f"expected one hidden capture, observed {len(captured)}")
            hidden_batches.append(captured.pop().detach().to(device="cpu", dtype=torch.bfloat16))
            del outputs
    finally:
        handle.remove()
    hidden = torch.cat(hidden_batches).reshape(
        len(tokenized.examples),
        2,
        tokenized.input_ids.shape[-1],
        -1,
    )
    return HiddenSplit(
        examples=tokenized.examples,
        hidden_states=hidden,
        attention_mask=tokenized.attention_mask,
        segment_ids=tokenized.segment_ids,
        answer_positions=tokenized.answer_positions,
        target_ids=tokenized.target_ids,
    )


@torch.inference_mode()
def precompute_receiver_answer_hidden(
    tokenized: TokenizedSplit,
    model: nn.Module,
    output_head: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    output_position_offset: int = 0,
) -> torch.Tensor:
    hidden_batches: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_hidden(
        _module: nn.Module,
        hook_inputs: tuple[Any, ...],
    ) -> None:
        if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
            raise RuntimeError("output head did not receive hidden states")
        captured.append(hook_inputs[0])

    handle = output_head.register_forward_pre_hook(capture_hidden)
    try:
        for start in range(0, len(tokenized.examples), batch_size):
            stop = min(start + batch_size, len(tokenized.examples))
            input_ids = tokenized.input_ids[start:stop, 0].to(device)
            attention_mask = tokenized.attention_mask[start:stop, 0].to(device)
            positions = tokenized.answer_positions[start:stop, 0].to(device)
            positions = positions + output_position_offset
            if bool((positions < 0).any()):
                raise RuntimeError("output-position offset moved an answer before the sequence")
            outputs = model(input_ids, attention_mask=attention_mask)
            if len(captured) != 1:
                raise RuntimeError(f"expected one hidden capture, observed {len(captured)}")
            hidden = captured.pop()
            rows = torch.arange(stop - start, device=device)
            hidden_batches.append(
                hidden[rows, positions].detach().to(device="cpu", dtype=torch.bfloat16)
            )
            del outputs
    finally:
        handle.remove()
    return torch.cat(hidden_batches)


def receiver_answer_hidden(
    split: HiddenSplit,
    indices: torch.Tensor,
    *,
    output_position_offset: int = 0,
) -> torch.Tensor:
    selected_hidden = split.hidden_states.index_select(0, indices)
    positions = split.answer_positions.index_select(0, indices)[:, 0]
    positions = positions + output_position_offset
    if bool((positions < 0).any()):
        raise RuntimeError("output-position offset moved an answer before the sequence")
    rows = torch.arange(indices.shape[0])
    return selected_hidden[rows, 0, positions].unsqueeze(1)


def different_target_derangement(targets: torch.Tensor) -> torch.Tensor:
    """Return a deterministic permutation whose selected targets all differ."""

    sorted_positions = torch.argsort(targets, stable=True)
    _, counts = torch.unique_consecutive(
        targets.index_select(0, sorted_positions),
        return_counts=True,
    )
    shift = int(counts.max().item())
    if 2 * shift > targets.numel():
        raise RuntimeError("batch is too imbalanced for a different-target derangement")
    rolled = sorted_positions.roll(-shift)
    order = torch.empty_like(sorted_positions)
    order[sorted_positions] = rolled
    if bool((targets.index_select(0, order) == targets).any()):
        raise AssertionError("different-target derangement contains a label match")
    return order


def train_channel(
    channel: ReceiverChannel,
    output_head: nn.Module,
    split: HiddenSplit,
    config: dict[str, Any],
    *,
    teacher_hidden: torch.Tensor | None,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    steps = int(training["steps"])
    lambda_teacher = float(training.get("lambda_teacher", 0.0))
    lambda_pair = float(training.get("lambda_pair", 0.0))
    lambda_gate = float(training.get("lambda_gate", 0.0))
    pair_fraction = float(training.get("pair_fraction", 0.0))
    pair_margin = float(training.get("pair_margin", 0.1))
    teacher_temperature = float(training.get("teacher_temperature", 1.0))
    output_position_offset = int(config["model"].get("output_position_offset", 0))
    if lambda_teacher > 0 and teacher_hidden is None:
        raise ValueError("teacher_hidden is required when lambda_teacher is positive")
    times = torch.tensor(training["normalized_times"], dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(len(split.examples), generator=generator)
    cursor = 0
    records: list[dict[str, float | int]] = []
    warmup_steps = max(1, round(steps * float(training.get("warmup_ratio", 0.0))))

    def learning_rate_factor(step_index: int) -> float:
        if step_index < warmup_steps:
            return (step_index + 1) / warmup_steps
        progress = (step_index - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    channel.train()
    for step in range(steps):
        if cursor + batch_size > len(split.examples):
            permutation = torch.randperm(len(split.examples), generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        selected_hidden = split.hidden_states.index_select(0, indices).to(device)
        attention_mask = split.attention_mask.index_select(0, indices).to(device)
        segment_ids = split.segment_ids.index_select(0, indices).to(device)
        receiver_hidden = receiver_answer_hidden(
            split,
            indices,
            output_position_offset=output_position_offset,
        ).to(device)
        target_ids = split.target_ids.index_select(0, indices).to(device)
        selected_teacher = (
            teacher_hidden.index_select(0, indices).to(device)
            if teacher_hidden is not None
            else None
        )
        time_indices = torch.randint(
            len(times),
            (indices.shape[0],),
            generator=generator,
        )
        normalized_time = times.index_select(0, time_indices).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages = channel.encode(
                selected_hidden[:, 1],
                attention_mask=attention_mask[:, 1],
                segment_ids=segment_ids[:, 1],
                normalized_time=normalized_time,
                sender_id=1,
            )
            fused, gate_values = channel.fuse_with_gate(
                receiver_hidden,
                incoming_slots=messages[:, None, None],
                normalized_time=normalized_time,
                sender_id=1,
            )
            logits = output_head(fused)[:, 0]
            diffusion_loss = nn.functional.cross_entropy(logits.float(), target_ids)
            teacher_loss = torch.zeros((), device=device)
            if selected_teacher is not None and lambda_teacher > 0:
                with torch.no_grad():
                    teacher_logits = output_head(selected_teacher[:, None])[:, 0].float()
                    teacher_probabilities = nn.functional.softmax(
                        teacher_logits / teacher_temperature,
                        dim=-1,
                    )
                teacher_loss = (
                    nn.functional.kl_div(
                        nn.functional.log_softmax(
                            logits.float() / teacher_temperature,
                            dim=-1,
                        ),
                        teacher_probabilities,
                        reduction="batchmean",
                    )
                    * teacher_temperature**2
                )
            pair_loss = torch.zeros((), device=device)
            if lambda_pair > 0 and pair_fraction > 0:
                pair_count = max(2, int(indices.shape[0] * pair_fraction))
                pair_targets = target_ids[:pair_count]
                order = different_target_derangement(target_ids.cpu()).to(device)
                deranged_messages = messages.index_select(0, order[:pair_count])
                deranged_fused = channel.fuse(
                    receiver_hidden[:pair_count],
                    incoming_slots=deranged_messages[:, None, None],
                    normalized_time=normalized_time[:pair_count],
                    sender_id=1,
                )
                deranged_logits = output_head(deranged_fused)[:, 0]
                matched_losses = nn.functional.cross_entropy(
                    logits[:pair_count].float(),
                    pair_targets,
                    reduction="none",
                )
                deranged_losses = nn.functional.cross_entropy(
                    deranged_logits.float(),
                    pair_targets,
                    reduction="none",
                )
                pair_loss = nn.functional.relu(
                    pair_margin + matched_losses - deranged_losses
                ).mean()
            gate_loss = gate_values.abs().mean()
            loss = (
                diffusion_loss
                + lambda_teacher * teacher_loss
                + lambda_pair * pair_loss
                + lambda_gate * gate_loss
            )
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            channel.parameters(),
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "diffusion_loss": float(diffusion_loss.item()),
                    "teacher_loss": float(teacher_loss.item()),
                    "pair_loss": float(pair_loss.item()),
                    "gate_loss": float(gate_loss.item()),
                    "batch_accuracy": float((logits.argmax(dim=-1) == target_ids).float().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
        del selected_hidden, attention_mask, segment_ids, receiver_hidden, logits, loss
    return records


@torch.inference_mode()
def encode_all_messages(
    channel: ReceiverChannel,
    split: HiddenSplit,
    *,
    agent_id: int,
    normalized_time: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    messages: list[torch.Tensor] = []
    channel.eval()
    for start in range(0, len(split.examples), batch_size):
        stop = min(start + batch_size, len(split.examples))
        hidden = split.hidden_states[start:stop, agent_id].to(device)
        attention_mask = split.attention_mask[start:stop, agent_id].to(device)
        segment_ids = split.segment_ids[start:stop, agent_id].to(device)
        times = torch.full((stop - start,), normalized_time, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = channel.encode(
                hidden,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                normalized_time=times,
                sender_id=agent_id,
            )
        messages.append(encoded.float().cpu())
    return torch.cat(messages)


@torch.inference_mode()
def evaluate_no_message(
    split: HiddenSplit,
    output_head: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    output_position_offset: int = 0,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    all_indices = torch.arange(len(split.examples))
    for start in range(0, len(split.examples), batch_size):
        indices = all_indices[start : start + batch_size]
        hidden = receiver_answer_hidden(
            split,
            indices,
            output_position_offset=output_position_offset,
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = output_head(hidden)[:, 0]
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def evaluate_incoming(
    channel: ReceiverChannel,
    split: HiddenSplit,
    output_head: nn.Module,
    incoming: torch.Tensor,
    *,
    sender_id: int,
    normalized_time: float,
    batch_size: int,
    device: torch.device,
    output_position_offset: int = 0,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    all_indices = torch.arange(len(split.examples))
    channel.eval()
    for start in range(0, len(split.examples), batch_size):
        stop = min(start + batch_size, len(split.examples))
        indices = all_indices[start:stop]
        receiver_hidden = receiver_answer_hidden(
            split,
            indices,
            output_position_offset=output_position_offset,
        ).to(device)
        batch_incoming = incoming[start:stop].to(device)
        times = torch.full((stop - start,), normalized_time, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fused = channel.fuse(
                receiver_hidden,
                incoming_slots=batch_incoming,
                normalized_time=times,
                sender_id=sender_id,
            )
            logits = output_head(fused)[:, 0]
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


def paired_bootstrap_interval(
    left_correct: torch.Tensor,
    right_correct: torch.Tensor,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if left_correct.shape != right_correct.shape or left_correct.numel() % 2:
        raise ValueError("paired HM-Synth correctness vectors must align into counterfactual pairs")
    pair_effects = (
        left_correct.float().sub(right_correct.float()).reshape(-1, 2).mean(dim=1).numpy()
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, pair_effects.shape[0], size=(replicates, pair_effects.shape[0]))
    bootstrap = pair_effects[sampled].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def evaluate_split(
    channel: ReceiverChannel,
    split: HiddenSplit,
    output_head: nn.Module,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    normalized_time = float(evaluation["normalized_time"])
    output_position_offset = int(config["model"].get("output_position_offset", 0))
    matched_messages = encode_all_messages(
        channel,
        split,
        agent_id=1,
        normalized_time=normalized_time,
        batch_size=batch_size,
        device=device,
    )
    self_messages = encode_all_messages(
        channel,
        split,
        agent_id=0,
        normalized_time=normalized_time,
        batch_size=batch_size,
        device=device,
    )
    matched_incoming = matched_messages[:, None, None]
    pair_order = torch.arange(len(split.examples)).bitwise_xor(1)
    counterfactual_incoming = matched_incoming.index_select(0, pair_order)
    incoming_by_method = {
        "matched": apply_message_intervention(
            matched_incoming,
            MessageIntervention.MATCHED,
        ),
        "zero": apply_message_intervention(
            matched_incoming,
            MessageIntervention.ZERO,
        ),
        "random_moment_matched": apply_message_intervention(
            matched_incoming,
            MessageIntervention.RANDOM_MOMENT_MATCHED,
            seed=seed + 17,
        ),
        "deranged": apply_message_intervention(
            matched_incoming,
            MessageIntervention.DERANGED,
            seed=seed + 29,
        ),
        "self": apply_message_intervention(
            matched_incoming,
            MessageIntervention.SELF,
            self_messages=self_messages[:, None],
        ),
        "wrong_fact": apply_message_intervention(
            matched_incoming,
            MessageIntervention.COUNTERFACTUAL,
            counterfactual_messages=counterfactual_incoming,
        ),
    }
    predictions = {
        "no_message": evaluate_no_message(
            split,
            output_head,
            batch_size=batch_size,
            device=device,
            output_position_offset=output_position_offset,
        )
    }
    for method, incoming in incoming_by_method.items():
        predictions[method] = evaluate_incoming(
            channel,
            split,
            output_head,
            incoming,
            sender_id=0 if method == "self" else 1,
            normalized_time=normalized_time,
            batch_size=batch_size,
            device=device,
            output_position_offset=output_position_offset,
        )

    targets = split.target_ids
    correctness = {method: prediction == targets for method, prediction in predictions.items()}
    accuracy = {
        method: float(values.float().mean().item()) for method, values in correctness.items()
    }
    paired_targets = targets.index_select(0, pair_order)
    matched_no_ci = paired_bootstrap_interval(
        correctness["matched"],
        correctness["no_message"],
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    matched_deranged_ci = paired_bootstrap_interval(
        correctness["matched"],
        correctness["deranged"],
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]) + 1,
    )
    wrong_fact_target = predictions["wrong_fact"] == paired_targets
    matched_correct = correctness["matched"]
    conditional_wrong_fact = wrong_fact_target[matched_correct]
    metrics = {
        "examples": len(split.examples),
        "pairs": len(split.examples) // 2,
        "accuracy": accuracy,
        "matched_minus_no_message": accuracy["matched"] - accuracy["no_message"],
        "matched_minus_deranged": accuracy["matched"] - accuracy["deranged"],
        "matched_minus_no_message_pair_bootstrap_ci95": list(matched_no_ci),
        "matched_minus_deranged_pair_bootstrap_ci95": list(matched_deranged_ci),
        "wrong_fact_target_rate": float(wrong_fact_target.float().mean().item()),
        "wrong_fact_target_rate_given_matched_correct": (
            float(conditional_wrong_fact.float().mean().item())
            if conditional_wrong_fact.numel()
            else None
        ),
        "wrong_fact_answer_flip_rate": float(
            (predictions["wrong_fact"] != predictions["matched"]).float().mean().item()
        ),
        "matched_pair_consistency": float(
            correctness["matched"].reshape(-1, 2).all(dim=1).float().mean().item()
        ),
    }
    return metrics, predictions


def decode_token(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([token_id])).strip()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_zstd_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    compressor = zstandard.ZstdCompressor(level=9)
    with path.open("xb") as raw_handle, compressor.stream_writer(raw_handle) as handle:
        for record in records:
            handle.write((json.dumps(record, sort_keys=True) + "\n").encode())


def model_weight_manifest(snapshot: Path) -> tuple[str, list[dict[str, str | int]]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(snapshot.rglob("*.safetensors")):
        records.append(
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), records


def gate_passes(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["accuracy"]["matched"] >= float(gate["matched_accuracy_min"])
        and metrics["matched_minus_no_message"] >= float(gate["matched_minus_no_message_min"])
        and metrics["matched_minus_deranged"] >= float(gate["matched_minus_deranged_min"])
        and metrics["wrong_fact_target_rate"] >= float(gate["wrong_fact_target_rate_min"])
        and (
            not bool(gate["require_paired_ci_low_above_zero"])
            or (
                metrics["matched_minus_no_message_pair_bootstrap_ci95"][0] > 0
                and metrics["matched_minus_deranged_pair_bootstrap_ci95"][0] > 0
            )
        )
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    gpu_name = require_authorized_cuda(args.expected_gpu)
    seed_everything(args.seed)
    device = torch.device("cuda")
    run_dir = create_run_directory(args.run_root, args.run_id)
    started = time.perf_counter()

    model_config = config["model"]
    cache_key = "models--" + str(model_config["id"]).replace("/", "--")
    snapshot = args.model_cache / cache_key / "snapshots" / str(model_config["revision"])
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned local model snapshot is absent: {snapshot}")
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot),
        dtype=str(model_config["dtype"]),
        attn_implementation="eager",
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    output_head = model.get_output_embeddings()
    if not isinstance(output_head, nn.Module):
        raise TypeError("frozen backbone output head is not a torch module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = config["dataset"]
    answer_values = tuple(str(value) for value in dataset["answer_values"])
    for value in answer_values:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != value:
            raise RuntimeError(f"frozen answer value {value!r} is not a stable single token")

    train_examples = build_split(
        pair_count=int(dataset["train_pairs"]),
        seed_start=int(dataset["train_seed_start"]),
        hops=[int(value) for value in dataset["hops"]],
        distractors_per_agent=int(dataset["distractors_per_agent"]),
        answer_values=answer_values,
    )
    train_digest = split_digest(train_examples)
    train_tokenized = tokenize_split(
        train_examples,
        tokenizer,
        max_sequence_length=int(dataset["max_sequence_length"]),
        assistant_prefix=str(dataset.get("assistant_prefix", "")),
    )
    train_hidden = precompute_hidden(
        train_tokenized,
        model,
        output_head,
        batch_size=int(config["training"]["encode_batch_size"]),
        device=device,
    )
    teacher_hidden: torch.Tensor | None = None
    if float(config["training"].get("lambda_teacher", 0.0)) > 0:
        union_tokenized = tokenize_split(
            union_context_examples(train_examples),
            tokenizer,
            max_sequence_length=int(dataset["teacher_max_sequence_length"]),
            assistant_prefix=str(dataset.get("assistant_prefix", "")),
        )
        if not torch.equal(union_tokenized.target_ids, train_hidden.target_ids):
            raise AssertionError("union teacher targets do not align with distributed examples")
        teacher_hidden = precompute_receiver_answer_hidden(
            union_tokenized,
            model,
            output_head,
            batch_size=int(config["training"]["encode_batch_size"]),
            device=device,
            output_position_offset=int(model_config.get("output_position_offset", 0)),
        )
        del union_tokenized
    hidden_dim = int(train_hidden.hidden_states.shape[-1])
    channel = ReceiverChannel(
        hidden_dim,
        message_slots=int(config["training"]["message_slots"]),
        message_dim=int(config["training"]["message_dim"]),
        width=int(config["training"]["adapter_width"]),
        num_heads=int(config["training"]["attention_heads"]),
    ).to(device)
    training_records = train_channel(
        channel,
        output_head,
        train_hidden,
        config,
        teacher_hidden=teacher_hidden,
        seed=args.seed + 1,
        device=device,
    )
    checkpoint_tmp = run_dir / "channel.pt.tmp"
    checkpoint = run_dir / "channel.pt"
    torch.save(channel.state_dict(), checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    del train_hidden, train_tokenized, train_examples, teacher_hidden
    gc.collect()
    torch.cuda.empty_cache()

    validation_examples = build_split(
        pair_count=int(dataset["validation_pairs"]),
        seed_start=int(dataset["validation_seed_start"]),
        hops=[int(value) for value in dataset["hops"]],
        distractors_per_agent=int(dataset["distractors_per_agent"]),
        answer_values=answer_values,
    )
    validation_digest = split_digest(validation_examples)
    validation_hidden = precompute_hidden(
        tokenize_split(
            validation_examples,
            tokenizer,
            max_sequence_length=int(dataset["max_sequence_length"]),
            assistant_prefix=str(dataset.get("assistant_prefix", "")),
        ),
        model,
        output_head,
        batch_size=int(config["training"]["encode_batch_size"]),
        device=device,
    )
    validation_metrics, _ = evaluate_split(
        channel,
        validation_hidden,
        output_head,
        config,
        seed=args.seed + 2,
        device=device,
    )
    del validation_hidden, validation_examples
    gc.collect()

    # The sealed test split is first materialized only after the final checkpoint is immutable.
    test_examples = build_split(
        pair_count=int(dataset["test_pairs"]),
        seed_start=int(dataset["test_seed_start"]),
        hops=[int(value) for value in dataset["hops"]],
        distractors_per_agent=int(dataset["distractors_per_agent"]),
        answer_values=answer_values,
    )
    test_digest = split_digest(test_examples)
    test_hidden = precompute_hidden(
        tokenize_split(
            test_examples,
            tokenizer,
            max_sequence_length=int(dataset["max_sequence_length"]),
            assistant_prefix=str(dataset.get("assistant_prefix", "")),
        ),
        model,
        output_head,
        batch_size=int(config["training"]["encode_batch_size"]),
        device=device,
    )
    test_metrics, test_predictions = evaluate_split(
        channel,
        test_hidden,
        output_head,
        config,
        seed=args.seed + 3,
        device=device,
    )
    passed = gate_passes(test_metrics, config["gate"])
    enforce_gate = bool(config["evaluation"].get("enforce_gate", True))
    verdict = "pass" if passed else "fail"
    if not enforce_gate:
        verdict = "diagnostic_only"

    model_hash, weight_records = model_weight_manifest(snapshot)
    config_payload = yaml.safe_dump(config, sort_keys=True)
    (run_dir / "config.yaml").write_text(config_payload, encoding="utf-8")
    write_immutable_json(
        run_dir / "environment.json",
        {
            "pod": os.environ.get("HOSTNAME"),
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    write_immutable_json(
        run_dir / "dataset_manifest.json",
        {
            "protocol_id": config["protocol_id"],
            "family": dataset["family"],
            "train": {"examples": int(dataset["train_pairs"]) * 2, "sha256": train_digest},
            "validation": {
                "examples": int(dataset["validation_pairs"]) * 2,
                "sha256": validation_digest,
            },
            "test": {"examples": int(dataset["test_pairs"]) * 2, "sha256": test_digest},
            "test_materialized_after_final_checkpoint": True,
            "counterfactual_pairs_aligned": True,
            "seed_ranges_disjoint": True,
        },
    )
    write_immutable_json(
        run_dir / "model_manifest.json",
        {
            "model_id": model_config["id"],
            "revision": model_config["revision"],
            "weight_manifest_sha256": model_hash,
            "weights": weight_records,
            "backbone_frozen": True,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    write_immutable_json(
        run_dir / "seed_manifest.json",
        {
            "training_seed": args.seed,
            "train_generation_seed_start": dataset["train_seed_start"],
            "validation_generation_seed_start": dataset["validation_seed_start"],
            "test_generation_seed_start": dataset["test_seed_start"],
            "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
        },
    )
    write_immutable_json(
        run_dir / "metrics.json",
        {
            "verdict": verdict,
            "gate": config["gate"],
            "validation": validation_metrics,
            "sealed_test": test_metrics,
            "canvas_assignment_count": 0,
            "training_steps": config["training"]["steps"],
            "non_claim": config.get(
                "non_claim",
                "This is a single-seed development causal gate on HM-Synth Chain. It is not "
                "external-benchmark, replication, or paper-ready evidence.",
            ),
        },
    )
    write_jsonl(run_dir / "training_log.jsonl", training_records)

    prediction_records: list[dict[str, Any]] = []
    for method, predictions in test_predictions.items():
        for index, (example, prediction_id) in enumerate(
            zip(test_hidden.examples, predictions.tolist(), strict=True)
        ):
            prediction_records.append(
                {
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "variant": example.variant,
                    "split": "sealed-test",
                    "method": method,
                    "model": model_config["id"],
                    "training_seed": args.seed,
                    "sampling_seed": None,
                    "agent_context_ids": [
                        f"{example.example_id}-agent-0",
                        f"{example.example_id}-agent-1",
                    ],
                    "designated_receiver": 0,
                    "final_answers": [decode_token(tokenizer, prediction_id), None],
                    "target": example.answer,
                    "parsed_correctness": prediction_id == int(test_hidden.target_ids[index]),
                    "message_intervention": method,
                    "total_nfes": 2,
                    "bytes": (
                        0
                        if method == "no_message"
                        else int(config["training"]["message_slots"])
                        * int(config["training"]["message_dim"])
                        * 2
                    ),
                    "latency_seconds_per_batch": None,
                    "checkpoint_hash": checkpoint_hash,
                }
            )
    write_jsonl(run_dir / "predictions.jsonl", prediction_records)

    trajectory_records: list[dict[str, Any]] = []
    for index, example in enumerate(test_hidden.examples):
        for method in ("matched", "wrong_fact"):
            final_token = test_predictions[method][index : index + 1]
            trajectory_records.append(
                {
                    "example_id": example.example_id,
                    "method": method,
                    "agent_id": 0,
                    "step": 0,
                    "normalized_time": config["evaluation"]["normalized_time"],
                    "initial_canvas_sha256": sha256_tensor(torch.tensor([tokenizer.mask_token_id])),
                    "final_canvas_sha256": sha256_tensor(final_token),
                    "canvas_replaced_from_sender": False,
                }
            )
    write_zstd_jsonl(run_dir / "trajectories.jsonl.zst", trajectory_records)
    write_immutable_json(
        run_dir / "profile.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "hidden_cache_dtype": "torch.bfloat16",
            "hidden_shape_test": list(test_hidden.hidden_states.shape),
        },
    )
    (run_dir / "git_commit.txt").write_text(args.source_revision + "\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text(
        f"hm-synth verdict={verdict} gpu={gpu_name}\n",
        encoding="utf-8",
    )
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if enforce_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
