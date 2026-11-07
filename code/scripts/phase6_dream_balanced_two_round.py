"""Train and causally audit a two-round Dream rendezvous prefix protocol."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import dllm
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from phase5_hm_synth_tiny import (
    HiddenSplit,
    TokenizedSplit,
    different_target_derangement,
    model_weight_manifest,
    paired_bootstrap_interval,
    precompute_hidden,
    split_digest,
    tokenize_split,
)
from phase6_dream_input_prefix import (
    assert_no_message_parity,
    decoder_hidden,
    embedding_rms,
    insert_prefix_embeddings,
    receiver_logits,
    sender_metadata,
)
from phase6_dream_lora_input_prefix import load_adapter_checkpoint
from torch import nn

from hidden_messages.adaptation import attach_lora_to_decoder_layers, lora_named_parameters
from hidden_messages.communication import InputPrefixChannel, batch_message_variance_loss
from hidden_messages.datasets.hm_synth import HMSynthExample, generate_rendezvous_lookup_pair
from hidden_messages.utils.checksums import sha256_file
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json
from hidden_messages.utils.reproducibility import require_authorized_cuda, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="H100")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("balanced two-round config must be a mapping")
    return payload


def build_split(
    *,
    pair_count: int,
    seed_start: int,
    slot_values: tuple[str, ...],
    answer_values: tuple[str, ...],
) -> list[HMSynthExample]:
    examples: list[HMSynthExample] = []
    for pair_index in range(pair_count):
        factual, counterfactual = generate_rendezvous_lookup_pair(
            seed_start + pair_index,
            slot_values=slot_values,
            answer_values=answer_values,
        )
        if factual.private_contexts[0] != counterfactual.private_contexts[0]:
            raise AssertionError("counterfactual receiver context changed")
        if factual.answer == counterfactual.answer:
            raise AssertionError("counterfactual target did not change")
        examples.extend((factual, counterfactual))
    return examples


def teacher_table(
    tokenizer: Any,
    *,
    labels: tuple[str, ...],
    template: str,
    message_slots: int,
) -> dict[str, list[int]]:
    table: dict[str, list[int]] = {}
    for label in labels:
        token_ids = [
            int(value)
            for value in tokenizer.encode(template.format(label=label), add_special_tokens=False)
        ]
        if len(token_ids) != message_slots:
            raise RuntimeError(
                f"teacher phrase for {label!r} has {len(token_ids)} tokens, "
                f"expected {message_slots}"
            )
        table[label] = token_ids
    return table


def strip_prefix_hidden(
    hidden: torch.Tensor,
    insertion_positions: torch.Tensor,
    *,
    prefix_length: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row_index in range(hidden.shape[0]):
        insertion = int(insertion_positions[row_index].item())
        rows.append(
            torch.cat(
                (
                    hidden[row_index, :insertion],
                    hidden[row_index, insertion + prefix_length :],
                ),
                dim=0,
            )
        )
    return torch.stack(rows)


def augmented_forward(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_positions: torch.Tensor,
    assistant_prefix_tokens: int,
    output_position_offset: int,
    prefix_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    insertion_positions = answer_positions - assistant_prefix_tokens
    output_positions = answer_positions + output_position_offset
    augmented, augmented_mask, shifted_positions = insert_prefix_embeddings(
        input_embedding(input_ids),
        attention_mask,
        insertion_positions,
        output_positions,
        prefix_embeddings,
    )
    hidden = decoder_hidden(decoder, augmented, augmented_mask)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    logits = output_head(hidden[rows, shifted_positions])
    stripped = strip_prefix_hidden(
        hidden,
        insertion_positions,
        prefix_length=prefix_embeddings.shape[1],
    )
    return logits, stripped


def cosine_factor(step_index: int, *, steps: int, warmup_steps: int) -> float:
    if step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    progress = (step_index - warmup_steps) / max(steps - warmup_steps, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def make_channel(
    model_dim: int,
    input_rms: float,
    channel_config: dict[str, Any],
    *,
    device: torch.device,
) -> InputPrefixChannel:
    return InputPrefixChannel(
        model_dim,
        input_embedding_rms=input_rms,
        message_slots=int(channel_config["message_slots"]),
        message_dim=int(channel_config["message_dim"]),
        width=int(channel_config["width"]),
        num_heads=int(channel_config["num_heads"]),
        max_agents=int(channel_config["max_agents"]),
    ).to(device)


def encode_batch(
    channel: InputPrefixChannel,
    hidden: torch.Tensor,
    attention: torch.Tensor,
    segments: torch.Tensor,
    *,
    normalized_time: float,
    sender_identity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.full((hidden.shape[0],), normalized_time, device=hidden.device)
    uncertainty, sender_ids = sender_metadata(
        hidden,
        normalized_time=times,
        sender_identity=sender_identity,
    )
    return channel(
        hidden,
        segment_ids=segments,
        normalized_time=times,
        sender_ids=sender_ids,
        uncertainty=uncertainty,
        attention_mask=attention,
    )


def train_round1(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    slot_table: dict[str, list[int]],
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    steps = int(training["round1_steps"])
    batch_size = int(training["batch_size"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training["round1_seed"]))
    permutation = torch.randperm(len(tokenized.examples), generator=generator)
    cursor = 0
    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = max(1, round(steps * float(training["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: cosine_factor(index, steps=steps, warmup_steps=warmup_steps),
    )
    records: list[dict[str, float | int]] = []
    channel.train()
    for step in range(steps):
        if cursor + batch_size > len(tokenized.examples):
            permutation = torch.randperm(len(tokenized.examples), generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        receiver_hidden = hidden_split.hidden_states.index_select(0, indices)[:, 0].to(device)
        receiver_attention = hidden_split.attention_mask.index_select(0, indices)[:, 0].to(device)
        receiver_segments = hidden_split.segment_ids.index_select(0, indices)[:, 0].to(device)
        sender_ids = tokenized.input_ids.index_select(0, indices)[:, 1].to(device)
        sender_attention = tokenized.attention_mask.index_select(0, indices)[:, 1].to(device)
        sender_positions = tokenized.answer_positions.index_select(0, indices)[:, 1].to(device)
        targets = tokenized.target_ids.index_select(0, indices).to(device)
        slot_labels = [
            str(tokenized.examples[int(index)].metadata["requested_slot"])
            for index in indices.tolist()
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages, prefixes = encode_batch(
                channel,
                receiver_hidden,
                receiver_attention,
                receiver_segments,
                normalized_time=float(training["round1_normalized_time"]),
                sender_identity=0,
            )
            logits, _ = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=sender_ids,
                attention_mask=sender_attention,
                answer_positions=sender_positions,
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=int(config["model"]["output_position_offset"]),
                prefix_embeddings=prefixes,
            )
            teacher_ids = torch.tensor(
                [slot_table[label] for label in slot_labels], dtype=torch.long, device=device
            )
            with torch.no_grad():
                target_prefixes = input_embedding(teacher_ids)
            alignment_loss = (
                1.0
                - nn.functional.cosine_similarity(
                    prefixes.float(), target_prefixes.float(), dim=-1
                ).mean()
            )
            task_loss = nn.functional.cross_entropy(logits.float(), targets)
            variance_loss, message_std = batch_message_variance_loss(
                [messages], target_std=float(training["target_message_std"])
            )
            loss = (
                task_loss
                + float(training["lambda_prefix_alignment"]) * alignment_loss
                + float(training["lambda_message_variance"]) * variance_loss
            )
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            channel.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "task_loss": float(task_loss.item()),
                    "prefix_alignment_loss": float(alignment_loss.item()),
                    "message_variance_loss": float(variance_loss.item()),
                    "message_std": float(message_std.item()),
                    "batch_accuracy": float((logits.argmax(dim=-1) == targets).float().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    channel.eval()
    return records


@torch.inference_mode()
def encode_all(
    channel: InputPrefixChannel,
    hidden: torch.Tensor,
    attention: torch.Tensor,
    segments: torch.Tensor,
    *,
    normalized_time: float,
    sender_identity: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    channel.eval()
    for start in range(0, hidden.shape[0], batch_size):
        stop = min(start + batch_size, hidden.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages, _ = encode_batch(
                channel,
                hidden[start:stop].to(device),
                attention[start:stop].to(device),
                segments[start:stop].to(device),
                normalized_time=normalized_time,
                sender_identity=sender_identity,
            )
        outputs.append(messages.to(device="cpu", dtype=torch.bfloat16))
    return torch.cat(outputs)


@torch.inference_mode()
def sender_after_slot(
    slot_channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    message_order: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    evaluation_batch = int(config["evaluation"]["batch_size"])
    training = config["training"]
    all_messages = encode_all(
        slot_channel,
        hidden_split.hidden_states[:, 0],
        hidden_split.attention_mask[:, 0],
        hidden_split.segment_ids[:, 0],
        normalized_time=float(training["round1_normalized_time"]),
        sender_identity=0,
        batch_size=evaluation_batch,
        device=device,
    )
    if message_order is not None:
        all_messages = all_messages.index_select(0, message_order)
    hidden_outputs: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokenized.examples), evaluation_batch):
        stop = min(start + evaluation_batch, len(tokenized.examples))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = slot_channel.project(all_messages[start:stop].to(device))
            logits, hidden = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=tokenized.input_ids[start:stop, 1].to(device),
                attention_mask=tokenized.attention_mask[start:stop, 1].to(device),
                answer_positions=tokenized.answer_positions[start:stop, 1].to(device),
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=int(config["model"]["output_position_offset"]),
                prefix_embeddings=prefixes,
            )
        hidden_outputs.append(hidden.to(device="cpu", dtype=torch.bfloat16))
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(hidden_outputs), torch.cat(predictions), all_messages


def train_round2(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    sender_hidden: torch.Tensor,
    code_table: dict[str, list[int]],
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    steps = int(training["round2_steps"])
    batch_size = int(training["batch_size"])
    if batch_size % 2:
        raise ValueError("round2 batch size must preserve aligned pairs")
    pair_batch_size = batch_size // 2
    pair_count = len(tokenized.examples) // 2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training["round2_seed"]))
    permutation = torch.randperm(pair_count, generator=generator)
    cursor = 0
    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = max(1, round(steps * float(training["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: cosine_factor(index, steps=steps, warmup_steps=warmup_steps),
    )
    records: list[dict[str, float | int]] = []
    channel.train()
    for step in range(steps):
        if cursor + pair_batch_size > pair_count:
            permutation = torch.randperm(pair_count, generator=generator)
            cursor = 0
        selected_pairs = permutation[cursor : cursor + pair_batch_size]
        cursor += pair_batch_size
        indices = torch.stack((selected_pairs * 2, selected_pairs * 2 + 1), dim=1).reshape(-1)
        selected_hidden = sender_hidden.index_select(0, indices).to(device)
        sender_attention = tokenized.attention_mask.index_select(0, indices)[:, 1].to(device)
        sender_segments = tokenized.segment_ids.index_select(0, indices)[:, 1].to(device)
        receiver_ids = tokenized.input_ids.index_select(0, indices)[:, 0].to(device)
        receiver_attention = tokenized.attention_mask.index_select(0, indices)[:, 0].to(device)
        receiver_positions = tokenized.answer_positions.index_select(0, indices)[:, 0].to(device)
        targets = tokenized.target_ids.index_select(0, indices).to(device)
        answer_labels = [tokenized.examples[int(index)].answer for index in indices.tolist()]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages, prefixes = encode_batch(
                channel,
                selected_hidden,
                sender_attention,
                sender_segments,
                normalized_time=float(training["round2_normalized_time"]),
                sender_identity=1,
            )
            logits, _ = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=receiver_ids,
                attention_mask=receiver_attention,
                answer_positions=receiver_positions,
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=int(config["model"]["output_position_offset"]),
                prefix_embeddings=prefixes,
            )
            task_loss = nn.functional.cross_entropy(logits.float(), targets)
            teacher_ids = torch.tensor(
                [code_table[label] for label in answer_labels], dtype=torch.long, device=device
            )
            with torch.no_grad():
                target_prefixes = input_embedding(teacher_ids)
            alignment_loss = (
                1.0
                - nn.functional.cosine_similarity(
                    prefixes.float(), target_prefixes.float(), dim=-1
                ).mean()
            )
            pair_order = torch.arange(batch_size, device=device).bitwise_xor(1)
            wrong_prefixes = channel.project(messages.index_select(0, pair_order))
            wrong_logits, _ = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=receiver_ids,
                attention_mask=receiver_attention,
                answer_positions=receiver_positions,
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=int(config["model"]["output_position_offset"]),
                prefix_embeddings=wrong_prefixes,
            )
            matched_losses = nn.functional.cross_entropy(logits.float(), targets, reduction="none")
            wrong_losses = nn.functional.cross_entropy(
                wrong_logits.float(), targets, reduction="none"
            )
            pair_loss = nn.functional.relu(
                float(training["pair_margin"]) + matched_losses - wrong_losses
            ).mean()
            variance_loss, message_std = batch_message_variance_loss(
                [messages], target_std=float(training["target_message_std"])
            )
            loss = (
                task_loss
                + float(training["lambda_pair"]) * pair_loss
                + float(training["lambda_prefix_alignment"]) * alignment_loss
                + float(training["lambda_message_variance"]) * variance_loss
            )
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            channel.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "task_loss": float(task_loss.item()),
                    "pair_loss": float(pair_loss.item()),
                    "prefix_alignment_loss": float(alignment_loss.item()),
                    "message_variance_loss": float(variance_loss.item()),
                    "message_std": float(message_std.item()),
                    "batch_accuracy": float((logits.argmax(dim=-1) == targets).float().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    channel.eval()
    return records


@torch.inference_mode()
def predict_no_prefix(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    *,
    agent_id: int,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(config["evaluation"]["batch_size"])
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = receiver_logits(
                decoder,
                input_embedding,
                output_head,
                input_ids=tokenized.input_ids[start:stop, agent_id].to(device),
                attention_mask=tokenized.attention_mask[start:stop, agent_id].to(device),
                output_positions=(
                    tokenized.answer_positions[start:stop, agent_id].to(device)
                    + int(config["model"]["output_position_offset"])
                ),
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def receiver_predictions(
    code_channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    messages: torch.Tensor,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(config["evaluation"]["batch_size"])
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = code_channel.project(messages[start:stop].to(device))
            logits, _ = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=tokenized.input_ids[start:stop, 0].to(device),
                attention_mask=tokenized.attention_mask[start:stop, 0].to(device),
                answer_positions=tokenized.answer_positions[start:stop, 0].to(device),
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=int(config["model"]["output_position_offset"]),
                prefix_embeddings=prefixes,
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


def wrong_slot_targets(
    examples: list[HMSynthExample],
    slot_order: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    target_ids: list[int] = []
    for index, source_index in enumerate(slot_order.tolist()):
        wrong_slot = str(examples[source_index].metadata["requested_slot"])
        slots = [str(value) for value in cast(list[str], examples[index].metadata["slot_values"])]
        if examples[index].variant == "factual":
            table_key = "factual_codes"
        else:
            table_key = "counterfactual_codes"
        codes = [str(value) for value in cast(list[str], examples[index].metadata[table_key])]
        expected = codes[slots.index(wrong_slot)]
        encoded = tokenizer.encode(expected, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError("wrong-slot code is not one token")
        target_ids.append(int(encoded[0]))
    return torch.tensor(target_ids, dtype=torch.long)


@torch.inference_mode()
def evaluate_split(
    slot_channel: InputPrefixChannel,
    code_channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    slot_labels = torch.tensor(
        [
            tokenizer.encode(str(example.metadata["requested_slot"]), add_special_tokens=False)[0]
            for example in tokenized.examples
        ],
        dtype=torch.long,
    )
    slot_order = different_target_derangement(slot_labels)
    matched_sender_hidden, sender_matched, _ = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        tokenized,
        hidden_split,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        message_order=None,
        device=device,
    )
    wrong_sender_hidden, sender_wrong_slot, _ = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        tokenized,
        hidden_split,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        message_order=slot_order,
        device=device,
    )
    sender_no_slot = predict_no_prefix(
        decoder,
        input_embedding,
        output_head,
        tokenized,
        agent_id=1,
        config=config,
        device=device,
    )
    no_message = predict_no_prefix(
        decoder,
        input_embedding,
        output_head,
        tokenized,
        agent_id=0,
        config=config,
        device=device,
    )
    matched_messages = encode_all(
        code_channel,
        matched_sender_hidden,
        tokenized.attention_mask[:, 1],
        tokenized.segment_ids[:, 1],
        normalized_time=float(config["training"]["round2_normalized_time"]),
        sender_identity=1,
        batch_size=batch_size,
        device=device,
    )
    final_only_messages = encode_all(
        code_channel,
        hidden_split.hidden_states[:, 1],
        hidden_split.attention_mask[:, 1],
        hidden_split.segment_ids[:, 1],
        normalized_time=float(config["training"]["round2_normalized_time"]),
        sender_identity=1,
        batch_size=batch_size,
        device=device,
    )
    wrong_slot_messages = encode_all(
        code_channel,
        wrong_sender_hidden,
        tokenized.attention_mask[:, 1],
        tokenized.segment_ids[:, 1],
        normalized_time=float(config["training"]["round2_normalized_time"]),
        sender_identity=1,
        batch_size=batch_size,
        device=device,
    )
    pair_order = torch.arange(len(tokenized.examples)).bitwise_xor(1)
    deranged_order = different_target_derangement(tokenized.target_ids)
    incoming = {
        "matched": matched_messages,
        "final_only": final_only_messages,
        "wrong_fact": matched_messages.index_select(0, pair_order),
        "deranged": matched_messages.index_select(0, deranged_order),
        "wrong_slot": wrong_slot_messages,
        "zero": torch.zeros_like(matched_messages),
    }
    predictions = {"no_message": no_message}
    for method, messages in incoming.items():
        predictions[method] = receiver_predictions(
            code_channel,
            decoder,
            input_embedding,
            output_head,
            tokenized,
            messages,
            config,
            assistant_prefix_tokens=assistant_prefix_tokens,
            device=device,
        )
    targets = tokenized.target_ids
    correctness = {method: values == targets for method, values in predictions.items()}
    accuracy = {
        method: float(values.float().mean().item()) for method, values in correctness.items()
    }
    matched_final_ci = paired_bootstrap_interval(
        correctness["matched"],
        correctness["final_only"],
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    matched_wrong_ci = paired_bootstrap_interval(
        correctness["matched"],
        correctness["wrong_fact"],
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]) + 1,
    )
    paired_targets = targets.index_select(0, pair_order)
    wrong_slot_expected = wrong_slot_targets(tokenized.examples, slot_order, tokenizer)
    metrics = {
        "examples": len(tokenized.examples),
        "pairs": len(tokenized.examples) // 2,
        "accuracy": accuracy,
        "sender_bridge_accuracy": {
            "matched_slot": float((sender_matched == targets).float().mean().item()),
            "no_slot": float((sender_no_slot == targets).float().mean().item()),
            "wrong_slot_original_target": float(
                (sender_wrong_slot == targets).float().mean().item()
            ),
            "wrong_slot_target": float(
                (sender_wrong_slot == wrong_slot_expected).float().mean().item()
            ),
        },
        "matched_minus_final_only": accuracy["matched"] - accuracy["final_only"],
        "matched_minus_final_only_pair_bootstrap_ci95": list(matched_final_ci),
        "matched_minus_wrong_fact": accuracy["matched"] - accuracy["wrong_fact"],
        "matched_minus_wrong_fact_pair_bootstrap_ci95": list(matched_wrong_ci),
        "wrong_fact_target_rate": float(
            (predictions["wrong_fact"] == paired_targets).float().mean().item()
        ),
        "wrong_slot_target_rate": float(
            (predictions["wrong_slot"] == wrong_slot_expected).float().mean().item()
        ),
        "wrong_fact_answer_flip_rate": float(
            (predictions["wrong_fact"] != predictions["matched"]).float().mean().item()
        ),
        "matched_pair_consistency": float(
            correctness["matched"].reshape(-1, 2).all(dim=1).float().mean().item()
        ),
        "derangement_has_zero_target_matches": not bool(
            (targets.index_select(0, deranged_order) == targets).any()
        ),
        "round1_message_std": float(
            encode_all(
                slot_channel,
                hidden_split.hidden_states[:, 0],
                hidden_split.attention_mask[:, 0],
                hidden_split.segment_ids[:, 0],
                normalized_time=float(config["training"]["round1_normalized_time"]),
                sender_identity=0,
                batch_size=batch_size,
                device=device,
            )
            .float()
            .std(dim=0, unbiased=False)
            .mean()
            .item()
        ),
        "round2_message_std": float(
            matched_messages.float().std(dim=0, unbiased=False).mean().item()
        ),
    }
    predictions["sender_matched_slot"] = sender_matched
    predictions["sender_no_slot"] = sender_no_slot
    predictions["sender_wrong_slot"] = sender_wrong_slot
    return metrics, predictions


def gate_passes(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    accuracy = metrics["accuracy"]
    matched_final_ci = metrics["matched_minus_final_only_pair_bootstrap_ci95"]
    matched_wrong_ci = metrics["matched_minus_wrong_fact_pair_bootstrap_ci95"]
    sender_bridge = metrics["sender_bridge_accuracy"]
    return bool(
        sender_bridge["matched_slot"] >= float(gate["sender_matched_min"])
        and sender_bridge["no_slot"] <= float(gate["sender_no_slot_max"])
        and accuracy["matched"] >= float(gate["matched_accuracy_min"])
        and accuracy["final_only"] <= float(gate["final_only_accuracy_max"])
        and metrics["matched_minus_final_only"] >= float(gate["matched_minus_final_only_min"])
        and matched_final_ci[0] > 0.0
        and metrics["matched_minus_wrong_fact"] >= float(gate["matched_minus_wrong_fact_min"])
        and matched_wrong_ci[0] > 0.0
        and metrics["wrong_fact_target_rate"] >= float(gate["wrong_fact_target_rate_min"])
        and metrics["wrong_slot_target_rate"] >= float(gate["wrong_slot_target_rate_min"])
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    gpu_name = require_authorized_cuda(args.expected_gpu)
    seed_everything(int(config["training"]["round1_seed"]))
    device = torch.device("cuda")
    run_dir = create_run_directory(args.run_root, args.run_id)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    shutil.copyfile(args.source_manifest, run_dir / "source-manifest.sha256")

    model_config = config["model"]
    cache_key = "models--" + str(model_config["id"]).replace("/", "--")
    snapshot = args.model_cache / cache_key / "snapshots" / str(model_config["revision"])
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot),
        dtype=str(model_config["dtype"]),
        attn_implementation=str(model_config["requested_attention_implementation"]),
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    decoder = model.get_decoder()
    input_embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    if not isinstance(decoder, nn.Module) or not isinstance(output_head, nn.Module):
        raise TypeError("Dream decoder or output head is not a torch module")
    if not isinstance(input_embedding, nn.Embedding):
        raise TypeError("Dream input embedding is not torch Embedding")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora_config = config["lora"]
    attachments = attach_lora_to_decoder_layers(
        decoder,
        layer_indices=[int(value) for value in lora_config["layer_indices"]],
        target_modules=[str(value) for value in lora_config["target_modules"]],
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
    )
    source_lora = (
        args.run_root
        / str(lora_config["checkpoint_run_id"])
        / str(lora_config["checkpoint_filename"])
    )
    if sha256_file(source_lora) != str(lora_config["checkpoint_sha256"]):
        raise RuntimeError("sealed LoRA checkpoint differs from two-round config")
    lora_copy = run_dir / "lora_adapter.pt"
    shutil.copyfile(source_lora, lora_copy)
    load_adapter_checkpoint(model, lora_copy)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for _, parameter in lora_named_parameters(model)):
        raise AssertionError("LoRA unexpectedly remains trainable")

    dataset = config["dataset"]
    slot_values = tuple(str(value) for value in dataset["slot_values"])
    answer_values = tuple(str(value) for value in dataset["answer_values"])
    common_split = {
        "slot_values": slot_values,
        "answer_values": answer_values,
    }
    train_examples = build_split(
        pair_count=int(dataset["train_pairs"]),
        seed_start=int(dataset["train_seed_start"]),
        **common_split,
    )
    validation_examples = build_split(
        pair_count=int(dataset["validation_pairs"]),
        seed_start=int(dataset["validation_seed_start"]),
        **common_split,
    )
    common_tokenize = {
        "tokenizer": tokenizer,
        "max_sequence_length": int(dataset["max_sequence_length"]),
        "assistant_prefix": str(dataset["assistant_prefix"]),
    }
    train_tokenized = tokenize_split(train_examples, **common_tokenize)
    validation_tokenized = tokenize_split(validation_examples, **common_tokenize)
    assistant_prefix_tokens = len(
        tokenizer.encode(str(dataset["assistant_prefix"]), add_special_tokens=False)
    )
    parity = assert_no_message_parity(
        model,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        output_position_offset=int(model_config["output_position_offset"]),
        device=device,
    )
    train_hidden = precompute_hidden(
        train_tokenized,
        model,
        output_head,
        batch_size=int(config["training"]["encoding_batch_size"]),
        device=device,
    )
    validation_hidden = precompute_hidden(
        validation_tokenized,
        model,
        output_head,
        batch_size=int(config["training"]["encoding_batch_size"]),
        device=device,
    )
    channel_config = config["channel"]
    input_rms = embedding_rms(input_embedding)
    slot_channel = make_channel(
        int(model.config.hidden_size), input_rms, channel_config, device=device
    )
    code_channel = make_channel(
        int(model.config.hidden_size), input_rms, channel_config, device=device
    )
    slot_table = teacher_table(
        tokenizer,
        labels=slot_values,
        template=str(channel_config["round1_alignment_template"]),
        message_slots=int(channel_config["message_slots"]),
    )
    code_table = teacher_table(
        tokenizer,
        labels=answer_values,
        template=str(channel_config["round2_alignment_template"]),
        message_slots=int(channel_config["message_slots"]),
    )
    round1_log = train_round1(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        train_hidden,
        slot_table,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    matched_train_sender_hidden, _, _ = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        train_hidden,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        message_order=None,
        device=device,
    )
    round2_log = train_round2(
        code_channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        matched_train_sender_hidden,
        code_table,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    slot_path = run_dir / "round1_slot_channel.pt"
    code_path = run_dir / "round2_code_channel.pt"
    torch.save(slot_channel.state_dict(), slot_path)
    torch.save(code_channel.state_dict(), code_path)
    del matched_train_sender_hidden, train_hidden
    gc.collect()
    torch.cuda.empty_cache()

    validation_metrics, validation_predictions = evaluate_split(
        slot_channel,
        code_channel,
        decoder,
        input_embedding,
        output_head,
        tokenizer,
        validation_tokenized,
        validation_hidden,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    validation_passed = gate_passes(validation_metrics, config["evaluation"]["gate"])
    engineering_smoke = config["stage"] == "engineering_smoke"
    open_test = validation_passed and not engineering_smoke
    test_metrics: dict[str, Any] | None = None
    test_predictions: dict[str, torch.Tensor] | None = None
    test_digest: str | None = None
    test_hidden_shape: list[int] | None = None
    test_tokenized: TokenizedSplit | None = None
    if open_test:
        test_examples = build_split(
            pair_count=int(dataset["development_test_pairs"]),
            seed_start=int(dataset["development_test_seed_start"]),
            **common_split,
        )
        test_digest = split_digest(test_examples)
        test_tokenized = tokenize_split(test_examples, **common_tokenize)
        test_hidden = precompute_hidden(
            test_tokenized,
            model,
            output_head,
            batch_size=int(config["training"]["encoding_batch_size"]),
            device=device,
        )
        test_hidden_shape = list(test_hidden.hidden_states.shape)
        test_metrics, test_predictions = evaluate_split(
            slot_channel,
            code_channel,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            test_tokenized,
            test_hidden,
            config,
            assistant_prefix_tokens=assistant_prefix_tokens,
            device=device,
        )
    pilot_passed = bool(
        test_metrics is not None and gate_passes(test_metrics, config["evaluation"]["gate"])
    )
    if engineering_smoke:
        verdict = "engineering_smoke_completed"
    elif not validation_passed:
        verdict = "stop_validation"
    elif pilot_passed:
        verdict = "go_multiseed_two_round_confirmation"
    else:
        verdict = "stop_development_test"

    model_digest, weight_records = model_weight_manifest(snapshot)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Dream decoder layers are not a ModuleList")
    write_immutable_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "protocol_id": config["protocol_id"],
            "stage": config["stage"],
            "source_revision": args.source_revision,
            "status": "completed",
        },
    )
    write_immutable_json(
        run_dir / "design.json",
        {
            "mechanism": "two_directional_static_input_prefix_rounds",
            "round1": "receiver_requested_slot_to_sender",
            "round2": "conditioned_sender_code_to_receiver",
            "directional_weights_shared": False,
            "architecture_shared": True,
            "round1_alignment_token_ids": slot_table,
            "round2_alignment_token_ids": code_table,
            "target_ids_inference_payload": False,
            "slot_ids_inference_payload": False,
            "no_message_parity": parity,
            "pretrained_backbone_frozen": True,
            "language_model_head_frozen": True,
            "task_lora_frozen": True,
            "receiver_canvas_replaced": False,
            "development_test_opened_only_after_validation_pass": open_test,
            "gate": config["evaluation"]["gate"],
            "non_claim": config["non_claim"],
        },
    )
    write_immutable_json(
        run_dir / "dataset_manifest.json",
        {
            "family": dataset["family"],
            "train": {
                "pairs": int(dataset["train_pairs"]),
                "seed_start": int(dataset["train_seed_start"]),
                "sha256": split_digest(train_examples),
            },
            "validation": {
                "pairs": int(dataset["validation_pairs"]),
                "seed_start": int(dataset["validation_seed_start"]),
                "sha256": split_digest(validation_examples),
            },
            "development_test": {
                "pairs": int(dataset["development_test_pairs"]),
                "seed_start": int(dataset["development_test_seed_start"]),
                "materialized": open_test,
                "sha256": test_digest,
            },
            "seed_ranges_disjoint": True,
            "counterfactual_pairs_aligned": True,
        },
    )
    write_immutable_json(
        run_dir / "model_manifest.json",
        {
            "model_id": model_config["id"],
            "revision": model_config["revision"],
            "weight_manifest_sha256": model_digest,
            "weights": weight_records,
            "lora_checkpoint_sha256": sha256_file(lora_copy),
            "lora_frozen": True,
            "lora_attachments": [attachment.qualified_name for attachment in attachments],
            "round1_channel_sha256": sha256_file(slot_path),
            "round2_channel_sha256": sha256_file(code_path),
            "effective_attention_classes": sorted(
                {type(getattr(layer, "self_attn", None)).__name__ for layer in layers}
            ),
            "pretrained_parameters_frozen": True,
            "language_model_head_frozen": True,
        },
    )
    write_immutable_json(
        run_dir / "seed_manifest.json",
        {
            "round1_training_seed": config["training"]["round1_seed"],
            "round2_training_seed": config["training"]["round2_seed"],
            "train_generation_seed_start": dataset["train_seed_start"],
            "validation_generation_seed_start": dataset["validation_seed_start"],
            "development_test_generation_seed_start": dataset["development_test_seed_start"],
            "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
        },
    )
    write_immutable_json(
        run_dir / "metrics.json",
        {
            "verdict": verdict,
            "validation_gate_passed": validation_passed,
            "development_pilot_gate_passed": pilot_passed,
            "gate": config["evaluation"]["gate"],
            "validation": validation_metrics,
            "development_test": test_metrics,
            "canvas_assignment_count": 0,
            "non_claim": config["non_claim"],
        },
    )
    write_jsonl(run_dir / "round1_training_log.jsonl", round1_log)
    write_jsonl(run_dir / "round2_training_log.jsonl", round2_log)
    endpoint_name = "development_test" if open_test else "validation"
    endpoint_tokenized = test_tokenized if open_test else validation_tokenized
    endpoint_predictions = test_predictions if open_test else validation_predictions
    if endpoint_tokenized is None or endpoint_predictions is None:
        raise AssertionError("endpoint artifacts are missing")
    prediction_records: list[dict[str, Any]] = []
    for index, example in enumerate(endpoint_tokenized.examples):
        for method, values in endpoint_predictions.items():
            prediction_records.append(
                {
                    "endpoint": endpoint_name,
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "variant": example.variant,
                    "method": method,
                    "prediction_token_id": int(values[index]),
                    "target": example.answer,
                    "target_token_id": int(endpoint_tokenized.target_ids[index]),
                    "correct": bool(values[index] == endpoint_tokenized.target_ids[index]),
                    "designated_receiver": 0,
                }
            )
    write_jsonl(run_dir / "predictions.jsonl", prediction_records)
    write_immutable_json(
        run_dir / "environment.json",
        {
            "pod": os.environ.get("HOSTNAME"),
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    write_immutable_json(
        run_dir / "profile.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "validation_hidden_shape": list(validation_hidden.hidden_states.shape),
            "development_test_hidden_shape": test_hidden_shape,
            "message_bytes_per_direction": int(channel_config["message_slots"])
            * int(channel_config["message_dim"])
            * 2,
        },
    )
    write_text_exclusive(run_dir / "git_commit.txt", args.source_revision + "\n")
    stdout = json.dumps(
        {
            "validation_gate_passed": validation_passed,
            "development_pilot_gate_passed": pilot_passed,
            "verdict": verdict,
        },
        sort_keys=True,
    )
    write_text_exclusive(run_dir / "stdout.log", stdout + "\n")
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    write_text_exclusive(run_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    print(stdout)


if __name__ == "__main__":
    main()
