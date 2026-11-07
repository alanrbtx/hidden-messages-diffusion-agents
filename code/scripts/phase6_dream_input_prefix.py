"""Train and causally audit a Dream input-embedding-prefix communication channel."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import dllm
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
import zstandard
from phase5_hm_synth_tiny import (
    HiddenSplit,
    TokenizedSplit,
    build_split,
    decode_token,
    different_target_derangement,
    gate_passes,
    model_weight_manifest,
    paired_bootstrap_interval,
    precompute_hidden,
    precompute_receiver_answer_hidden,
    split_digest,
    tokenize_split,
    union_context_examples,
)
from torch import nn

from hidden_messages.communication import (
    InputPrefixChannel,
    apply_message_intervention,
    batch_message_variance_loss,
)
from hidden_messages.contracts import MessageIntervention
from hidden_messages.utils.checksums import sha256_file, sha256_tensor
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
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("input-prefix config must be a mapping")
    return payload


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_zstd_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    compressor = zstandard.ZstdCompressor(level=9)
    with path.open("xb") as raw_handle, compressor.stream_writer(raw_handle) as handle:
        for record in records:
            handle.write((json.dumps(record, sort_keys=True) + "\n").encode())


@torch.inference_mode()
def embedding_rms(embedding: nn.Embedding, *, chunk_rows: int = 4096) -> float:
    total = torch.zeros((), dtype=torch.float64, device=embedding.weight.device)
    count = 0
    for start in range(0, embedding.weight.shape[0], chunk_rows):
        chunk = embedding.weight[start : start + chunk_rows]
        total += chunk.float().square().sum(dtype=torch.float64)
        count += chunk.numel()
    return float((total / count).sqrt().item())


def full_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, L]")
    return torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2),
        attention_mask.unsqueeze(1).unsqueeze(-1),
    )


def insert_prefix_embeddings(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    insertion_positions: torch.Tensor,
    output_positions: torch.Tensor,
    prefix_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, model_dim = token_embeddings.shape
    if prefix_embeddings.shape[:1] != (batch_size,):
        raise ValueError("prefix batch does not match receiver batch")
    if prefix_embeddings.shape[2] != model_dim:
        raise ValueError("prefix and receiver embedding dimensions differ")
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask must align with token_embeddings")
    if insertion_positions.shape != (batch_size,) or output_positions.shape != (batch_size,):
        raise ValueError("insertion and output positions must have shape [B]")
    prefix_length = prefix_embeddings.shape[1]
    embedded_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    shifted_positions: list[int] = []
    for row_index in range(batch_size):
        insertion = int(insertion_positions[row_index].item())
        output = int(output_positions[row_index].item())
        if not 0 <= insertion <= sequence_length:
            raise ValueError("prefix insertion position is outside the receiver sequence")
        if not 0 <= output < sequence_length:
            raise ValueError("receiver output position is outside the receiver sequence")
        embedded_rows.append(
            torch.cat(
                (
                    token_embeddings[row_index, :insertion],
                    prefix_embeddings[row_index],
                    token_embeddings[row_index, insertion:],
                ),
                dim=0,
            )
        )
        prefix_mask = torch.ones(
            prefix_length,
            dtype=torch.bool,
            device=attention_mask.device,
        )
        mask_rows.append(
            torch.cat(
                (
                    attention_mask[row_index, :insertion],
                    prefix_mask,
                    attention_mask[row_index, insertion:],
                )
            )
        )
        shifted_positions.append(output + prefix_length if output >= insertion else output)
    return (
        torch.stack(embedded_rows),
        torch.stack(mask_rows),
        torch.tensor(shifted_positions, dtype=torch.long, device=output_positions.device),
    )


def decoder_hidden(
    decoder: nn.Module,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = decoder(
        inputs_embeds=embeddings,
        attention_mask=full_attention_mask(attention_mask),
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs[0]
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("Dream decoder did not return tensor hidden states")
    return hidden


def receiver_logits(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_positions: torch.Tensor,
    insertion_positions: torch.Tensor | None = None,
    prefix_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    token_embeddings = input_embedding(input_ids)
    if prefix_embeddings is not None:
        if insertion_positions is None:
            raise ValueError("insertion positions are required with prefix embeddings")
        token_embeddings, attention_mask, output_positions = insert_prefix_embeddings(
            token_embeddings,
            attention_mask,
            insertion_positions,
            output_positions,
            prefix_embeddings,
        )
    elif insertion_positions is not None:
        raise ValueError("insertion positions are invalid without prefix embeddings")
    hidden = decoder_hidden(decoder, token_embeddings, attention_mask)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    logits = output_head(hidden[rows, output_positions])
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Dream output head did not return tensor logits")
    return logits


@torch.inference_mode()
def assert_no_message_parity(
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    *,
    output_position_offset: int,
    device: torch.device,
) -> dict[str, float | bool]:
    input_ids = tokenized.input_ids[:2, 0].to(device)
    attention_mask = tokenized.attention_mask[:2, 0].to(device)
    output_positions = tokenized.answer_positions[:2, 0].to(device) + output_position_offset
    captured: list[torch.Tensor] = []

    def capture_hidden(_module: nn.Module, hook_inputs: tuple[Any, ...]) -> None:
        if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
            raise RuntimeError("Dream output head did not receive hidden states")
        captured.append(hook_inputs[0])

    handle = output_head.register_forward_pre_hook(capture_hidden)
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one parity hidden capture, observed {len(captured)}")
    candidate_hidden = decoder_hidden(decoder, input_embedding(input_ids), attention_mask)
    hidden_exact = torch.equal(captured[0], candidate_hidden)
    max_hidden_difference = float((captured[0].float() - candidate_hidden.float()).abs().max())
    rows = torch.arange(input_ids.shape[0], device=device)
    reference_logits = output_head(captured[0][rows, output_positions])
    candidate_logits = output_head(candidate_hidden[rows, output_positions])
    logits_exact = torch.equal(reference_logits, candidate_logits)
    argmax_exact = torch.equal(
        reference_logits.argmax(dim=-1),
        candidate_logits.argmax(dim=-1),
    )
    if not hidden_exact or not logits_exact or not argmax_exact:
        raise AssertionError("no-message decoder path is not exactly equal to frozen Dream")
    return {
        "hidden_exact": hidden_exact,
        "logits_exact": logits_exact,
        "argmax_exact": argmax_exact,
        "max_hidden_abs_difference": max_hidden_difference,
    }


def sender_metadata(
    hidden: torch.Tensor,
    *,
    normalized_time: torch.Tensor,
    sender_identity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, _ = hidden.shape
    uncertainty = torch.zeros(
        batch_size,
        sequence_length,
        3,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    sender_ids = torch.full(
        (batch_size,),
        sender_identity,
        dtype=torch.long,
        device=hidden.device,
    )
    if normalized_time.shape != (batch_size,):
        raise ValueError("normalized_time must have shape [B]")
    return uncertainty, sender_ids


def train_channel(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    config: dict[str, Any],
    *,
    teacher_hidden: torch.Tensor | None,
    assistant_prefix_tokens: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    if batch_size < 2 or batch_size % 2:
        raise ValueError("input-prefix training batch_size must be a positive even number")
    pair_batch_size = batch_size // 2
    pair_count = len(hidden_split.examples) // 2
    steps = int(training["steps"])
    normalized_time_value = float(training["normalized_time"])
    lambda_teacher = float(training.get("lambda_teacher", 0.0))
    lambda_pair = float(training.get("lambda_pair", 0.0))
    lambda_variance = float(training.get("lambda_message_variance", 0.0))
    pair_fraction = float(training.get("pair_fraction", 0.0))
    pair_margin = float(training.get("pair_margin", 0.1))
    target_message_std = float(training.get("target_message_std", 0.2))
    teacher_temperature = float(training.get("teacher_temperature", 1.0))
    output_offset = int(config["model"]["output_position_offset"])
    if lambda_teacher > 0 and teacher_hidden is None:
        raise ValueError("teacher_hidden is required when lambda_teacher is positive")
    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pair_permutation = torch.randperm(pair_count, generator=generator)
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
        if cursor + pair_batch_size > pair_count:
            pair_permutation = torch.randperm(pair_count, generator=generator)
            cursor = 0
        selected_pairs = pair_permutation[cursor : cursor + pair_batch_size]
        cursor += pair_batch_size
        indices = torch.stack((selected_pairs * 2, selected_pairs * 2 + 1), dim=1).reshape(-1)
        selected_hidden = hidden_split.hidden_states.index_select(0, indices).to(device)
        selected_attention = hidden_split.attention_mask.index_select(0, indices).to(device)
        selected_segments = hidden_split.segment_ids.index_select(0, indices).to(device)
        receiver_ids = tokenized.input_ids.index_select(0, indices)[:, 0].to(device)
        receiver_mask = tokenized.attention_mask.index_select(0, indices)[:, 0].to(device)
        answer_positions = tokenized.answer_positions.index_select(0, indices)[:, 0].to(device)
        output_positions = answer_positions + output_offset
        insertion_positions = answer_positions - assistant_prefix_tokens
        targets = hidden_split.target_ids.index_select(0, indices).to(device)
        selected_teacher = (
            teacher_hidden.index_select(0, indices).to(device)
            if teacher_hidden is not None
            else None
        )
        times = torch.full((batch_size,), normalized_time_value, device=device)
        uncertainty, sender_ids = sender_metadata(
            selected_hidden[:, 1],
            normalized_time=times,
            sender_identity=1,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages, prefixes = channel(
                selected_hidden[:, 1],
                segment_ids=selected_segments[:, 1],
                normalized_time=times,
                sender_ids=sender_ids,
                uncertainty=uncertainty,
                attention_mask=selected_attention[:, 1],
            )
            logits = receiver_logits(
                decoder,
                input_embedding,
                output_head,
                input_ids=receiver_ids,
                attention_mask=receiver_mask,
                output_positions=output_positions,
                insertion_positions=insertion_positions,
                prefix_embeddings=prefixes,
            )
            task_loss = nn.functional.cross_entropy(logits.float(), targets)
            teacher_loss = torch.zeros((), device=device)
            if selected_teacher is not None and lambda_teacher > 0:
                with torch.no_grad():
                    teacher_logits = output_head(selected_teacher).float()
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
                pair_examples = max(2, int(batch_size * pair_fraction) // 2 * 2)
                pair_examples = min(pair_examples, batch_size)
                counterfactual_order = torch.arange(pair_examples, device=device).bitwise_xor(1)
                counterfactual_prefixes = channel.project(
                    messages[:pair_examples].index_select(0, counterfactual_order)
                )
                counterfactual_logits = receiver_logits(
                    decoder,
                    input_embedding,
                    output_head,
                    input_ids=receiver_ids[:pair_examples],
                    attention_mask=receiver_mask[:pair_examples],
                    output_positions=output_positions[:pair_examples],
                    insertion_positions=insertion_positions[:pair_examples],
                    prefix_embeddings=counterfactual_prefixes,
                )
                matched_losses = nn.functional.cross_entropy(
                    logits[:pair_examples].float(),
                    targets[:pair_examples],
                    reduction="none",
                )
                counterfactual_losses = nn.functional.cross_entropy(
                    counterfactual_logits.float(),
                    targets[:pair_examples],
                    reduction="none",
                )
                pair_loss = nn.functional.relu(
                    pair_margin + matched_losses - counterfactual_losses
                ).mean()
            variance_loss, message_std = batch_message_variance_loss(
                [messages],
                target_std=target_message_std,
            )
            loss = (
                task_loss
                + lambda_teacher * teacher_loss
                + lambda_pair * pair_loss
                + lambda_variance * variance_loss
            )
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            channel.parameters(),
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "task_loss": float(task_loss.item()),
                    "teacher_loss": float(teacher_loss.item()),
                    "pair_loss": float(pair_loss.item()),
                    "message_variance_loss": float(variance_loss.item()),
                    "message_std": float(message_std.item()),
                    "prefix_rms": float(prefixes.float().square().mean().sqrt().item()),
                    "prefix_scale": float(channel.log_scale.detach().clamp(-2, 2).exp().item()),
                    "batch_accuracy": float((logits.argmax(dim=-1) == targets).float().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
        del selected_hidden, selected_attention, selected_segments, logits, loss
    return records


@torch.inference_mode()
def encode_all_messages(
    channel: InputPrefixChannel,
    split: HiddenSplit,
    *,
    source_agent: int,
    sender_identity: int,
    normalized_time: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    encoded_batches: list[torch.Tensor] = []
    channel.eval()
    for start in range(0, len(split.examples), batch_size):
        stop = min(start + batch_size, len(split.examples))
        hidden = split.hidden_states[start:stop, source_agent].to(device)
        attention = split.attention_mask[start:stop, source_agent].to(device)
        segments = split.segment_ids[start:stop, source_agent].to(device)
        times = torch.full((stop - start,), normalized_time, device=device)
        uncertainty, sender_ids = sender_metadata(
            hidden,
            normalized_time=times,
            sender_identity=sender_identity,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages = channel.encode(
                hidden,
                segment_ids=segments,
                normalized_time=times,
                sender_ids=sender_ids,
                uncertainty=uncertainty,
                attention_mask=attention,
            )
        encoded_batches.append(messages.float().cpu())
    return torch.cat(encoded_batches)


@torch.inference_mode()
def evaluate_no_message(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    *,
    output_position_offset: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        input_ids = tokenized.input_ids[start:stop, 0].to(device)
        attention_mask = tokenized.attention_mask[start:stop, 0].to(device)
        output_positions = (
            tokenized.answer_positions[start:stop, 0].to(device) + output_position_offset
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = receiver_logits(
                decoder,
                input_embedding,
                output_head,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_positions=output_positions,
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def evaluate_incoming(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    incoming_messages: torch.Tensor,
    *,
    assistant_prefix_tokens: int,
    output_position_offset: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    channel.eval()
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        input_ids = tokenized.input_ids[start:stop, 0].to(device)
        attention_mask = tokenized.attention_mask[start:stop, 0].to(device)
        answer_positions = tokenized.answer_positions[start:stop, 0].to(device)
        output_positions = answer_positions + output_position_offset
        insertion_positions = answer_positions - assistant_prefix_tokens
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = channel.project(incoming_messages[start:stop].to(device))
            logits = receiver_logits(
                decoder,
                input_embedding,
                output_head,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_positions=output_positions,
                insertion_positions=insertion_positions,
                prefix_embeddings=prefixes,
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


def evaluate_split(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    normalized_time = float(evaluation["normalized_time"])
    output_offset = int(config["model"]["output_position_offset"])
    matched_messages = encode_all_messages(
        channel,
        hidden_split,
        source_agent=1,
        sender_identity=1,
        normalized_time=normalized_time,
        batch_size=batch_size,
        device=device,
    )
    self_messages = encode_all_messages(
        channel,
        hidden_split,
        source_agent=0,
        sender_identity=1,
        normalized_time=normalized_time,
        batch_size=batch_size,
        device=device,
    )
    targets = hidden_split.target_ids
    pair_order = torch.arange(len(hidden_split.examples)).bitwise_xor(1)
    deranged_order = different_target_derangement(targets)
    wrapped = matched_messages[:, None, None]
    random_messages = apply_message_intervention(
        wrapped,
        MessageIntervention.RANDOM_MOMENT_MATCHED,
        seed=seed + 17,
    )[:, 0, 0]
    incoming_by_method = {
        "matched": matched_messages,
        "zero": torch.zeros_like(matched_messages),
        "random_moment_matched": random_messages,
        "deranged": matched_messages.index_select(0, deranged_order),
        "self": self_messages,
        "wrong_fact": matched_messages.index_select(0, pair_order),
    }
    predictions = {
        "no_message": evaluate_no_message(
            decoder,
            input_embedding,
            output_head,
            tokenized,
            output_position_offset=output_offset,
            batch_size=batch_size,
            device=device,
        )
    }
    for method, incoming in incoming_by_method.items():
        predictions[method] = evaluate_incoming(
            channel,
            decoder,
            input_embedding,
            output_head,
            tokenized,
            incoming,
            assistant_prefix_tokens=assistant_prefix_tokens,
            output_position_offset=output_offset,
            batch_size=batch_size,
            device=device,
        )
    correctness = {method: prediction == targets for method, prediction in predictions.items()}
    accuracy = {
        method: float(values.float().mean().item()) for method, values in correctness.items()
    }
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
    paired_targets = targets.index_select(0, pair_order)
    wrong_fact_target = predictions["wrong_fact"] == paired_targets
    matched_correct = correctness["matched"]
    conditional_wrong_fact = wrong_fact_target[matched_correct]
    with torch.inference_mode():
        projected = channel.project(matched_messages[:batch_size].to(device)).float()
    metrics = {
        "examples": len(hidden_split.examples),
        "pairs": len(hidden_split.examples) // 2,
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
        "derangement_has_zero_target_matches": not bool(
            (targets.index_select(0, deranged_order) == targets).any()
        ),
        "prefix_rms": float(projected.square().mean().sqrt().item()),
        "message_std_across_examples": float(
            matched_messages.float().std(dim=0, unbiased=False).mean().item()
        ),
    }
    return metrics, predictions


def assert_pair_alignment(tokenized: TokenizedSplit) -> None:
    if len(tokenized.examples) % 2:
        raise ValueError("HM-Synth split must contain aligned counterfactual pairs")
    if not torch.equal(tokenized.input_ids[0::2, 0], tokenized.input_ids[1::2, 0]):
        raise AssertionError("counterfactual pairs changed the designated receiver input")
    if bool((tokenized.target_ids[0::2] == tokenized.target_ids[1::2]).any()):
        raise AssertionError("counterfactual pairs contain an unchanged target")


def prediction_records(
    split_name: str,
    tokenized: TokenizedSplit,
    predictions: dict[str, torch.Tensor],
    tokenizer: Any,
    *,
    model_id: str,
    training_seed: int,
    checkpoint_hash: str,
    message_bytes: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, method_predictions in predictions.items():
        for index, (example, prediction_id) in enumerate(
            zip(tokenized.examples, method_predictions.tolist(), strict=True)
        ):
            records.append(
                {
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "variant": example.variant,
                    "split": split_name,
                    "method": method,
                    "model": model_id,
                    "training_seed": training_seed,
                    "designated_receiver": 0,
                    "prediction": decode_token(tokenizer, int(prediction_id)),
                    "target": example.answer,
                    "parsed_correctness": int(prediction_id) == int(tokenized.target_ids[index]),
                    "message_intervention": method,
                    "bytes": 0 if method == "no_message" else message_bytes,
                    "checkpoint_hash": checkpoint_hash,
                    "receiver_canvas_replaced": False,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    gpu_name = require_authorized_cuda(args.expected_gpu)
    seed_everything(args.seed)
    device = torch.device("cuda")
    run_dir = create_run_directory(args.run_root, args.run_id)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    shutil.copyfile(args.source_manifest, run_dir / "source-manifest.sha256")

    model_config = config["model"]
    cache_key = "models--" + str(model_config["id"]).replace("/", "--")
    snapshot = args.model_cache / cache_key / "snapshots" / str(model_config["revision"])
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned Dream snapshot is absent: {snapshot}")
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot),
        dtype=str(model_config["dtype"]),
        attn_implementation="eager",
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    decoder = model.get_decoder()
    input_embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    if not isinstance(decoder, nn.Module):
        raise TypeError("Dream decoder is not a torch module")
    if not isinstance(input_embedding, nn.Embedding):
        raise TypeError("Dream input embedding is not nn.Embedding")
    if not isinstance(output_head, nn.Module):
        raise TypeError("Dream output head is not a torch module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = config["dataset"]
    answer_values = tuple(str(value) for value in dataset["answer_values"])
    assistant_prefix = str(dataset["assistant_prefix"])
    assistant_prefix_tokens = len(tokenizer.encode(assistant_prefix, add_special_tokens=False))
    if assistant_prefix_tokens <= 0:
        raise RuntimeError("assistant prefix must tokenize to at least one token")
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
        assistant_prefix=assistant_prefix,
    )
    assert_pair_alignment(train_tokenized)
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
            assistant_prefix=assistant_prefix,
        )
        teacher_hidden = precompute_receiver_answer_hidden(
            union_tokenized,
            model,
            output_head,
            batch_size=int(config["training"]["encode_batch_size"]),
            device=device,
            output_position_offset=int(model_config["output_position_offset"]),
        )
        del union_tokenized
    parity = assert_no_message_parity(
        model,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        output_position_offset=int(model_config["output_position_offset"]),
        device=device,
    )
    input_rms = embedding_rms(input_embedding)
    training = config["training"]
    channel = InputPrefixChannel(
        int(model.config.hidden_size),
        input_embedding_rms=input_rms,
        message_slots=int(training["message_slots"]),
        message_dim=int(training["message_dim"]),
        width=int(training["adapter_width"]),
        num_heads=int(training["attention_heads"]),
        max_agents=2,
    ).to(device)
    training_records = train_channel(
        channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        train_hidden,
        config,
        teacher_hidden=teacher_hidden,
        assistant_prefix_tokens=assistant_prefix_tokens,
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
    validation_tokenized = tokenize_split(
        validation_examples,
        tokenizer,
        max_sequence_length=int(dataset["max_sequence_length"]),
        assistant_prefix=assistant_prefix,
    )
    assert_pair_alignment(validation_tokenized)
    validation_hidden = precompute_hidden(
        validation_tokenized,
        model,
        output_head,
        batch_size=int(training["encode_batch_size"]),
        device=device,
    )
    validation_metrics, validation_predictions = evaluate_split(
        channel,
        decoder,
        input_embedding,
        output_head,
        validation_tokenized,
        validation_hidden,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        seed=args.seed + 2,
        device=device,
    )
    validation_passed = gate_passes(validation_metrics, config["gate"])
    engineering_smoke = config["stage"] == "engineering_smoke"
    open_test = validation_passed and not engineering_smoke
    development_test_metrics: dict[str, Any] | None = None
    development_test_predictions: dict[str, torch.Tensor] | None = None
    test_digest: str | None = None
    test_tokenized: TokenizedSplit | None = None
    test_hidden: HiddenSplit | None = None
    if open_test:
        test_examples = build_split(
            pair_count=int(dataset["test_pairs"]),
            seed_start=int(dataset["test_seed_start"]),
            hops=[int(value) for value in dataset["hops"]],
            distractors_per_agent=int(dataset["distractors_per_agent"]),
            answer_values=answer_values,
        )
        test_digest = split_digest(test_examples)
        test_tokenized = tokenize_split(
            test_examples,
            tokenizer,
            max_sequence_length=int(dataset["max_sequence_length"]),
            assistant_prefix=assistant_prefix,
        )
        assert_pair_alignment(test_tokenized)
        test_hidden = precompute_hidden(
            test_tokenized,
            model,
            output_head,
            batch_size=int(training["encode_batch_size"]),
            device=device,
        )
        development_test_metrics, development_test_predictions = evaluate_split(
            channel,
            decoder,
            input_embedding,
            output_head,
            test_tokenized,
            test_hidden,
            config,
            assistant_prefix_tokens=assistant_prefix_tokens,
            seed=args.seed + 3,
            device=device,
        )
    pilot_passed = bool(
        development_test_metrics is not None
        and gate_passes(development_test_metrics, config["gate"])
    )
    if engineering_smoke:
        verdict = "engineering_smoke_completed"
    elif not validation_passed:
        verdict = "stop_validation"
    elif pilot_passed:
        verdict = "development_pilot_pass"
    else:
        verdict = "stop_development_test"

    model_hash, weight_records = model_weight_manifest(snapshot)
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
            "mechanism": "sender_hidden_to_receiver_input_embedding_prefix",
            "prefix_insertion": "before_visible_assistant_prefix",
            "assistant_prefix_tokens": assistant_prefix_tokens,
            "input_embedding_rms": input_rms,
            "no_message_path": "unmodified_frozen_dream_without_prefix",
            "no_message_parity": parity,
            "sender_encoder_does_not_consult_receiver": True,
            "receiver_canvas_replaced": False,
            "backbone_frozen": True,
            "training_answers_fully_masked_single_token": True,
            "validation_is_single_configuration_endpoint": True,
            "development_test_opened_only_after_validation_pass": open_test,
            "gate": config["gate"],
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
                "sha256": train_digest,
            },
            "validation": {
                "pairs": int(dataset["validation_pairs"]),
                "seed_start": int(dataset["validation_seed_start"]),
                "sha256": validation_digest,
            },
            "development_test": {
                "pairs": int(dataset["test_pairs"]),
                "seed_start": int(dataset["test_seed_start"]),
                "materialized": open_test,
                "sha256": test_digest,
            },
            "seed_ranges_disjoint": True,
            "disjoint_from_stage_a_probe": True,
            "counterfactual_pairs_aligned": True,
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
            "input_embedding_rms": input_rms,
        },
    )
    write_immutable_json(
        run_dir / "seed_manifest.json",
        {
            "training_seed": args.seed,
            "train_generation_seed_start": dataset["train_seed_start"],
            "validation_generation_seed_start": dataset["validation_seed_start"],
            "development_test_generation_seed_start": dataset["test_seed_start"],
            "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
        },
    )
    write_immutable_json(
        run_dir / "metrics.json",
        {
            "verdict": verdict,
            "gate": config["gate"],
            "validation_gate_passed": validation_passed,
            "development_pilot_gate_passed": pilot_passed,
            "validation": validation_metrics,
            "development_test": development_test_metrics,
            "canvas_assignment_count": 0,
            "training_steps": training["steps"],
            "non_claim": config["non_claim"],
        },
    )
    write_jsonl(run_dir / "training_log.jsonl", training_records)
    endpoint_name = "development_test" if open_test else "validation"
    endpoint_tokenized = test_tokenized if open_test else validation_tokenized
    endpoint_predictions = development_test_predictions if open_test else validation_predictions
    if endpoint_tokenized is None or endpoint_predictions is None:
        raise AssertionError("selected endpoint artifacts are missing")
    message_bytes = int(training["message_slots"]) * int(training["message_dim"]) * 2
    write_jsonl(
        run_dir / "predictions.jsonl",
        prediction_records(
            endpoint_name,
            endpoint_tokenized,
            endpoint_predictions,
            tokenizer,
            model_id=str(model_config["id"]),
            training_seed=args.seed,
            checkpoint_hash=checkpoint_hash,
            message_bytes=message_bytes,
        ),
    )
    trajectory_records: list[dict[str, Any]] = []
    for index, example in enumerate(endpoint_tokenized.examples):
        for method in ("matched", "wrong_fact"):
            final_token = endpoint_predictions[method][index : index + 1]
            initial_canvas = endpoint_tokenized.input_ids[index, 0]
            trajectory_records.append(
                {
                    "example_id": example.example_id,
                    "method": method,
                    "agent_id": 0,
                    "step": 0,
                    "normalized_time": config["evaluation"]["normalized_time"],
                    "initial_canvas_sha256": sha256_tensor(initial_canvas),
                    "receiver_canvas_sha256_after_message": sha256_tensor(initial_canvas),
                    "final_token_sha256": sha256_tensor(final_token),
                    "canvas_replaced_from_sender": False,
                }
            )
    write_zstd_jsonl(run_dir / "trajectories.jsonl.zst", trajectory_records)
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
            "hidden_cache_dtype": "torch.bfloat16",
            "validation_hidden_shape": list(validation_hidden.hidden_states.shape),
            "development_test_hidden_shape": (
                list(test_hidden.hidden_states.shape) if test_hidden is not None else None
            ),
        },
    )
    (run_dir / "git_commit.txt").write_text(args.source_revision + "\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text(
        json.dumps(
            {
                "validation_gate_passed": validation_passed,
                "development_pilot_gate_passed": pilot_passed,
                "verdict": verdict,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print((run_dir / "stdout.log").read_text(encoding="utf-8"), end="")
    if not engineering_smoke and not pilot_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
