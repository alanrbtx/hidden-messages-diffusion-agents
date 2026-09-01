"""Train and audit a dense latent two-round Dream prefix protocol."""

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
from phase6_dream_balanced_oracle_prefix_audit import (
    build_split,
    predict_no_prefix,
    prefix_table,
    wrong_slot_targets,
)
from phase6_dream_balanced_two_round import augmented_forward
from phase6_dream_input_prefix import assert_no_message_parity, embedding_rms
from phase6_dream_lora_input_prefix import load_adapter_checkpoint
from phase6_dream_semantic_two_round import answer_states, class_targets, cosine_factor
from torch import nn

from hidden_messages.adaptation import attach_lora_to_decoder_layers, lora_named_parameters
from hidden_messages.communication import DenseLatentPrefixChannel, batch_message_variance_loss
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
        raise TypeError("dense latent two-round config must be a mapping")
    return payload


def target_prefix_embeddings(
    table: dict[str, list[int]],
    values: tuple[str, ...],
    input_embedding: nn.Embedding,
    *,
    device: torch.device,
) -> torch.Tensor:
    token_ids = torch.tensor([table[value] for value in values], dtype=torch.long, device=device)
    with torch.no_grad():
        return input_embedding(token_ids).detach()


def representation_parameters(channel: DenseLatentPrefixChannel) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in (channel.state_encoder, channel.message_norm, channel.message_classifier):
        parameters.extend(module.parameters())
    return parameters


def prefix_parameters(channel: DenseLatentPrefixChannel) -> list[nn.Parameter]:
    return [*channel.prefix_projection.parameters(), channel.log_scale]


def train_representation(
    channel: DenseLatentPrefixChannel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    if batch_size > hidden.shape[0]:
        raise ValueError("training batch exceeds the available examples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(hidden.shape[0], generator=generator)
    cursor = 0
    optimizer = torch.optim.AdamW(
        representation_parameters(channel),
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
        if cursor + batch_size > hidden.shape[0]:
            permutation = torch.randperm(hidden.shape[0], generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        selected_hidden = hidden.index_select(0, indices).to(device)
        selected_targets = targets.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages = channel.encode(selected_hidden)
            logits = channel.classify_messages(messages)
            classification_loss = nn.functional.cross_entropy(logits.float(), selected_targets)
            variance_loss, message_std = batch_message_variance_loss(
                [messages], target_std=float(training["target_message_std"])
            )
            loss = classification_loss + float(training["lambda_message_variance"]) * variance_loss
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            representation_parameters(channel), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "classification_loss": float(classification_loss.item()),
                    "message_variance_loss": float(variance_loss.item()),
                    "message_feature_std_mean": float(message_std.item()),
                    "batch_accuracy": float(
                        (logits.argmax(dim=-1) == selected_targets).float().mean().item()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    channel.freeze_representation()
    return records


def train_prefix_decoder(
    channel: DenseLatentPrefixChannel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(hidden.shape[0], generator=generator)
    cursor = 0
    optimizer = torch.optim.AdamW(
        prefix_parameters(channel),
        lr=float(training["prefix_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = max(1, round(steps * float(training["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: cosine_factor(index, steps=steps, warmup_steps=warmup_steps),
    )
    records: list[dict[str, float | int]] = []
    for step in range(steps):
        if cursor + batch_size > hidden.shape[0]:
            permutation = torch.randperm(hidden.shape[0], generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        selected_hidden = hidden.index_select(0, indices).to(device)
        selected_targets = targets.index_select(0, indices).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages = channel.encode(selected_hidden)
            target_prefixes = prototypes.index_select(0, selected_targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = channel.project(messages)
            cosine_loss = (
                1.0
                - nn.functional.cosine_similarity(
                    prefixes.float(), target_prefixes.float(), dim=-1
                ).mean()
            )
            relative_mse = (
                prefixes.float() - target_prefixes.float()
            ).square().mean() / target_prefixes.float().square().mean().clamp_min(1e-12)
            loss = cosine_loss + float(training["lambda_prefix_relative_mse"]) * relative_mse
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            prefix_parameters(channel), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "prefix_cosine": float(1.0 - cosine_loss.item()),
                    "prefix_relative_mse": float(relative_mse.item()),
                    "prefix_scale": float(channel.log_scale.exp().item()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    channel.eval()
    return records


@torch.inference_mode()
def encode_all(
    channel: DenseLatentPrefixChannel,
    hidden: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    channel.eval()
    for start in range(0, hidden.shape[0], batch_size):
        stop = min(start + batch_size, hidden.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            messages = channel.encode(hidden[start:stop].to(device))
        outputs.append(messages.to(device="cpu", dtype=torch.bfloat16))
    return torch.cat(outputs)


@torch.inference_mode()
def sender_after_slot(
    slot_channel: DenseLatentPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    slot_messages: torch.Tensor,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(config["evaluation"]["batch_size"])
    hidden_outputs: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    output_offset = int(config["model"]["output_position_offset"])
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = slot_channel.project(slot_messages[start:stop].to(device))
            logits, hidden = augmented_forward(
                decoder,
                input_embedding,
                output_head,
                input_ids=tokenized.input_ids[start:stop, 1].to(device),
                attention_mask=tokenized.attention_mask[start:stop, 1].to(device),
                answer_positions=tokenized.answer_positions[start:stop, 1].to(device),
                assistant_prefix_tokens=assistant_prefix_tokens,
                output_position_offset=output_offset,
                prefix_embeddings=prefixes,
            )
        positions = tokenized.answer_positions[start:stop, 1].to(device) + output_offset
        rows = torch.arange(stop - start, device=device)
        hidden_outputs.append(hidden[rows, positions].to(device="cpu", dtype=torch.bfloat16))
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(hidden_outputs), torch.cat(predictions)


@torch.inference_mode()
def receiver_predictions(
    code_channel: DenseLatentPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    code_messages: torch.Tensor,
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
            prefixes = code_channel.project(code_messages[start:stop].to(device))
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


def dense_message_diagnostics(
    messages: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, float | int | str | bool]:
    values = messages.float().flatten(start_dim=1)
    present_labels = sorted({int(value) for value in targets.tolist()})
    if not present_labels:
        raise RuntimeError("dense-message diagnostics require at least one example")
    feature_std = values.std(dim=0, unbiased=False)
    centroids = torch.stack([values[targets == label].mean(dim=0) for label in present_labels])
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    normalized_centroids = nn.functional.normalize(centroids, dim=-1)
    cosine_distance = 1.0 - normalized_centroids @ normalized_centroids.transpose(0, 1)
    off_diagonal = ~torch.eye(len(present_labels), dtype=torch.bool)
    within_class_std = torch.stack(
        [values[targets == label].std(dim=0, unbiased=False).mean() for label in present_labels]
    )
    minimum_centroid_distance = (
        float(cosine_distance[off_diagonal].min().item()) if len(present_labels) > 1 else 0.0
    )
    return {
        "transport": "signed_dense_bfloat16_slots",
        "probability_simplex_payload": False,
        "token_id_payload": False,
        "shape_slots": messages.shape[1],
        "shape_message_dim": messages.shape[2],
        "declared_class_count": num_classes,
        "present_class_count": len(present_labels),
        "declared_class_coverage": len(present_labels) / num_classes,
        "mean_l2_norm": float(values.norm(dim=-1).mean().item()),
        "feature_std_mean": float(feature_std.mean().item()),
        "within_class_feature_std_mean": float(within_class_std.mean().item()),
        "centroid_effective_rank": float(effective_rank.item()),
        "centroid_min_cosine_distance": minimum_centroid_distance,
        "exact_zero_fraction": float((values == 0).float().mean().item()),
    }


@torch.inference_mode()
def projected_prefix_cosine(
    channel: DenseLatentPrefixChannel,
    messages: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    similarities: list[torch.Tensor] = []
    for start in range(0, messages.shape[0], batch_size):
        stop = min(start + batch_size, messages.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefixes = channel.project(messages[start:stop].to(device))
        target = prototypes.index_select(0, targets[start:stop].to(device))
        similarities.append(
            nn.functional.cosine_similarity(prefixes.float(), target.float(), dim=-1).cpu()
        )
    return float(torch.cat(similarities).mean().item())


@torch.inference_mode()
def classify_messages(
    channel: DenseLatentPrefixChannel,
    messages: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, messages.shape[0], batch_size):
        stop = min(start + batch_size, messages.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = channel.classify_messages(messages[start:stop].to(device))
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def evaluate_split(
    slot_channel: DenseLatentPrefixChannel,
    code_channel: DenseLatentPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    slot_values: tuple[str, ...],
    answer_values: tuple[str, ...],
    slot_prototypes: torch.Tensor,
    code_prototypes: torch.Tensor,
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    output_offset = int(config["model"]["output_position_offset"])
    receiver_hidden = answer_states(hidden_split, agent_id=0, output_position_offset=output_offset)
    raw_sender_hidden = answer_states(
        hidden_split, agent_id=1, output_position_offset=output_offset
    )
    slot_targets = class_targets(tokenized.examples, slot_values, metadata_key="requested_slot")
    code_targets = class_targets(tokenized.examples, answer_values, metadata_key=None)
    slot_messages = encode_all(
        slot_channel,
        receiver_hidden,
        batch_size=batch_size,
        device=device,
    )
    matched_sender_hidden, sender_matched = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        tokenized,
        slot_messages,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    slot_order = different_target_derangement(slot_targets)
    wrong_slot_messages = slot_messages.index_select(0, slot_order)
    wrong_sender_hidden, sender_wrong_slot = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        tokenized,
        wrong_slot_messages,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    sender_no_slot = predict_no_prefix(
        decoder,
        input_embedding,
        output_head,
        tokenized,
        agent_id=1,
        output_position_offset=output_offset,
        batch_size=batch_size,
        device=device,
    )
    matched_code_messages = encode_all(
        code_channel,
        matched_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    final_only_messages = encode_all(
        code_channel,
        raw_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    wrong_slot_code_messages = encode_all(
        code_channel,
        wrong_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    pair_order = torch.arange(len(tokenized.examples)).bitwise_xor(1)
    deranged_order = different_target_derangement(tokenized.target_ids)
    incoming = {
        "matched": matched_code_messages,
        "final_only": final_only_messages,
        "wrong_fact": matched_code_messages.index_select(0, pair_order),
        "deranged": matched_code_messages.index_select(0, deranged_order),
        "wrong_slot": wrong_slot_code_messages,
        "zero": torch.zeros_like(matched_code_messages),
    }
    predictions = {
        "no_message": predict_no_prefix(
            decoder,
            input_embedding,
            output_head,
            tokenized,
            agent_id=0,
            output_position_offset=output_offset,
            batch_size=batch_size,
            device=device,
        )
    }
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
    wrong_slot_expected = wrong_slot_targets(tokenized.examples, slot_order, tokenizer)
    paired_targets = targets.index_select(0, pair_order)
    slot_diagnostics = dense_message_diagnostics(
        slot_messages,
        slot_targets,
        num_classes=len(slot_values),
    )
    code_diagnostics = dense_message_diagnostics(
        matched_code_messages,
        code_targets,
        num_classes=len(answer_values),
    )
    metrics = {
        "examples": len(tokenized.examples),
        "pairs": len(tokenized.examples) // 2,
        "accuracy": accuracy,
        "slot_classifier_accuracy": float(
            (
                classify_messages(slot_channel, slot_messages, batch_size=batch_size, device=device)
                == slot_targets
            )
            .float()
            .mean()
            .item()
        ),
        "code_classifier_accuracy": float(
            (
                classify_messages(
                    code_channel, matched_code_messages, batch_size=batch_size, device=device
                )
                == code_targets
            )
            .float()
            .mean()
            .item()
        ),
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
        "matched_pair_consistency": float(
            correctness["matched"].reshape(-1, 2).all(dim=1).float().mean().item()
        ),
        "derangement_has_zero_target_matches": not bool(
            (targets.index_select(0, deranged_order) == targets).any()
        ),
        "slot_message": slot_diagnostics,
        "matched_code_message": code_diagnostics,
        "final_only_code_message": dense_message_diagnostics(
            final_only_messages,
            code_targets,
            num_classes=len(answer_values),
        ),
        "slot_prefix_cosine": projected_prefix_cosine(
            slot_channel,
            slot_messages,
            slot_targets,
            slot_prototypes,
            batch_size=batch_size,
            device=device,
        ),
        "code_prefix_cosine": projected_prefix_cosine(
            code_channel,
            matched_code_messages,
            code_targets,
            code_prototypes,
            batch_size=batch_size,
            device=device,
        ),
    }
    predictions["sender_matched_slot"] = sender_matched
    predictions["sender_no_slot"] = sender_no_slot
    predictions["sender_wrong_slot"] = sender_wrong_slot
    return metrics, predictions


def gate_passes(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    accuracy = metrics["accuracy"]
    sender = metrics["sender_bridge_accuracy"]
    slot_message = metrics["slot_message"]
    code_message = metrics["matched_code_message"]
    return bool(
        metrics["slot_classifier_accuracy"] >= float(gate["slot_classifier_min"])
        and metrics["code_classifier_accuracy"] >= float(gate["code_classifier_min"])
        and metrics["slot_prefix_cosine"] >= float(gate["prefix_cosine_min"])
        and metrics["code_prefix_cosine"] >= float(gate["prefix_cosine_min"])
        and slot_message["feature_std_mean"] >= float(gate["message_feature_std_min"])
        and code_message["feature_std_mean"] >= float(gate["message_feature_std_min"])
        and slot_message["declared_class_coverage"] >= float(gate["class_coverage_min"])
        and code_message["declared_class_coverage"] >= float(gate["class_coverage_min"])
        and slot_message["centroid_effective_rank"] >= float(gate["centroid_effective_rank_min"])
        and code_message["centroid_effective_rank"] >= float(gate["centroid_effective_rank_min"])
        and slot_message["centroid_min_cosine_distance"]
        >= float(gate["centroid_min_cosine_distance_min"])
        and code_message["centroid_min_cosine_distance"]
        >= float(gate["centroid_min_cosine_distance_min"])
        and sender["matched_slot"] >= float(gate["sender_matched_min"])
        and sender["no_slot"] <= float(gate["sender_no_slot_max"])
        and accuracy["matched"] >= float(gate["matched_accuracy_min"])
        and accuracy["final_only"] <= float(gate["final_only_accuracy_max"])
        and metrics["matched_minus_final_only"] >= float(gate["matched_minus_final_only_min"])
        and metrics["matched_minus_final_only_pair_bootstrap_ci95"][0] > 0.0
        and metrics["matched_minus_wrong_fact"] >= float(gate["matched_minus_wrong_fact_min"])
        and metrics["matched_minus_wrong_fact_pair_bootstrap_ci95"][0] > 0.0
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
        raise RuntimeError("sealed LoRA checkpoint differs from dense-channel config")
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
    split_kwargs = {"slot_values": slot_values, "answer_values": answer_values}
    train_examples = build_split(
        pair_count=int(dataset["train_pairs"]),
        seed_start=int(dataset["train_seed_start"]),
        **split_kwargs,
    )
    validation_examples = build_split(
        pair_count=int(dataset["validation_pairs"]),
        seed_start=int(dataset["validation_seed_start"]),
        **split_kwargs,
    )
    tokenize_kwargs = {
        "tokenizer": tokenizer,
        "max_sequence_length": int(dataset["max_sequence_length"]),
        "assistant_prefix": str(dataset["assistant_prefix"]),
    }
    train_tokenized = tokenize_split(train_examples, **tokenize_kwargs)
    validation_tokenized = tokenize_split(validation_examples, **tokenize_kwargs)
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
    slot_table = prefix_table(
        tokenizer,
        labels=slot_values,
        template=str(channel_config["round1_alignment_template"]),
    )
    code_table = prefix_table(
        tokenizer,
        labels=answer_values,
        template=str(channel_config["round2_alignment_template"]),
    )
    slot_prototypes = target_prefix_embeddings(
        slot_table, slot_values, input_embedding, device=device
    )
    code_prototypes = target_prefix_embeddings(
        code_table, answer_values, input_embedding, device=device
    )
    if slot_prototypes.shape[1] != int(channel_config["message_slots"]):
        raise RuntimeError("round-1 teacher prefix length differs from message slots")
    if code_prototypes.shape[1] != int(channel_config["message_slots"]):
        raise RuntimeError("round-2 teacher prefix length differs from message slots")
    input_rms = embedding_rms(input_embedding)

    def make_channel(num_classes: int) -> DenseLatentPrefixChannel:
        return DenseLatentPrefixChannel(
            int(model.config.hidden_size),
            num_classes,
            input_embedding_rms=input_rms,
            message_slots=int(channel_config["message_slots"]),
            message_dim=int(channel_config["message_dim"]),
            width=int(channel_config["width"]),
            classifier_width=int(channel_config["classifier_width"]),
        ).to(device)

    slot_channel = make_channel(len(slot_values))
    code_channel = make_channel(len(answer_values))
    output_offset = int(model_config["output_position_offset"])
    train_slot_targets = class_targets(train_examples, slot_values, metadata_key="requested_slot")
    train_code_targets = class_targets(train_examples, answer_values, metadata_key=None)
    round1_representation_log = train_representation(
        slot_channel,
        answer_states(train_hidden, agent_id=0, output_position_offset=output_offset),
        train_slot_targets,
        config,
        seed=int(config["training"]["round1_seed"]),
        steps=int(config["training"]["round1_representation_steps"]),
        device=device,
    )
    round1_prefix_log = train_prefix_decoder(
        slot_channel,
        answer_states(train_hidden, agent_id=0, output_position_offset=output_offset),
        train_slot_targets,
        slot_prototypes,
        config,
        seed=int(config["training"]["round1_prefix_seed"]),
        steps=int(config["training"]["round1_prefix_steps"]),
        device=device,
    )
    train_slot_messages = encode_all(
        slot_channel,
        answer_states(train_hidden, agent_id=0, output_position_offset=output_offset),
        batch_size=int(config["evaluation"]["batch_size"]),
        device=device,
    )
    matched_train_sender_hidden, _ = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        train_slot_messages,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    round2_representation_log = train_representation(
        code_channel,
        matched_train_sender_hidden,
        train_code_targets,
        config,
        seed=int(config["training"]["round2_seed"]),
        steps=int(config["training"]["round2_representation_steps"]),
        device=device,
    )
    round2_prefix_log = train_prefix_decoder(
        code_channel,
        matched_train_sender_hidden,
        train_code_targets,
        code_prototypes,
        config,
        seed=int(config["training"]["round2_prefix_seed"]),
        steps=int(config["training"]["round2_prefix_steps"]),
        device=device,
    )
    slot_path = run_dir / "round1_dense_latent_channel.pt"
    code_path = run_dir / "round2_dense_latent_channel.pt"
    torch.save(slot_channel.state_dict(), slot_path)
    torch.save(code_channel.state_dict(), code_path)
    del matched_train_sender_hidden, train_slot_messages, train_hidden
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
        slot_values,
        answer_values,
        slot_prototypes,
        code_prototypes,
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
            **split_kwargs,
        )
        test_digest = split_digest(test_examples)
        test_tokenized = tokenize_split(test_examples, **tokenize_kwargs)
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
            slot_values,
            answer_values,
            slot_prototypes,
            code_prototypes,
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
        verdict = "stop_dense_latent_validation"
    elif pilot_passed:
        verdict = "go_dense_latent_persistent_denoising"
    else:
        verdict = "stop_dense_latent_development_test"

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
            "mechanism": "two_directional_supervised_dense_latent_prefix_channel",
            "transmitted_shape": [
                int(channel_config["message_slots"]),
                int(channel_config["message_dim"]),
            ],
            "transmitted_dtype": channel_config["transport_dtype"],
            "probability_simplex_payload": False,
            "token_id_payload": False,
            "literal_prefix_embeddings_used_during_inference": False,
            "literal_prefix_embeddings_used_as_training_targets": True,
            "hard_argmax_used_to_construct_message": False,
            "auxiliary_semantic_labels_used_during_training": True,
            "pretrained_backbone_frozen": True,
            "language_model_head_frozen": True,
            "task_lora_frozen": True,
            "receiver_canvas_replaced": False,
            "no_message_parity": parity,
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
            "lora_attachments": [attachment.qualified_name for attachment in attachments],
            "round1_channel_sha256": sha256_file(slot_path),
            "round2_channel_sha256": sha256_file(code_path),
            "effective_attention_classes": sorted(
                {type(getattr(layer, "self_attn", None)).__name__ for layer in layers}
            ),
            "pretrained_parameters_frozen": True,
            "language_model_head_frozen": True,
            "lora_frozen": True,
        },
    )
    write_immutable_json(
        run_dir / "seed_manifest.json",
        {
            "round1_training_seed": config["training"]["round1_seed"],
            "round1_prefix_seed": config["training"]["round1_prefix_seed"],
            "round2_training_seed": config["training"]["round2_seed"],
            "round2_prefix_seed": config["training"]["round2_prefix_seed"],
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
    write_jsonl(run_dir / "round1_representation_log.jsonl", round1_representation_log)
    write_jsonl(run_dir / "round1_prefix_log.jsonl", round1_prefix_log)
    write_jsonl(run_dir / "round2_representation_log.jsonl", round2_representation_log)
    write_jsonl(run_dir / "round2_prefix_log.jsonl", round2_prefix_log)
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
