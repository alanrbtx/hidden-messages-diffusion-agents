"""Train and audit a supervised continuous semantic two-round Dream channel."""

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
from phase6_dream_input_prefix import assert_no_message_parity
from phase6_dream_lora_input_prefix import load_adapter_checkpoint
from torch import nn

from hidden_messages.adaptation import attach_lora_to_decoder_layers, lora_named_parameters
from hidden_messages.communication import SemanticPrototypeChannel
from hidden_messages.datasets.hm_synth import HMSynthExample
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
        raise TypeError("semantic two-round config must be a mapping")
    return payload


def cosine_factor(step_index: int, *, steps: int, warmup_steps: int) -> float:
    if step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    progress = (step_index - warmup_steps) / max(steps - warmup_steps, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def answer_states(
    hidden_split: HiddenSplit,
    *,
    agent_id: int,
    output_position_offset: int,
) -> torch.Tensor:
    positions = hidden_split.answer_positions[:, agent_id] + output_position_offset
    if bool((positions < 0).any()):
        raise RuntimeError("output-position offset moved before the sequence")
    rows = torch.arange(len(hidden_split.examples))
    return hidden_split.hidden_states[rows, agent_id, positions]


def class_targets(
    examples: list[HMSynthExample],
    values: tuple[str, ...],
    *,
    metadata_key: str | None,
) -> torch.Tensor:
    labels = [
        example.answer if metadata_key is None else str(example.metadata[metadata_key])
        for example in examples
    ]
    return torch.tensor([values.index(label) for label in labels], dtype=torch.long)


def prototype_embeddings(
    table: dict[str, list[int]],
    values: tuple[str, ...],
    input_embedding: nn.Embedding,
    *,
    device: torch.device,
) -> torch.Tensor:
    token_ids = torch.tensor([table[value] for value in values], dtype=torch.long, device=device)
    with torch.no_grad():
        return input_embedding(token_ids).detach()


def train_channel(
    channel: SemanticPrototypeChannel,
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
        if cursor + batch_size > hidden.shape[0]:
            permutation = torch.randperm(hidden.shape[0], generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        selected_hidden = hidden.index_select(0, indices).to(device)
        selected_targets = targets.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = channel.classify(selected_hidden)
            loss = nn.functional.cross_entropy(logits.float(), selected_targets)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            channel.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            probabilities = torch.softmax(
                logits.float() / channel.temperature,
                dim=-1,
            )
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            records.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "batch_accuracy": float(
                        (logits.argmax(dim=-1) == selected_targets).float().mean()
                    ),
                    "mean_message_entropy": float(entropy.mean().item()),
                    "mean_max_probability": float(probabilities.max(dim=-1).values.mean().item()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
    channel.eval()
    return records


@torch.inference_mode()
def encode_all(
    channel: SemanticPrototypeChannel,
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
            probabilities = channel.encode(hidden[start:stop].to(device))
        outputs.append(probabilities.to(device="cpu", dtype=torch.bfloat16))
    return torch.cat(outputs)


@torch.inference_mode()
def sender_after_slot(
    slot_channel: SemanticPrototypeChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    slot_probabilities: torch.Tensor,
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
            prefixes = slot_channel.project(slot_probabilities[start:stop].to(device))
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
    code_channel: SemanticPrototypeChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    code_probabilities: torch.Tensor,
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
            prefixes = code_channel.project(code_probabilities[start:stop].to(device))
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


def probability_diagnostics(probabilities: torch.Tensor) -> dict[str, float]:
    values = probabilities.float()
    entropy = -(values * values.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "mean_entropy": float(entropy.mean().item()),
        "mean_max_probability": float(values.max(dim=-1).values.mean().item()),
    }


@torch.inference_mode()
def evaluate_split(
    slot_channel: SemanticPrototypeChannel,
    code_channel: SemanticPrototypeChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    slot_values: tuple[str, ...],
    answer_values: tuple[str, ...],
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
        hidden_split,
        agent_id=1,
        output_position_offset=output_offset,
    )
    slot_targets = class_targets(
        tokenized.examples,
        slot_values,
        metadata_key="requested_slot",
    )
    code_targets = class_targets(tokenized.examples, answer_values, metadata_key=None)
    slot_probabilities = encode_all(
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
        slot_probabilities,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    slot_order = different_target_derangement(slot_targets)
    wrong_slot_probabilities = slot_probabilities.index_select(0, slot_order)
    wrong_sender_hidden, sender_wrong_slot = sender_after_slot(
        slot_channel,
        decoder,
        input_embedding,
        output_head,
        tokenized,
        wrong_slot_probabilities,
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
    matched_code_probabilities = encode_all(
        code_channel,
        matched_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    final_only_probabilities = encode_all(
        code_channel,
        raw_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    wrong_slot_code_probabilities = encode_all(
        code_channel,
        wrong_sender_hidden,
        batch_size=batch_size,
        device=device,
    )
    pair_order = torch.arange(len(tokenized.examples)).bitwise_xor(1)
    deranged_order = different_target_derangement(tokenized.target_ids)
    incoming = {
        "matched": matched_code_probabilities,
        "final_only": final_only_probabilities,
        "wrong_fact": matched_code_probabilities.index_select(0, pair_order),
        "deranged": matched_code_probabilities.index_select(0, deranged_order),
        "wrong_slot": wrong_slot_code_probabilities,
        "zero": torch.zeros_like(matched_code_probabilities),
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
    for method, probabilities in incoming.items():
        predictions[method] = receiver_predictions(
            code_channel,
            decoder,
            input_embedding,
            output_head,
            tokenized,
            probabilities,
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
    metrics = {
        "examples": len(tokenized.examples),
        "pairs": len(tokenized.examples) // 2,
        "accuracy": accuracy,
        "slot_classifier_accuracy": float(
            (slot_probabilities.argmax(dim=-1) == slot_targets).float().mean().item()
        ),
        "code_classifier_accuracy": float(
            (matched_code_probabilities.argmax(dim=-1) == code_targets).float().mean().item()
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
        "slot_message": probability_diagnostics(slot_probabilities),
        "matched_code_message": probability_diagnostics(matched_code_probabilities),
        "final_only_code_message": probability_diagnostics(final_only_probabilities),
    }
    predictions["sender_matched_slot"] = sender_matched
    predictions["sender_no_slot"] = sender_no_slot
    predictions["sender_wrong_slot"] = sender_wrong_slot
    return metrics, predictions


def gate_passes(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    accuracy = metrics["accuracy"]
    sender = metrics["sender_bridge_accuracy"]
    return bool(
        metrics["slot_classifier_accuracy"] >= float(gate["slot_classifier_min"])
        and sender["matched_slot"] >= float(gate["sender_matched_min"])
        and sender["no_slot"] <= float(gate["sender_no_slot_max"])
        and metrics["code_classifier_accuracy"] >= float(gate["code_classifier_min"])
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
        raise RuntimeError("sealed LoRA checkpoint differs from semantic-channel config")
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
        template=str(channel_config["round1_template"]),
    )
    code_table = prefix_table(
        tokenizer,
        labels=answer_values,
        template=str(channel_config["round2_template"]),
    )
    slot_channel = SemanticPrototypeChannel(
        int(model.config.hidden_size),
        prototype_embeddings(slot_table, slot_values, input_embedding, device=device),
        width=int(channel_config["classifier_width"]),
        temperature=float(channel_config["temperature"]),
    ).to(device)
    code_channel = SemanticPrototypeChannel(
        int(model.config.hidden_size),
        prototype_embeddings(code_table, answer_values, input_embedding, device=device),
        width=int(channel_config["classifier_width"]),
        temperature=float(channel_config["temperature"]),
    ).to(device)
    output_offset = int(model_config["output_position_offset"])
    train_slot_targets = class_targets(train_examples, slot_values, metadata_key="requested_slot")
    train_code_targets = class_targets(train_examples, answer_values, metadata_key=None)
    round1_log = train_channel(
        slot_channel,
        answer_states(train_hidden, agent_id=0, output_position_offset=output_offset),
        train_slot_targets,
        config,
        seed=int(config["training"]["round1_seed"]),
        steps=int(config["training"]["round1_steps"]),
        device=device,
    )
    train_slot_probabilities = encode_all(
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
        train_slot_probabilities,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    round2_log = train_channel(
        code_channel,
        matched_train_sender_hidden,
        train_code_targets,
        config,
        seed=int(config["training"]["round2_seed"]),
        steps=int(config["training"]["round2_steps"]),
        device=device,
    )
    slot_path = run_dir / "round1_semantic_channel.pt"
    code_path = run_dir / "round2_semantic_channel.pt"
    torch.save(slot_channel.state_dict(), slot_path)
    torch.save(code_channel.state_dict(), code_path)
    del matched_train_sender_hidden, train_slot_probabilities, train_hidden
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
        verdict = "go_semantic_multiseed_confirmation"
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
            "mechanism": "two_directional_supervised_continuous_semantic_prototype_channel",
            "round1": "receiver_state_to_soft_requested_slot_distribution",
            "round2": "conditioned_sender_state_to_soft_code_distribution",
            "hard_argmax_used_for_prefix": False,
            "target_ids_inference_payload": False,
            "prototype_token_ids": {"round1": slot_table, "round2": code_table},
            "continuous_transport_dtype": channel_config["transport_dtype"],
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
            "message_bytes_per_direction": len(slot_values) * 2,
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
