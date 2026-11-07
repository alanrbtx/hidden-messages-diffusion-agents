"""Audit literal slot/code prefixes before training another two-round channel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import dllm
import torch
import yaml  # type: ignore[import-untyped]
from phase5_hm_synth_tiny import (
    TokenizedSplit,
    different_target_derangement,
    model_weight_manifest,
    paired_bootstrap_interval,
    split_digest,
    tokenize_split,
)
from phase6_dream_input_prefix import assert_no_message_parity, receiver_logits
from phase6_dream_lora_input_prefix import load_adapter_checkpoint
from torch import nn

from hidden_messages.adaptation import attach_lora_to_decoder_layers, lora_named_parameters
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
        raise TypeError("oracle-prefix audit config must be a mapping")
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


def prefix_table(
    tokenizer: Any,
    *,
    labels: tuple[str, ...],
    template: str,
) -> dict[str, list[int]]:
    table: dict[str, list[int]] = {}
    lengths: set[int] = set()
    for label in labels:
        token_ids = [
            int(value)
            for value in tokenizer.encode(template.format(label=label), add_special_tokens=False)
        ]
        if not token_ids:
            raise RuntimeError(f"template {template!r} produced an empty prefix")
        table[label] = token_ids
        lengths.add(len(token_ids))
    if len(lengths) != 1:
        raise RuntimeError(f"template {template!r} has label-dependent token length: {lengths}")
    return table


@torch.inference_mode()
def predict_no_prefix(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    *,
    agent_id: int,
    output_position_offset: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
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
                    + output_position_offset
                ),
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def predict_literal_prefix(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    labels: list[str],
    table: dict[str, list[int]],
    *,
    agent_id: int,
    assistant_prefix_tokens: int,
    output_position_offset: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if len(labels) != len(tokenized.examples):
        raise ValueError("prefix labels do not align with examples")
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokenized.examples), batch_size):
        stop = min(start + batch_size, len(tokenized.examples))
        prefix_ids = torch.tensor(
            [table[label] for label in labels[start:stop]],
            dtype=torch.long,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = receiver_logits(
                decoder,
                input_embedding,
                output_head,
                input_ids=tokenized.input_ids[start:stop, agent_id].to(device),
                attention_mask=tokenized.attention_mask[start:stop, agent_id].to(device),
                output_positions=(
                    tokenized.answer_positions[start:stop, agent_id].to(device)
                    + output_position_offset
                ),
                insertion_positions=(
                    tokenized.answer_positions[start:stop, agent_id].to(device)
                    - assistant_prefix_tokens
                ),
                prefix_embeddings=input_embedding(prefix_ids),
            )
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


def select_candidate(records: list[dict[str, Any]]) -> int:
    if not records:
        raise ValueError("candidate records cannot be empty")
    selected = 0
    best = float(records[0]["matched_accuracy"])
    for index, record in enumerate(records[1:], start=1):
        score = float(record["matched_accuracy"])
        if score > best:
            selected = index
            best = score
    return selected


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


def candidate_search(
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    tokenizer: Any,
    templates: list[str],
    labels_by_example: list[str],
    label_values: tuple[str, ...],
    *,
    agent_id: int,
    assistant_prefix_tokens: int,
    output_position_offset: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[dict[str, list[int]]]]:
    records: list[dict[str, Any]] = []
    predictions: list[torch.Tensor] = []
    tables: list[dict[str, list[int]]] = []
    for candidate_index, template in enumerate(templates):
        table = prefix_table(tokenizer, labels=label_values, template=template)
        predicted = predict_literal_prefix(
            decoder,
            input_embedding,
            output_head,
            tokenized,
            labels_by_example,
            table,
            agent_id=agent_id,
            assistant_prefix_tokens=assistant_prefix_tokens,
            output_position_offset=output_position_offset,
            batch_size=batch_size,
            device=device,
        )
        records.append(
            {
                "candidate_index": candidate_index,
                "template": template,
                "token_count": len(next(iter(table.values()))),
                "token_ids": table,
                "matched_accuracy": float(
                    (predicted == tokenized.target_ids).float().mean().item()
                ),
            }
        )
        predictions.append(predicted)
        tables.append(table)
    return records, predictions, tables


def gate_passes(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, bool]:
    round1 = metrics["round1_slot_to_sender"]
    round2 = metrics["round2_code_to_receiver"]
    round1_ci_low = round1["matched_minus_wrong_slot_pair_bootstrap_ci95"][0]
    round2_ci_low = round2["matched_minus_wrong_fact_pair_bootstrap_ci95"][0]
    checks = {
        "round1_matched_at_least_min": round1["matched_accuracy"]
        >= float(gate["round1_matched_min"]),
        "round1_no_slot_at_most_max": round1["no_prefix_accuracy"]
        <= float(gate["round1_no_slot_max"]),
        "round1_effect_at_least_min": round1["matched_minus_wrong_slot"]
        >= float(gate["round1_matched_minus_wrong_slot_min"]),
        "round1_effect_ci_low_above_zero": round1_ci_low > 0.0,
        "round1_wrong_slot_target_at_least_min": round1["wrong_slot_target_rate"]
        >= float(gate["round1_wrong_slot_target_rate_min"]),
        "round2_matched_at_least_min": round2["matched_accuracy"]
        >= float(gate["round2_matched_min"]),
        "round2_no_code_at_most_max": round2["no_prefix_accuracy"]
        <= float(gate["round2_no_code_max"]),
        "round2_effect_at_least_min": round2["matched_minus_wrong_fact"]
        >= float(gate["round2_matched_minus_wrong_fact_min"]),
        "round2_effect_ci_low_above_zero": round2_ci_low > 0.0,
        "round2_wrong_fact_target_at_least_min": round2["wrong_fact_target_rate"]
        >= float(gate["round2_wrong_fact_target_rate_min"]),
    }
    checks["all"] = all(checks.values())
    return checks


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
    seed_everything(int(config["evaluation"]["seed"]))
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
        raise RuntimeError("sealed LoRA checkpoint differs from oracle-prefix config")
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
    discovery_examples = build_split(
        pair_count=int(dataset["discovery_pairs"]),
        seed_start=int(dataset["discovery_seed_start"]),
        **split_kwargs,
    )
    audit_examples = build_split(
        pair_count=int(dataset["audit_pairs"]),
        seed_start=int(dataset["audit_seed_start"]),
        **split_kwargs,
    )
    tokenize_kwargs = {
        "tokenizer": tokenizer,
        "max_sequence_length": int(dataset["max_sequence_length"]),
        "assistant_prefix": str(dataset["assistant_prefix"]),
    }
    discovery = tokenize_split(discovery_examples, **tokenize_kwargs)
    audit = tokenize_split(audit_examples, **tokenize_kwargs)
    assistant_prefix_tokens = len(
        tokenizer.encode(str(dataset["assistant_prefix"]), add_special_tokens=False)
    )
    parity = assert_no_message_parity(
        model,
        decoder,
        input_embedding,
        output_head,
        discovery,
        output_position_offset=int(model_config["output_position_offset"]),
        device=device,
    )
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    prefix_search = config["prefix_search"]
    discovery_slot_labels = [
        str(example.metadata["requested_slot"]) for example in discovery.examples
    ]
    discovery_code_labels = [example.answer for example in discovery.examples]
    slot_records, slot_predictions, slot_tables = candidate_search(
        decoder,
        input_embedding,
        output_head,
        discovery,
        tokenizer,
        [str(value) for value in prefix_search["round1_templates"]],
        discovery_slot_labels,
        slot_values,
        agent_id=1,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    code_records, code_predictions, code_tables = candidate_search(
        decoder,
        input_embedding,
        output_head,
        discovery,
        tokenizer,
        [str(value) for value in prefix_search["round2_templates"]],
        discovery_code_labels,
        answer_values,
        agent_id=0,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    selected_slot_index = select_candidate(slot_records)
    selected_code_index = select_candidate(code_records)

    audit_slot_labels = [str(example.metadata["requested_slot"]) for example in audit.examples]
    audit_code_labels = [example.answer for example in audit.examples]
    slot_ids = torch.tensor(
        [slot_values.index(label) for label in audit_slot_labels], dtype=torch.long
    )
    slot_order = different_target_derangement(slot_ids)
    wrong_slot_labels = [audit_slot_labels[index] for index in slot_order.tolist()]
    pair_order = torch.arange(len(audit.examples)).bitwise_xor(1)
    wrong_code_labels = [audit_code_labels[index] for index in pair_order.tolist()]
    round1_matched = predict_literal_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        audit_slot_labels,
        slot_tables[selected_slot_index],
        agent_id=1,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    round1_wrong = predict_literal_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        wrong_slot_labels,
        slot_tables[selected_slot_index],
        agent_id=1,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    sender_no_prefix = predict_no_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        agent_id=1,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    round2_matched = predict_literal_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        audit_code_labels,
        code_tables[selected_code_index],
        agent_id=0,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    round2_wrong = predict_literal_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        wrong_code_labels,
        code_tables[selected_code_index],
        agent_id=0,
        assistant_prefix_tokens=assistant_prefix_tokens,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    receiver_no_prefix = predict_no_prefix(
        decoder,
        input_embedding,
        output_head,
        audit,
        agent_id=0,
        output_position_offset=int(model_config["output_position_offset"]),
        batch_size=batch_size,
        device=device,
    )
    targets = audit.target_ids
    round1_matched_correct = round1_matched == targets
    round1_wrong_correct = round1_wrong == targets
    round2_matched_correct = round2_matched == targets
    round2_wrong_correct = round2_wrong == targets
    wrong_slot_expected = wrong_slot_targets(audit.examples, slot_order, tokenizer)
    paired_targets = targets.index_select(0, pair_order)
    round1_ci = paired_bootstrap_interval(
        round1_matched_correct,
        round1_wrong_correct,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    round2_ci = paired_bootstrap_interval(
        round2_matched_correct,
        round2_wrong_correct,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]) + 1,
    )
    round1_matched_accuracy = float(round1_matched_correct.float().mean().item())
    round1_wrong_accuracy = float(round1_wrong_correct.float().mean().item())
    round2_matched_accuracy = float(round2_matched_correct.float().mean().item())
    round2_wrong_accuracy = float(round2_wrong_correct.float().mean().item())
    round2_wrong_fact_target_rate = float((round2_wrong == paired_targets).float().mean().item())
    audit_metrics = {
        "examples": len(audit.examples),
        "pairs": len(audit.examples) // 2,
        "round1_slot_to_sender": {
            "matched_accuracy": round1_matched_accuracy,
            "no_prefix_accuracy": float((sender_no_prefix == targets).float().mean().item()),
            "wrong_slot_original_target_accuracy": round1_wrong_accuracy,
            "wrong_slot_target_rate": float(
                (round1_wrong == wrong_slot_expected).float().mean().item()
            ),
            "matched_minus_wrong_slot": round1_matched_accuracy - round1_wrong_accuracy,
            "matched_minus_wrong_slot_pair_bootstrap_ci95": list(round1_ci),
        },
        "round2_code_to_receiver": {
            "matched_accuracy": round2_matched_accuracy,
            "no_prefix_accuracy": float((receiver_no_prefix == targets).float().mean().item()),
            "wrong_fact_original_target_accuracy": round2_wrong_accuracy,
            "wrong_fact_target_rate": round2_wrong_fact_target_rate,
            "matched_minus_wrong_fact": round2_matched_accuracy - round2_wrong_accuracy,
            "matched_minus_wrong_fact_pair_bootstrap_ci95": list(round2_ci),
        },
    }
    checks = gate_passes(audit_metrics, evaluation["gate"])
    engineering_smoke = config["stage"] == "engineering_smoke"
    if engineering_smoke:
        verdict = "engineering_smoke_completed"
    elif checks["all"]:
        verdict = "go_semantic_two_round_successor"
    else:
        verdict = "stop_oracle_prefix_surface"

    discovery_prediction_records: list[dict[str, Any]] = []
    for direction, candidate_predictions in (
        ("round1_slot_to_sender", slot_predictions),
        ("round2_code_to_receiver", code_predictions),
    ):
        for candidate_index, values in enumerate(candidate_predictions):
            for example_index, example in enumerate(discovery.examples):
                discovery_prediction_records.append(
                    {
                        "split": "discovery",
                        "direction": direction,
                        "candidate_index": candidate_index,
                        "example_id": example.example_id,
                        "pair_id": example.pair_id,
                        "variant": example.variant,
                        "prediction_token_id": int(values[example_index]),
                        "target_token_id": int(discovery.target_ids[example_index]),
                        "correct": bool(
                            values[example_index] == discovery.target_ids[example_index]
                        ),
                    }
                )
    audit_predictions = {
        "round1_matched": round1_matched,
        "round1_wrong_slot": round1_wrong,
        "sender_no_prefix": sender_no_prefix,
        "round2_matched": round2_matched,
        "round2_wrong_fact": round2_wrong,
        "receiver_no_prefix": receiver_no_prefix,
    }
    audit_prediction_records: list[dict[str, Any]] = []
    for method, values in audit_predictions.items():
        for example_index, example in enumerate(audit.examples):
            audit_prediction_records.append(
                {
                    "split": "audit",
                    "method": method,
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "variant": example.variant,
                    "prediction_token_id": int(values[example_index]),
                    "target_token_id": int(targets[example_index]),
                    "correct": bool(values[example_index] == targets[example_index]),
                }
            )

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
            "mechanism": "oracle_literal_input_embedding_prefix_surface_audit",
            "selection_rule": prefix_search["selection_rule"],
            "selected_round1_candidate": slot_records[selected_slot_index],
            "selected_round2_candidate": code_records[selected_code_index],
            "target_ids_used_only_for_oracle_surface": True,
            "communication_trained": False,
            "pretrained_backbone_frozen": True,
            "language_model_head_frozen": True,
            "task_lora_frozen": True,
            "no_message_parity": parity,
            "gate": evaluation["gate"],
            "non_claim": config["non_claim"],
        },
    )
    write_immutable_json(
        run_dir / "dataset_manifest.json",
        {
            "family": dataset["family"],
            "discovery": {
                "pairs": int(dataset["discovery_pairs"]),
                "seed_start": int(dataset["discovery_seed_start"]),
                "sha256": split_digest(discovery_examples),
            },
            "audit": {
                "pairs": int(dataset["audit_pairs"]),
                "seed_start": int(dataset["audit_seed_start"]),
                "sha256": split_digest(audit_examples),
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
            "effective_attention_classes": sorted(
                {type(getattr(layer, "self_attn", None)).__name__ for layer in layers}
            ),
            "pretrained_parameters_frozen": True,
            "language_model_head_frozen": True,
            "lora_frozen": True,
        },
    )
    write_immutable_json(
        run_dir / "metrics.json",
        {
            "verdict": verdict,
            "gate": evaluation["gate"],
            "gate_checks": checks,
            "discovery": {
                "round1_candidates": slot_records,
                "round2_candidates": code_records,
                "selected_round1_candidate_index": selected_slot_index,
                "selected_round2_candidate_index": selected_code_index,
            },
            "audit": audit_metrics,
            "communication_trained": False,
            "non_claim": config["non_claim"],
        },
    )
    write_jsonl(run_dir / "discovery_predictions.jsonl", discovery_prediction_records)
    write_jsonl(run_dir / "audit_predictions.jsonl", audit_prediction_records)
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
        },
    )
    write_text_exclusive(run_dir / "git_commit.txt", args.source_revision + "\n")
    stdout = json.dumps({"gate_passed": checks["all"], "verdict": verdict}, sort_keys=True)
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
