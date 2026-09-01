"""Independently verify the sealed v15 dependency-span confirmation."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from hidden_messages.communication import DenseLatentPrefixChannel
from hidden_messages.datasets.hm_synth import (
    HMSynthExample,
    generate_rendezvous_lookup_pair,
    pair_preserving_control_batches,
)

METHODS = {
    "no_message",
    "schedule_only",
    "matched",
    "matched_ephemeral",
    "wrong_fact",
    "deranged",
    "wrong_slot",
    "zero",
    "final_only",
    "sequential_final_two_round",
}
DYNAMIC_METHODS = {
    "matched",
    "matched_ephemeral",
    "wrong_fact",
    "deranged",
    "wrong_slot",
    "zero",
}
COMPARATORS = [
    "no_message",
    "schedule_only",
    "final_only",
    "wrong_fact",
    "deranged",
    "matched_ephemeral",
    "zero",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inventory(inventory: Path, root: Path) -> int:
    lines = inventory.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"empty checksum inventory: {inventory}")
    seen: set[str] = set()
    for line in lines:
        expected, separator, relative = line.partition("  ")
        if separator != "  " or not expected or relative in seen:
            raise RuntimeError(f"invalid inventory entry: {line!r}")
        seen.add(relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"checksum mismatch: {path}")
    return len(lines)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def build_split(
    pair_count: int,
    seed_start: int,
    slot_values: tuple[str, ...],
    answer_values: tuple[str, ...],
) -> list[HMSynthExample]:
    examples: list[HMSynthExample] = []
    for pair_index in range(pair_count):
        examples.extend(
            generate_rendezvous_lookup_pair(
                seed_start + pair_index,
                slot_values=slot_values,
                answer_values=answer_values,
            )
        )
    return examples


def split_digest(examples: list[HMSynthExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.canonical_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def different_target_order(labels: list[str]) -> list[int]:
    sorted_positions = sorted(range(len(labels)), key=lambda index: (labels[index], index))
    shift = max(Counter(labels).values())
    if 2 * shift > len(labels):
        raise RuntimeError("different-target control is infeasible")
    rolled = sorted_positions[shift:] + sorted_positions[:shift]
    order = [0] * len(labels)
    for destination, source in zip(sorted_positions, rolled, strict=True):
        order[destination] = source
    if any(labels[index] == labels[source] for index, source in enumerate(order)):
        raise AssertionError("different-target order retained a label")
    return order


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    pair_effects = (left.astype(np.float32) - right.astype(np.float32)).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, pair_effects.shape[0], size=(replicates, pair_effects.shape[0]))
    bootstrap = pair_effects[sampled].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return [float(low), float(high)]


def assert_close(observed: float, expected: float, label: str, *, tolerance: float = 1e-7) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"reported {label} differs from raw recomputation")


@torch.inference_mode()
def recompute_channel_accuracy(
    states: torch.Tensor,
    targets: torch.Tensor,
    checkpoint: Path,
    channel_config: dict[str, Any],
    *,
    num_classes: int,
    device: torch.device,
    batch_size: int,
) -> float:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    input_rms = float(state_dict["input_embedding_rms"].item())
    channel = DenseLatentPrefixChannel(
        int(states.shape[1]),
        num_classes,
        input_embedding_rms=input_rms,
        message_slots=int(channel_config["message_slots"]),
        message_dim=int(channel_config["message_dim"]),
        width=int(channel_config["width"]),
        classifier_width=int(channel_config["classifier_width"]),
    ).to(device)
    channel.load_state_dict(state_dict)
    channel.eval()
    predictions: list[torch.Tensor] = []
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = channel.classify(states[start:stop].to(device))
        predictions.append(logits.argmax(dim=-1).cpu())
    accuracy = float((torch.cat(predictions) == targets).float().mean().item())
    del channel, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


def recompute_endpoint(
    *,
    endpoint: str,
    examples: list[HMSynthExample],
    config: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    reported: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    generation = config["generation"]
    batch_examples = int(evaluation["batch_examples"])
    control_batches = pair_preserving_control_batches(examples, batch_examples=batch_examples)
    scheduled = [example for batch in control_batches for example in batch]
    rows = [row for row in prediction_rows if str(row["endpoint"]) == endpoint]
    if len(rows) != len(scheduled) * len(METHODS):
        raise AssertionError(f"{endpoint} raw prediction row count is incomplete")
    if any(int(row["designated_receiver"]) != 0 for row in rows):
        raise AssertionError(f"{endpoint} changed the designated receiver")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    token_by_answer: dict[str, int] = {}
    for row in rows:
        key = (str(row["example_id"]), str(row["method"]))
        if key in records:
            raise AssertionError(f"duplicate {endpoint} raw prediction: {key}")
        records[key] = row
        answer = str(row["target"])
        token_id = int(row["target_token_id"])
        if answer in token_by_answer and token_by_answer[answer] != token_id:
            raise AssertionError("one answer maps to multiple target token IDs")
        token_by_answer[answer] = token_id
        if bool(row["correct"]) != (int(row["prediction_token_id"]) == token_id):
            raise AssertionError(f"{endpoint} correct flag differs from token equality")
    if {method for _, method in records} != METHODS:
        raise AssertionError(f"{endpoint} methods differ from the frozen matrix")
    if {example_id for example_id, _ in records} != {example.example_id for example in scheduled}:
        raise AssertionError(f"{endpoint} example IDs differ from the frozen cohort")

    targets = np.asarray([token_by_answer[example.answer] for example in scheduled], dtype=np.int64)
    predictions = {
        method: np.asarray(
            [
                int(records[(example.example_id, method)]["prediction_token_id"])
                for example in scheduled
            ],
            dtype=np.int64,
        )
        for method in sorted(METHODS)
    }
    correctness = {method: values == targets for method, values in predictions.items()}
    accuracy = {
        method: float(values.astype(np.float32).mean()) for method, values in correctness.items()
    }
    contrasts: dict[str, dict[str, Any]] = {}
    for offset, comparator in enumerate(COMPARATORS):
        contrasts[f"matched_minus_{comparator}"] = {
            "effect": accuracy["matched"] - accuracy[comparator],
            "pair_bootstrap_ci95": paired_bootstrap_interval(
                correctness["matched"],
                correctness[comparator],
                replicates=int(evaluation["bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]) + offset,
            ),
        }

    pair_order = np.arange(len(scheduled), dtype=np.int64) ^ 1
    wrong_fact_target_rate = float(
        (predictions["wrong_fact"] == targets[pair_order]).astype(np.float32).mean()
    )
    wrong_slot_expected: list[int] = []
    for batch in control_batches:
        slot_labels = [str(example.metadata["requested_slot"]) for example in batch]
        order = different_target_order(slot_labels)
        for index, source_index in enumerate(order):
            wrong_slot = str(batch[source_index].metadata["requested_slot"])
            slots = [str(value) for value in cast(list[str], batch[index].metadata["slot_values"])]
            table_key = (
                "factual_codes" if batch[index].variant == "factual" else "counterfactual_codes"
            )
            codes = [str(value) for value in cast(list[str], batch[index].metadata[table_key])]
            wrong_slot_expected.append(token_by_answer[codes[slots.index(wrong_slot)]])
    wrong_slot_target_rate = float(
        (predictions["wrong_slot"] == np.asarray(wrong_slot_expected, dtype=np.int64))
        .astype(np.float32)
        .mean()
    )
    matched_pair_consistency = float(
        correctness["matched"].reshape(-1, 2).all(axis=1).astype(np.float32).mean()
    )

    max_new_tokens = int(generation["max_new_tokens"])
    receiver_commit = int(generation["answer_commit_not_before_steps"]["receiver"])
    sender_commit = int(generation["answer_commit_not_before_steps"]["sender"])
    receiver_reveals = np.asarray(
        [
            records[(example.example_id, "matched")]["receiver_generation_reveal_steps"]
            for example in scheduled
        ],
        dtype=np.int64,
    )
    sender_reveals = np.asarray(
        [
            records[(example.example_id, "matched")]["sender_generation_reveal_steps"]
            for example in scheduled
        ],
        dtype=np.int64,
    )
    expected_reveal_shape = (len(scheduled), max_new_tokens)
    if (
        receiver_reveals.shape != expected_reveal_shape
        or sender_reveals.shape != expected_reveal_shape
    ):
        raise AssertionError(f"{endpoint} generation reveal matrix has the wrong shape")
    receiver_generation_commit = float(
        (receiver_reveals.min(axis=1) >= receiver_commit).astype(np.float32).mean()
    )
    sender_generation_commit = float(
        (sender_reveals.min(axis=1) >= sender_commit).astype(np.float32).mean()
    )

    endpoint_trajectories = [row for row in trajectory_rows if str(row["endpoint"]) == endpoint]
    expected_trajectory_rows = len(control_batches) * len(DYNAMIC_METHODS) * 2
    if len(endpoint_trajectories) != expected_trajectory_rows:
        raise AssertionError(f"{endpoint} raw trajectory row count is incomplete")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_trajectories:
        groups[(int(row["batch_start"]), str(row["method"]))].append(row)
    if {method for _, method in groups} != DYNAMIC_METHODS:
        raise AssertionError(f"{endpoint} trajectory methods differ from the frozen matrix")
    if any(
        sorted(int(row["step_index"]) for row in group) != [2, 4]
        or sorted(int(row["round_index"]) for row in group) != [1, 2]
        for group in groups.values()
    ):
        raise AssertionError(f"{endpoint} communication occurred outside passes 2/4")
    for (_, method), group in groups.items():
        if method != "matched":
            continue
        by_round = {int(row["round_index"]): row for row in group}
        if by_round[1]["slot_payload_kind"] != "learned_dense":
            raise AssertionError(f"{endpoint} matched stage one did not carry a learned payload")
        if by_round[2]["code_payload_kind"] != "learned_dense":
            raise AssertionError(f"{endpoint} matched stage two did not carry a learned payload")
        if by_round[1]["cache_was_present"] or not by_round[2]["cache_was_present"]:
            raise AssertionError(f"{endpoint} matched cache lifecycle differs from the design")

    for method, value in accuracy.items():
        assert_close(float(reported["accuracy"][method]), value, f"{endpoint}.accuracy.{method}")
    for name, values in contrasts.items():
        assert_close(
            float(reported["contrasts"][name]["effect"]),
            float(values["effect"]),
            f"{endpoint}.{name}.effect",
        )
        for index, value in enumerate(values["pair_bootstrap_ci95"]):
            assert_close(
                float(reported["contrasts"][name]["pair_bootstrap_ci95"][index]),
                float(value),
                f"{endpoint}.{name}.ci{index}",
            )
    assert_close(
        float(reported["wrong_fact_target_rate"]),
        wrong_fact_target_rate,
        f"{endpoint}.wrong_fact_target_rate",
    )
    assert_close(
        float(reported["wrong_slot_target_rate"]),
        wrong_slot_target_rate,
        f"{endpoint}.wrong_slot_target_rate",
    )
    assert_close(
        float(reported["matched_pair_consistency"]),
        matched_pair_consistency,
        f"{endpoint}.matched_pair_consistency",
    )
    assert_close(
        float(reported["receiver_generation_commit_schedule_respected_fraction"]),
        receiver_generation_commit,
        f"{endpoint}.receiver_generation_commit",
    )
    assert_close(
        float(reported["sender_generation_commit_schedule_respected_fraction"]),
        sender_generation_commit,
        f"{endpoint}.sender_generation_commit",
    )
    expected_bytes = (
        2 * int(config["channel"]["message_slots"]) * int(config["channel"]["message_dim"]) * 2
    )
    if int(reported["message_bytes_per_example"]["matched"]) != expected_bytes:
        raise AssertionError(f"{endpoint} message-byte accounting differs from the protocol")

    gate = evaluation["gate"]
    gate_checks = {
        "communication_off_upstream_parity": bool(reported["communication_off_upstream_parity"]),
        "all_refresh_rounds_executed": bool(reported["all_refresh_rounds_executed"]),
        "cache_applied_after_each_refresh": bool(reported["cache_applied_after_each_refresh"]),
        "zero_canvas_assignments": int(reported["canvas_assignment_count"]) == 0,
        "receiver_commit_schedule": float(reported["receiver_commit_schedule_respected_fraction"])
        == 1.0,
        "sender_commit_schedule": float(reported["sender_commit_schedule_respected_fraction"])
        == 1.0,
        "receiver_generation_commit_schedule": receiver_generation_commit == 1.0,
        "sender_generation_commit_schedule": sender_generation_commit == 1.0,
        "matched_accuracy": accuracy["matched"] >= float(gate["matched_accuracy_min"]),
        "no_message_accuracy": accuracy["no_message"] <= float(gate["no_message_accuracy_max"]),
        "schedule_only_accuracy": accuracy["schedule_only"]
        <= float(gate["schedule_only_accuracy_max"]),
        "final_only_accuracy": accuracy["final_only"] <= float(gate["final_only_accuracy_max"]),
        "wrong_fact_target_rate": wrong_fact_target_rate
        >= float(gate["wrong_fact_target_rate_min"]),
        "wrong_slot_target_rate": wrong_slot_target_rate
        >= float(gate["wrong_slot_target_rate_min"]),
    }
    contrast_gate_map = {
        "matched_minus_no_message": "matched_minus_no_message_min",
        "matched_minus_schedule_only": "matched_minus_schedule_only_min",
        "matched_minus_final_only": "matched_minus_final_only_min",
        "matched_minus_wrong_fact": "matched_minus_wrong_fact_min",
        "matched_minus_deranged": "matched_minus_deranged_min",
        "matched_minus_zero": "matched_minus_zero_min",
    }
    for contrast_name, gate_name in contrast_gate_map.items():
        contrast = contrasts[contrast_name]
        gate_checks[contrast_name] = bool(
            contrast["effect"] >= float(gate[gate_name])
            and contrast["pair_bootstrap_ci95"][0] > 0.0
        )
    return {
        "accuracy": accuracy,
        "contrasts": contrasts,
        "wrong_fact_target_rate": wrong_fact_target_rate,
        "wrong_slot_target_rate": wrong_slot_target_rate,
        "matched_pair_consistency": matched_pair_consistency,
        "receiver_generation_commit_schedule_respected_fraction": receiver_generation_commit,
        "sender_generation_commit_schedule_respected_fraction": sender_generation_commit,
        "gate_passed": all(gate_checks.values()),
        "failed_gates": sorted(name for name, passed in gate_checks.items() if not passed),
        "raw_prediction_rows": len(rows),
        "raw_trajectory_rows": len(endpoint_trajectories),
    }


def normalized_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    normalized = copy.deepcopy(payload)
    for key in ("round1_seed", "round1_prefix_seed", "round2_seed", "round2_prefix_seed"):
        normalized["training"].pop(key)
    normalized["evaluation"].pop("bootstrap_seed")
    normalized["dataset"].pop("train_seed_start")
    normalized["dataset"].pop("validation_seed_start")
    return normalized


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def audit_member(
    run_dir: Path,
    source_root: Path,
    expected_source_manifest_sha256: str,
    *,
    verify_source: bool,
) -> tuple[dict[str, Any], int | None]:
    run = load_json(run_dir / "run.json")
    metrics = load_json(run_dir / "metrics.json")
    dataset_manifest = load_json(run_dir / "dataset_manifest.json")
    model_manifest = load_json(run_dir / "model_manifest.json")
    seeds = load_json(run_dir / "seed_manifest.json")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("member config must be a mapping")
    if run["stage"] != "confirmation" or run["protocol_id"] != (
        "phase6-dream7b-dependency-span-confirmation-v15"
    ):
        raise AssertionError("member identity differs from the frozen confirmation")
    if config["stage"] != "confirmation" or config["evaluation"].get("oracle_stage") is not None:
        raise AssertionError("member config is not the learned-only confirmation design")
    if config["generation"].get("communication_step_indices") != [2, 4]:
        raise AssertionError("member communication steps differ from passes 2/4")
    if config["generation"].get("answer_commit_scope") != "full_generation":
        raise AssertionError("member did not use the full-generation dependency barrier")

    package_files = verify_inventory(run_dir / "SHA256SUMS", run_dir)
    source_manifest = run_dir / "source-manifest.sha256"
    if sha256_file(source_manifest) != expected_source_manifest_sha256:
        raise AssertionError("member source manifest differs from the aggregate source manifest")
    source_files = verify_inventory(source_manifest, source_root) if verify_source else None

    dataset = config["dataset"]
    slot_values = tuple(str(value) for value in dataset["slot_values"])
    answer_values = tuple(str(value) for value in dataset["answer_values"])
    train = build_split(
        int(dataset["train_pairs"]), int(dataset["train_seed_start"]), slot_values, answer_values
    )
    validation = build_split(
        int(dataset["validation_pairs"]),
        int(dataset["validation_seed_start"]),
        slot_values,
        answer_values,
    )
    sealed_test_materialized = bool(dataset_manifest["development_test"]["materialized"])
    sealed_test = (
        build_split(
            int(dataset["development_test_pairs"]),
            int(dataset["development_test_seed_start"]),
            slot_values,
            answer_values,
        )
        if sealed_test_materialized
        else None
    )
    expected_splits = {
        "train": split_digest(train),
        "validation": split_digest(validation),
        "development_test": split_digest(sealed_test) if sealed_test is not None else None,
    }
    for name, digest in expected_splits.items():
        if dataset_manifest[name]["sha256"] != digest:
            raise AssertionError(f"{run_dir.name} {name} digest differs from reconstruction")
    if sealed_test_materialized != (metrics["development_test"] is not None):
        raise AssertionError("sealed-test materialization differs from member metrics")

    prediction_rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trajectory_rows = [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_endpoints = {"validation"}
    if sealed_test_materialized:
        expected_endpoints.add("development_test")
    if {str(row["endpoint"]) for row in prediction_rows} != expected_endpoints:
        raise AssertionError("confirmation predictions contain the wrong endpoint set")
    validation_recomputed = recompute_endpoint(
        endpoint="validation",
        examples=validation,
        config=config,
        prediction_rows=prediction_rows,
        trajectory_rows=trajectory_rows,
        reported=metrics["validation"],
    )
    test_recomputed = (
        recompute_endpoint(
            endpoint="development_test",
            examples=sealed_test,
            config=config,
            prediction_rows=prediction_rows,
            trajectory_rows=trajectory_rows,
            reported=metrics["development_test"],
        )
        if sealed_test is not None
        else None
    )

    if (
        sha256_file(run_dir / "round1_time_conditioned_channel.pt")
        != model_manifest["round1_channel_sha256"]
    ):
        raise AssertionError("round-1 channel hash differs from the model manifest")
    if (
        sha256_file(run_dir / "round2_time_conditioned_channel.pt")
        != model_manifest["round2_channel_sha256"]
    ):
        raise AssertionError("round-2 channel hash differs from the model manifest")
    state_path = run_dir / "intermediate_training_states.pt"
    validation_state_path = run_dir / "intermediate_validation_states.pt"
    if sha256_file(state_path) != model_manifest["intermediate_training_states_sha256"]:
        raise AssertionError("training-state archive hash differs from the model manifest")
    if (
        sha256_file(validation_state_path)
        != model_manifest["intermediate_validation_states_sha256"]
    ):
        raise AssertionError("validation-state archive hash differs from the model manifest")

    device = torch.device("cuda")
    batch_size = int(config["training"]["diagnostic_batch_size"])
    training_archive = torch.load(state_path, map_location="cpu", weights_only=False)
    expected_capture = config["training"]["expected_capture_steps"]
    capture_steps_exact = bool(
        set(int(value) for value in training_archive["receiver_capture"]["step_indices"])
        == {int(expected_capture["receiver"])}
        and set(int(value) for value in training_archive["sender_capture"]["slot_step_indices"])
        == {int(expected_capture["receiver"])}
        and set(int(value) for value in training_archive["sender_capture"]["sender_step_indices"])
        == {int(expected_capture["sender"])}
    )
    if not capture_steps_exact or not metrics["training"]["capture_steps_exact"]:
        raise AssertionError("training states were not captured exactly at passes 2/4")
    train_slot_accuracy = recompute_channel_accuracy(
        training_archive["receiver_pass2"],
        training_archive["slot_targets"],
        run_dir / "round1_time_conditioned_channel.pt",
        config["channel"],
        num_classes=len(slot_values),
        device=device,
        batch_size=batch_size,
    )
    train_code_accuracy = recompute_channel_accuracy(
        training_archive["sender_pass4"],
        training_archive["code_targets"],
        run_dir / "round2_time_conditioned_channel.pt",
        config["channel"],
        num_classes=len(answer_values),
        device=device,
        batch_size=batch_size,
    )
    assert_close(
        float(metrics["training"]["slot_classifier_accuracy"]),
        train_slot_accuracy,
        f"{run_dir.name}.training.slot_classifier_accuracy",
    )
    assert_close(
        float(metrics["training"]["code_classifier_accuracy"]),
        train_code_accuracy,
        f"{run_dir.name}.training.code_classifier_accuracy",
    )

    validation_archive = torch.load(validation_state_path, map_location="cpu", weights_only=False)
    validation_slot_accuracy = recompute_channel_accuracy(
        validation_archive["receiver_pass2"],
        validation_archive["slot_targets"],
        run_dir / "round1_time_conditioned_channel.pt",
        config["channel"],
        num_classes=len(slot_values),
        device=device,
        batch_size=batch_size,
    )
    validation_code_accuracy = recompute_channel_accuracy(
        validation_archive["sender_pass4"],
        validation_archive["code_targets"],
        run_dir / "round2_time_conditioned_channel.pt",
        config["channel"],
        num_classes=len(answer_values),
        device=device,
        batch_size=batch_size,
    )
    assert_close(
        float(metrics["validation_channel"]["slot_classifier_accuracy"]),
        validation_slot_accuracy,
        f"{run_dir.name}.validation_channel.slot_classifier_accuracy",
    )
    assert_close(
        float(metrics["validation_channel"]["code_classifier_accuracy"]),
        validation_code_accuracy,
        f"{run_dir.name}.validation_channel.code_classifier_accuracy",
    )

    training_gate = config["training"]["gate"]
    training = metrics["training"]
    training_gate_passed = bool(
        train_slot_accuracy >= float(training_gate["slot_classifier_min"])
        and train_code_accuracy >= float(training_gate["code_classifier_min"])
        and float(training["slot_prefix_cosine"]) >= float(training_gate["prefix_cosine_min"])
        and float(training["code_prefix_cosine"]) >= float(training_gate["prefix_cosine_min"])
        and float(training["slot_message"]["feature_std_mean"])
        >= float(training_gate["message_feature_std_min"])
        and float(training["code_message"]["feature_std_mean"])
        >= float(training_gate["message_feature_std_min"])
        and float(training["slot_message"]["centroid_effective_rank"])
        >= float(training_gate["centroid_effective_rank_min"])
        and float(training["code_message"]["centroid_effective_rank"])
        >= float(training_gate["centroid_effective_rank_min"])
        and float(training["slot_message"]["centroid_min_cosine_distance"])
        >= float(training_gate["centroid_min_cosine_distance_min"])
        and float(training["code_message"]["centroid_min_cosine_distance"])
        >= float(training_gate["centroid_min_cosine_distance_min"])
        and capture_steps_exact
    )
    if training_gate_passed != bool(metrics["training_gate_passed"]):
        raise AssertionError("reported training gate differs from independent recomputation")
    if bool(metrics["validation_gate_passed"]) != bool(
        training_gate_passed and validation_recomputed["gate_passed"]
    ):
        raise AssertionError("reported validation gate differs from raw recomputation")
    test_gate_passed = bool(test_recomputed is not None and test_recomputed["gate_passed"])
    if bool(metrics["development_pilot_gate_passed"]) != test_gate_passed:
        raise AssertionError("reported sealed-test gate differs from raw recomputation")
    member_passed = bool(
        training_gate_passed and validation_recomputed["gate_passed"] and test_gate_passed
    )
    if not training_gate_passed:
        expected_verdict = "stop_time_conditioned_channel_training"
    elif not validation_recomputed["gate_passed"]:
        expected_verdict = "stop_commit_aware_validation"
    elif test_gate_passed:
        expected_verdict = "confirmation_seed_passed"
    else:
        expected_verdict = "stop_commit_aware_development_test"
    if metrics["verdict"] != expected_verdict:
        raise AssertionError("confirmation member has an unexpected verdict")
    return (
        {
            "run_id": run_dir.name,
            "package_files_verified": package_files,
            "run_inventory_sha256": sha256_file(run_dir / "SHA256SUMS"),
            "source_manifest_sha256": sha256_file(source_manifest),
            "train_sha256": expected_splits["train"],
            "validation_sha256": expected_splits["validation"],
            "sealed_test_sha256": expected_splits["development_test"],
            "seeds": seeds,
            "model_manifest": model_manifest,
            "training_classifier_accuracy": {
                "slot": train_slot_accuracy,
                "code": train_code_accuracy,
            },
            "validation_classifier_accuracy": {
                "slot": validation_slot_accuracy,
                "code": validation_code_accuracy,
            },
            "training_gate_passed": training_gate_passed,
            "validation": validation_recomputed,
            "sealed_test": test_recomputed,
            "member_passed": member_passed,
            "reported_verdict": metrics["verdict"],
        },
        source_files,
    )


def main() -> None:
    args = parse_args()
    if args.audit_dir.exists():
        raise FileExistsError(f"audit destination already exists: {args.audit_dir}")
    aggregate_files = verify_inventory(args.aggregate_dir / "SHA256SUMS", args.aggregate_dir)
    aggregate_run = load_json(args.aggregate_dir / "run.json")
    aggregate_metrics = load_json(args.aggregate_dir / "metrics.json")
    member_manifest = load_json(args.aggregate_dir / "member_manifest.json")
    if aggregate_run["stage"] != "confirmation_aggregate" or aggregate_run["protocol_id"] != (
        "phase6-dream7b-dependency-span-confirmation-v15"
    ):
        raise AssertionError("aggregate identity differs from the frozen v15 protocol")
    aggregate_source_manifest = args.aggregate_dir / "source-manifest.sha256"
    source_manifest_sha256 = sha256_file(aggregate_source_manifest)
    run_root = args.aggregate_dir.parent
    member_ids = [str(value) for value in member_manifest["member_run_ids"]]
    if len(member_ids) != 3 or len(set(member_ids)) != 3:
        raise AssertionError("aggregate does not declare three distinct members")

    member_audits: list[dict[str, Any]] = []
    source_files_verified: int | None = None
    configs: list[dict[str, Any]] = []
    for index, run_id in enumerate(member_ids):
        run_dir = run_root / run_id
        member_audit, source_files = audit_member(
            run_dir,
            args.source_root,
            source_manifest_sha256,
            verify_source=index == 0,
        )
        if source_files is not None:
            source_files_verified = source_files
        member_audits.append(member_audit)
        configs.append(normalized_config(run_dir / "config.yaml"))
    if source_files_verified is None:
        raise AssertionError("source snapshot was not independently verified")
    reference = json.dumps(configs[0], sort_keys=True)
    if any(json.dumps(config, sort_keys=True) != reference for config in configs):
        raise AssertionError("member configs differ beyond frozen seed identities")

    train_digests = {member["train_sha256"] for member in member_audits}
    validation_digests = {member["validation_sha256"] for member in member_audits}
    materialized_test_digests = {
        member["sealed_test_sha256"]
        for member in member_audits
        if member["sealed_test_sha256"] is not None
    }
    round1_hashes = {member["model_manifest"]["round1_channel_sha256"] for member in member_audits}
    round2_hashes = {member["model_manifest"]["round2_channel_sha256"] for member in member_audits}
    if len(train_digests) != 3 or len(validation_digests) != 3:
        raise AssertionError("member train or validation cohorts are not distinct")
    if len(materialized_test_digests) > 1:
        raise AssertionError("materialized members do not share one sealed test")
    if len(round1_hashes) != 3 or len(round2_hashes) != 3:
        raise AssertionError("member channel checkpoints are not distinct")

    all_seed_passed = all(bool(member["member_passed"]) for member in member_audits)
    verdict = (
        "PASS_PHASE6_DEPENDENCY_SPAN_SYNTHETIC_MULTISEED"
        if all_seed_passed
        else "STOP_PHASE6_DEPENDENCY_SPAN_SYNTHETIC_MULTISEED"
    )
    if bool(aggregate_metrics["all_seed_gates_passed"]) != all_seed_passed:
        raise AssertionError("aggregate seed verdict differs from member recomputation")
    if aggregate_metrics["verdict"] != verdict:
        raise AssertionError("aggregate verdict differs from member recomputation")
    common_test_digest = (
        next(iter(materialized_test_digests)) if materialized_test_digests else None
    )
    if aggregate_metrics["common_test_sha256"] != common_test_digest:
        raise AssertionError("aggregate common test digest differs from reconstruction")

    metric_extractors = {
        "matched_accuracy": lambda test: test["accuracy"]["matched"],
        "no_message_accuracy": lambda test: test["accuracy"]["no_message"],
        "schedule_only_accuracy": lambda test: test["accuracy"]["schedule_only"],
        "final_only_accuracy": lambda test: test["accuracy"]["final_only"],
        "deranged_accuracy": lambda test: test["accuracy"]["deranged"],
        "wrong_fact_accuracy": lambda test: test["accuracy"]["wrong_fact"],
        "wrong_slot_accuracy": lambda test: test["accuracy"]["wrong_slot"],
        "matched_minus_no_message": lambda test: test["contrasts"]["matched_minus_no_message"][
            "effect"
        ],
        "matched_minus_final_only": lambda test: test["contrasts"]["matched_minus_final_only"][
            "effect"
        ],
        "matched_minus_deranged": lambda test: test["contrasts"]["matched_minus_deranged"][
            "effect"
        ],
        "wrong_fact_target_rate": lambda test: test["wrong_fact_target_rate"],
        "wrong_slot_target_rate": lambda test: test["wrong_slot_target_rate"],
        "matched_pair_consistency": lambda test: test["matched_pair_consistency"],
    }
    all_tests_opened = all(member["sealed_test"] is not None for member in member_audits)
    if bool(aggregate_metrics["all_sealed_tests_opened"]) != all_tests_opened:
        raise AssertionError("aggregate test-opening flag differs from member artifacts")
    recomputed_aggregate: dict[str, dict[str, float]] = {}
    if all_tests_opened:
        for name, extractor in metric_extractors.items():
            values = [float(extractor(member["sealed_test"])) for member in member_audits]
            recomputed_aggregate[name] = summarize(values)
            for statistic_name, value in recomputed_aggregate[name].items():
                assert_close(
                    float(aggregate_metrics["aggregate"][name][statistic_name]),
                    value,
                    f"aggregate.{name}.{statistic_name}",
                )
    elif aggregate_metrics["aggregate"]:
        raise AssertionError("aggregate metrics were reported without all sealed tests")

    audit = {
        "audit_status": "AUDIT_PASS",
        "run_id": args.aggregate_dir.name,
        "verifier_pod": os.environ.get("HOSTNAME"),
        "verifier_gpu": args.gpu_name,
        "verifier_gpu_uuid": args.gpu_uuid,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "aggregate_files_verified": aggregate_files,
        "aggregate_inventory_sha256": sha256_file(args.aggregate_dir / "SHA256SUMS"),
        "source_manifest_sha256": source_manifest_sha256,
        "source_files_verified": source_files_verified,
        "member_count": 3,
        "members": member_audits,
        "common_sealed_test_sha256": common_test_digest,
        "all_seed_gates_passed": all_seed_passed,
        "recomputed_aggregate": recomputed_aggregate,
        "reported_aggregate_matches_raw": True,
        "verdict": verdict,
        "evidence_tier": ("prospective_three_training_seed_synthetic_dependency_span_confirmation"),
        "article_status": "BLOCKED_PENDING_TERMINAL_GATES",
    }
    args.audit_dir.mkdir(parents=True)
    audit_path = args.audit_dir / "independent_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.audit_dir / "SHA256SUMS").write_text(
        f"{sha256_file(audit_path)}  {audit_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
