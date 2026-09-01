"""Aggregate the sealed v15 dependency-span confirmation members."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hidden_messages.utils.checksums import sha256_file
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--aggregate-run-id", required=True)
    parser.add_argument("--member-run-id", action="append", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_run_checksums(run_dir: Path) -> str:
    checksum_path = run_dir / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"empty checksum inventory: {checksum_path}")
    for line in lines:
        expected, separator, filename = line.partition("  ")
        if separator != "  " or not expected or Path(filename).name != filename:
            raise RuntimeError(f"invalid checksum line in {checksum_path}: {line!r}")
        artifact = run_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"checksum mismatch: {artifact}")
    return sha256_file(checksum_path)


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


def main() -> None:
    args = parse_args()
    if len(args.member_run_id) != 3 or len(set(args.member_run_id)) != 3:
        raise ValueError("dependency-span confirmation requires three distinct members")

    members: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for run_id in args.member_run_id:
        run_dir = args.run_root / run_id
        run = load_json(run_dir / "run.json")
        metrics = load_json(run_dir / "metrics.json")
        dataset = load_json(run_dir / "dataset_manifest.json")
        seeds = load_json(run_dir / "seed_manifest.json")
        model = load_json(run_dir / "model_manifest.json")
        profile = load_json(run_dir / "profile.json")
        inventory_sha256 = verify_run_checksums(run_dir)
        if run.get("run_id") != run_id or run.get("stage") != "confirmation":
            raise RuntimeError(f"invalid dependency-span confirmation member: {run_id}")
        if run.get("protocol_id") != "phase6-dream7b-dependency-span-confirmation-v15":
            raise RuntimeError(f"protocol identity differs in {run_id}")
        if run.get("source_revision") != args.source_revision:
            raise RuntimeError(f"source revision differs in {run_id}")
        members.append(
            {
                "run_id": run_id,
                "metrics": metrics,
                "dataset": dataset,
                "seeds": seeds,
                "model": model,
                "profile": profile,
                "sha256sums_sha256": inventory_sha256,
            }
        )
        configs.append(normalized_config(run_dir / "config.yaml"))

    reference = json.dumps(configs[0], sort_keys=True)
    if any(json.dumps(config, sort_keys=True) != reference for config in configs):
        raise RuntimeError("confirmation configs differ beyond frozen member seeds")

    seed_keys = (
        "round1_training_seed",
        "round1_prefix_seed",
        "round2_training_seed",
        "round2_prefix_seed",
    )
    if any(len({member["seeds"][key] for member in members}) != 3 for key in seed_keys):
        raise RuntimeError("one or more channel-training seed families are not distinct")
    train_digests = [member["dataset"]["train"]["sha256"] for member in members]
    validation_digests = [member["dataset"]["validation"]["sha256"] for member in members]
    round1_hashes = [member["model"]["round1_channel_sha256"] for member in members]
    round2_hashes = [member["model"]["round2_channel_sha256"] for member in members]
    if len(set(train_digests)) != 3 or len(set(validation_digests)) != 3:
        raise RuntimeError("train or validation cohorts are not distinct")
    if len(set(round1_hashes)) != 3 or len(set(round2_hashes)) != 3:
        raise RuntimeError("learned channel checkpoints are not distinct")

    test_seed_starts = {member["dataset"]["development_test"]["seed_start"] for member in members}
    if len(test_seed_starts) != 1:
        raise RuntimeError("members did not declare one common sealed test")
    test_opened = [
        bool(member["dataset"]["development_test"]["materialized"]) for member in members
    ]
    if any(
        opened != (member["metrics"]["development_test"] is not None)
        for member, opened in zip(members, test_opened, strict=True)
    ):
        raise RuntimeError("sealed-test materialization flag differs from member metrics")
    materialized_test_digests = {
        member["dataset"]["development_test"]["sha256"]
        for member, opened in zip(members, test_opened, strict=True)
        if opened
    }
    if None in materialized_test_digests or len(materialized_test_digests) > 1:
        raise RuntimeError("materialized sealed-test identities differ")
    all_tests_opened = all(test_opened)
    common_test_digest = (
        next(iter(materialized_test_digests)) if materialized_test_digests else None
    )

    common_lora = {member["model"]["lora_checkpoint_sha256"] for member in members}
    common_model = {member["model"]["weight_manifest_sha256"] for member in members}
    common_eval_gate = {
        json.dumps(member["metrics"]["evaluation_gate"], sort_keys=True) for member in members
    }
    common_training_gate = {
        json.dumps(member["metrics"]["training_gate"], sort_keys=True) for member in members
    }
    if any(
        len(values) != 1
        for values in (common_lora, common_model, common_eval_gate, common_training_gate)
    ):
        raise RuntimeError("model, LoRA, or frozen gates differ across members")

    seed_passes = [
        bool(member["metrics"]["training_gate_passed"])
        and bool(member["metrics"]["validation_gate_passed"])
        and bool(member["metrics"]["development_pilot_gate_passed"])
        and member["metrics"]["verdict"] == "confirmation_seed_passed"
        for member in members
    ]
    all_seed_passed = all(seed_passes)
    verdict = (
        "PASS_PHASE6_DEPENDENCY_SPAN_SYNTHETIC_MULTISEED"
        if all_seed_passed
        else "STOP_PHASE6_DEPENDENCY_SPAN_SYNTHETIC_MULTISEED"
    )

    endpoint_values: dict[str, list[float]] = {}
    per_seed: list[dict[str, Any]] = []
    for member, passed, opened in zip(members, seed_passes, test_opened, strict=True):
        test = member["metrics"]["development_test"]
        if opened:
            values = {
                "matched_accuracy": float(test["accuracy"]["matched"]),
                "no_message_accuracy": float(test["accuracy"]["no_message"]),
                "schedule_only_accuracy": float(test["accuracy"]["schedule_only"]),
                "final_only_accuracy": float(test["accuracy"]["final_only"]),
                "deranged_accuracy": float(test["accuracy"]["deranged"]),
                "wrong_fact_accuracy": float(test["accuracy"]["wrong_fact"]),
                "wrong_slot_accuracy": float(test["accuracy"]["wrong_slot"]),
                "matched_minus_no_message": float(
                    test["contrasts"]["matched_minus_no_message"]["effect"]
                ),
                "matched_minus_final_only": float(
                    test["contrasts"]["matched_minus_final_only"]["effect"]
                ),
                "matched_minus_deranged": float(
                    test["contrasts"]["matched_minus_deranged"]["effect"]
                ),
                "wrong_fact_target_rate": float(test["wrong_fact_target_rate"]),
                "wrong_slot_target_rate": float(test["wrong_slot_target_rate"]),
                "matched_pair_consistency": float(test["matched_pair_consistency"]),
            }
            for name, value in values.items():
                endpoint_values.setdefault(name, []).append(value)
        per_seed.append(
            {
                "run_id": member["run_id"],
                "channel_training_seeds": {key: member["seeds"][key] for key in seed_keys},
                "training_gate_passed": member["metrics"]["training_gate_passed"],
                "validation_gate_passed": member["metrics"]["validation_gate_passed"],
                "sealed_test_opened": opened,
                "sealed_test_gate_passed": member["metrics"]["development_pilot_gate_passed"],
                "member_verdict": member["metrics"]["verdict"],
                "gate_passed": passed,
                "round1_channel_sha256": member["model"]["round1_channel_sha256"],
                "round2_channel_sha256": member["model"]["round2_channel_sha256"],
                "sealed_test": test,
            }
        )

    aggregate_dir = create_run_directory(args.run_root, args.aggregate_run_id)
    shutil.copyfile(args.source_manifest, aggregate_dir / "source-manifest.sha256")
    write_immutable_json(
        aggregate_dir / "run.json",
        {
            "run_id": args.aggregate_run_id,
            "protocol_id": "phase6-dream7b-dependency-span-confirmation-v15",
            "stage": "confirmation_aggregate",
            "source_revision": args.source_revision,
            "status": "completed",
        },
    )
    write_immutable_json(
        aggregate_dir / "member_manifest.json",
        {
            "member_count": 3,
            "member_run_ids": args.member_run_id,
            "member_sha256sum_files_verified_before_aggregation": True,
            "members": [
                {
                    "run_id": member["run_id"],
                    "sha256sums_sha256": member["sha256sums_sha256"],
                    "train_sha256": member["dataset"]["train"]["sha256"],
                    "validation_sha256": member["dataset"]["validation"]["sha256"],
                    "sealed_test_sha256": member["dataset"]["development_test"]["sha256"],
                    "round1_channel_sha256": member["model"]["round1_channel_sha256"],
                    "round2_channel_sha256": member["model"]["round2_channel_sha256"],
                }
                for member in members
            ],
        },
    )
    write_immutable_json(
        aggregate_dir / "metrics.json",
        {
            "verdict": verdict,
            "evidence_tier": (
                "prospective_three_training_seed_synthetic_dependency_span_confirmation"
            ),
            "all_seed_gates_passed": all_seed_passed,
            "seed_count": 3,
            "training_gate": members[0]["metrics"]["training_gate"],
            "evaluation_gate": members[0]["metrics"]["evaluation_gate"],
            "declared_common_test_seed_start": next(iter(test_seed_starts)),
            "all_sealed_tests_opened": all_tests_opened,
            "common_test_sha256": common_test_digest,
            "common_lora_checkpoint_sha256": next(iter(common_lora)),
            "common_model_weight_manifest_sha256": next(iter(common_model)),
            "aggregate": (
                {name: summarize(values) for name, values in endpoint_values.items()}
                if all_tests_opened
                else {}
            ),
            "per_seed": per_seed,
            "non_claim": (
                "A pass confirms only supervised dense two-stage communication on the synthetic "
                "rendezvous family under a fixed Dream-7B LoRA. Natural benchmarks, matched "
                "baselines, shared-adapter transfer, a second backbone, and paper readiness "
                "remain open."
            ),
        },
    )
    write_immutable_json(
        aggregate_dir / "profile.json",
        {
            "member_wall_seconds_sum": sum(
                float(member["profile"]["wall_seconds"]) for member in members
            ),
            "member_peak_allocated_bytes_max": max(
                int(member["profile"]["peak_allocated_bytes"]) for member in members
            ),
        },
    )
    stdout = json.dumps(
        {"all_seed_gates_passed": all_seed_passed, "verdict": verdict}, sort_keys=True
    )
    (aggregate_dir / "stdout.log").write_text(stdout + "\n", encoding="utf-8")
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(aggregate_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (aggregate_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(stdout)


if __name__ == "__main__":
    main()
