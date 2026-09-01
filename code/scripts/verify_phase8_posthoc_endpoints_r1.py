"""Independently verify the Phase 8 post-hoc QASC/HiddenBench endpoint artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]

from hidden_messages.utils.checksums import sha256_file
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json
from hidden_messages.utils.reproducibility import require_authorized_cuda

CONDITIONS = (
    "no_message",
    "matched",
    "final_only",
    "deranged",
    "zero",
    "random",
    "self",
)
SOURCE_FILES = {
    "config.yaml",
    "dataset_manifest.json",
    "design.json",
    "environment.json",
    "git_commit.txt",
    "metrics.json",
    "model_manifest.json",
    "prefix_training_log.jsonl",
    "profile.json",
    "qasc_dense_latent_prefix_channel.pt",
    "representation_training_log.jsonl",
    "run.json",
    "source-manifest.sha256",
    "stdout.log",
}
MEASUREMENT_FILES = SOURCE_FILES | {"predictions.jsonl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-run-dir", type=Path, required=True)
    parser.add_argument("--source-training-run-dir", type=Path, required=True)
    parser.add_argument("--qasc-adapter-run-dir", type=Path, required=True)
    parser.add_argument("--qasc-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--hiddenbench-adapter-run-dir", type=Path, required=True)
    parser.add_argument("--hiddenbench-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--measurement-source-manifest", type=Path, required=True)
    parser.add_argument("--audit-source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="H100")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one YAML mapping")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"{path.name} line {line_number} lacks a terminal newline")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name} line {line_number} is not an object")
            records.append(value)
    return records


def verify_inventory(run_dir: Path, expected_files: set[str]) -> dict[str, str]:
    inventory_path = run_dir / "SHA256SUMS"
    records: dict[str, str] = {}
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
            or Path(name).name != name
            or name in records
        ):
            raise ValueError(f"invalid inventory line in {inventory_path}: {line!r}")
        records[name] = digest
    if set(records) != expected_files:
        raise AssertionError(f"{run_dir.name} inventory file set differs")
    actual = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != expected_files:
        raise AssertionError(f"{run_dir.name} actual artifact set differs")
    for name, expected in records.items():
        if sha256_file(run_dir / name) != expected:
            raise AssertionError(f"artifact checksum mismatch: {run_dir.name}/{name}")
    return records


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def assert_close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError(f"{label} keys differ")
        for key in expected:
            assert_close(observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"{label} list shape differs")
        for index, value in enumerate(expected):
            assert_close(observed[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(finite_float(observed, label), expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{label} differs: {observed!r} != {expected!r}")
    elif observed != expected:
        raise AssertionError(f"{label} differs: {observed!r} != {expected!r}")


def read_id_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_baseline_labels(path: Path, benchmark: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in read_jsonl(path / "predictions.jsonl"):
        example_id = str(record["example_id"])
        if benchmark == "qasc":
            label = record["conditions"]["agent0_local"]["constrained_label"]
        else:
            label = record["base_conditions"]["receiver_local"]["constrained_label"]
        labels[example_id] = str(label)
    return labels


def load_qasc_truth(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        if row.get("source_split") != "validation" or row.get("designated_receiver") != 0:
            raise AssertionError("QASC validation identity changed")
    return rows


def load_hiddenbench_truth(path: Path, task_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if str(row["source_example_id"]) in task_ids:
            rows.append(row)
    return rows


def paired_bootstrap(
    differences: torch.Tensor, *, replicates: int, seed: int, device: torch.device
) -> list[float]:
    values = differences.float().to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    estimates: list[torch.Tensor] = []
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = torch.randint(
            0,
            values.numel(),
            (count, values.numel()),
            generator=generator,
            device=device,
        )
        estimates.append(values[indices].mean(dim=1))
    quantiles = torch.quantile(torch.cat(estimates), torch.tensor([0.025, 0.975], device=device))
    return [float(value) for value in quantiles.cpu().tolist()]


def accuracy(values: torch.Tensor, mask: torch.Tensor | None = None) -> float | None:
    selected = values if mask is None else values[mask]
    return None if selected.numel() == 0 else float(selected.float().mean().item())


def task_differences(
    left: torch.Tensor, right: torch.Tensor, records: list[dict[str, Any]]
) -> torch.Tensor:
    task_ids = sorted({str(record["source_task_id"]) for record in records}, key=int)
    values: list[torch.Tensor] = []
    for task_id in task_ids:
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["source_task_id"]) == task_id
        ]
        if len(indices) != 5:
            raise AssertionError("HiddenBench cluster does not contain five permutations")
        values.append((left[indices].float() - right[indices].float()).mean())
    return torch.stack(values)


def recompute_metrics(
    qasc_records: list[dict[str, Any]],
    hiddenbench_records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    correct: dict[str, dict[str, torch.Tensor]] = {}
    for benchmark, records in (
        ("qasc", qasc_records),
        ("hiddenbench", hiddenbench_records),
    ):
        correct[benchmark] = {
            condition: torch.tensor(
                [bool(record["conditions"][condition]["correct"]) for record in records],
                dtype=torch.bool,
            )
            for condition in CONDITIONS
        }

    evaluation = config["evaluation"]
    replicates = int(evaluation["bootstrap_replicates"])
    qasc_strict = torch.tensor(
        [bool(record["union_required"]) for record in qasc_records], dtype=torch.bool
    )
    qasc_metrics: dict[str, Any] = {
        "examples": len(qasc_records),
        "union_required_examples": int(qasc_strict.sum().item()),
        "accuracy": {condition: accuracy(correct["qasc"][condition]) for condition in CONDITIONS},
        "union_required_accuracy": {
            condition: accuracy(correct["qasc"][condition], qasc_strict) for condition in CONDITIONS
        },
    }
    for offset, reference in enumerate(("no_message", "deranged", "final_only")):
        key = f"matched_minus_{reference}"
        differences = correct["qasc"]["matched"].float() - correct["qasc"][reference].float()
        qasc_metrics[key] = float(differences.mean().item())
        qasc_metrics[f"{key}_paired_bootstrap_ci95"] = paired_bootstrap(
            differences,
            replicates=replicates,
            seed=int(evaluation["qasc_bootstrap_seed"]) + offset,
            device=device,
        )

    hb_strict = torch.tensor(
        [bool(record["context_required"]) for record in hiddenbench_records], dtype=torch.bool
    )
    n4 = torch.tensor(
        [int(record["group_size"]) == 4 for record in hiddenbench_records], dtype=torch.bool
    )
    n3 = ~n4
    hb_metrics: dict[str, Any] = {
        "examples": len(hiddenbench_records),
        "source_tasks": len({str(record["source_task_id"]) for record in hiddenbench_records}),
        "context_required_examples": int(hb_strict.sum().item()),
        "context_required_source_tasks": len(
            {
                str(record["source_task_id"])
                for record, include in zip(hiddenbench_records, hb_strict.tolist(), strict=True)
                if include
            }
        ),
        "accuracy": {
            condition: accuracy(correct["hiddenbench"][condition]) for condition in CONDITIONS
        },
        "context_required_accuracy": {
            condition: accuracy(correct["hiddenbench"][condition], hb_strict)
            for condition in CONDITIONS
        },
        "n4_accuracy": {
            condition: accuracy(correct["hiddenbench"][condition], n4) for condition in CONDITIONS
        },
        "n3_accuracy": {
            condition: accuracy(correct["hiddenbench"][condition], n3) for condition in CONDITIONS
        },
    }
    for offset, reference in enumerate(("no_message", "deranged", "final_only")):
        key = f"matched_minus_{reference}"
        values = task_differences(
            correct["hiddenbench"]["matched"],
            correct["hiddenbench"][reference],
            hiddenbench_records,
        )
        hb_metrics[key] = float(values.mean().item())
        hb_metrics[f"{key}_task_cluster_bootstrap_ci95"] = paired_bootstrap(
            values,
            replicates=replicates,
            seed=int(evaluation["hiddenbench_bootstrap_seed"]) + offset,
            device=device,
        )
    n4_records = [
        record for record, include in zip(hiddenbench_records, n4.tolist(), strict=True) if include
    ]
    n4_values = task_differences(
        correct["hiddenbench"]["matched"][n4],
        correct["hiddenbench"]["no_message"][n4],
        n4_records,
    )
    hb_metrics["n4_matched_minus_no_message"] = float(n4_values.mean().item())
    hb_metrics["n4_matched_minus_no_message_task_cluster_bootstrap_ci95"] = paired_bootstrap(
        n4_values,
        replicates=replicates,
        seed=int(evaluation["hiddenbench_bootstrap_seed"]) + 10,
        device=device,
    )
    return {"qasc": qasc_metrics, "hiddenbench": hb_metrics}


def recompute_gate(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    qasc = metrics["qasc"]
    hiddenbench = metrics["hiddenbench"]
    qgate = config["evaluation"]["qasc_gate"]
    hgate = config["evaluation"]["hiddenbench_gate"]
    qasc_checks = {
        "matched_minus_no_message_min": qasc["matched_minus_no_message"]
        >= float(qgate["matched_minus_no_message_min"]),
        "matched_minus_deranged_min": qasc["matched_minus_deranged"]
        >= float(qgate["matched_minus_deranged_min"]),
        "matched_minus_final_only_min": qasc["matched_minus_final_only"]
        >= float(qgate["matched_minus_final_only_min"]),
        "all_ci_low_above_zero": all(
            qasc[f"matched_minus_{reference}_paired_bootstrap_ci95"][0] > 0
            for reference in ("no_message", "deranged", "final_only")
        ),
    }
    hiddenbench_checks = {
        "matched_minus_no_message_min": hiddenbench["matched_minus_no_message"]
        >= float(hgate["matched_minus_no_message_min"]),
        "matched_minus_deranged_min": hiddenbench["matched_minus_deranged"]
        >= float(hgate["matched_minus_deranged_min"]),
        "matched_minus_final_only_min": hiddenbench["matched_minus_final_only"]
        >= float(hgate["matched_minus_final_only_min"]),
        "all_ci_low_above_zero": all(
            hiddenbench[f"matched_minus_{reference}_task_cluster_bootstrap_ci95"][0] > 0
            for reference in ("no_message", "deranged", "final_only")
        ),
        "n4_matched_minus_no_message_min": hiddenbench["n4_matched_minus_no_message"]
        >= float(hgate["n4_matched_minus_no_message_min"]),
    }
    return {
        **{f"qasc_{key}": value for key, value in qasc_checks.items()},
        **{f"hiddenbench_{key}": value for key, value in hiddenbench_checks.items()},
        "qasc_pass": all(qasc_checks.values()),
        "hiddenbench_pass": all(hiddenbench_checks.values()),
        "exact_no_message_parity": True,
        "independent_agent_canvases": True,
        "designated_receiver_no_vote": True,
    }


def validate_prediction_records(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    dataset: dict[str, Any],
    qasc_adapter: Path,
    qasc_baseline: Path,
    hiddenbench_adapter: Path,
    hiddenbench_baseline: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qasc_records = [record for record in records if record.get("benchmark") == "qasc"]
    hb_records = [record for record in records if record.get("benchmark") == "hiddenbench"]
    if len(qasc_records) != 1000 or len(hb_records) != 100:
        raise AssertionError("endpoint prediction row counts differ from the frozen cohorts")
    if len(records) != len(qasc_records) + len(hb_records):
        raise AssertionError("predictions contain an unknown benchmark")

    qasc_truth = load_qasc_truth(qasc_adapter / str(config["qasc"]["development_file"]))
    task_ids = set(str(value) for value in dataset["hiddenbench"]["development_task_ids"])
    hb_truth = load_hiddenbench_truth(
        hiddenbench_adapter / str(config["hiddenbench"]["file"]), task_ids
    )
    if [record["example_id"] for record in qasc_records] != [
        row["example_id"] for row in qasc_truth
    ]:
        raise AssertionError("QASC prediction order/identity differs from the frozen data")
    if [record["example_id"] for record in hb_records] != [row["example_id"] for row in hb_truth]:
        raise AssertionError("HiddenBench prediction order/identity differs from the frozen data")

    qasc_baseline_labels = load_baseline_labels(qasc_baseline, "qasc")
    hb_baseline_labels = load_baseline_labels(hiddenbench_baseline, "hiddenbench")
    qasc_strict_ids = read_id_set(qasc_baseline / str(config["qasc"]["union_required_ids_file"]))
    hb_strict_ids = read_id_set(
        hiddenbench_baseline / str(config["hiddenbench"]["context_required_ids_file"])
    )
    truth_by_benchmark = {"qasc": qasc_truth, "hiddenbench": hb_truth}
    baseline_by_benchmark = {"qasc": qasc_baseline_labels, "hiddenbench": hb_baseline_labels}
    strict_by_benchmark = {"qasc": qasc_strict_ids, "hiddenbench": hb_strict_ids}
    strict_name = {"qasc": "union_required", "hiddenbench": "context_required"}
    valid_labels = {"qasc": set("ABCDEFGH"), "hiddenbench": set("ABCD")}

    for benchmark, benchmark_records in (("qasc", qasc_records), ("hiddenbench", hb_records)):
        for record, truth in zip(benchmark_records, truth_by_benchmark[benchmark], strict=True):
            if (
                record.get("answer_label") != truth.get("answer_label")
                or record.get("source_task_id") != str(truth["source_example_id"])
                or record.get("designated_receiver") != 0
                or set(record.get("conditions", {})) != set(CONDITIONS)
            ):
                raise AssertionError(f"{benchmark} prediction metadata differs from truth")
            if benchmark == "hiddenbench":
                metadata = truth["metadata"]
                if (
                    record.get("group_size") != len(truth["private_contexts"])
                    or record.get("permutation_index") != metadata["permutation_index"]
                ):
                    raise AssertionError("HiddenBench group/permutation identity differs")
            flag_name = strict_name[benchmark]
            if record.get(flag_name) is not (
                str(record["example_id"]) in strict_by_benchmark[benchmark]
            ):
                raise AssertionError(f"{benchmark} registered subset flag differs")
            for condition in CONDITIONS:
                outcome = record["conditions"][condition]
                label = outcome.get("label")
                if label not in valid_labels[benchmark]:
                    raise AssertionError(f"{benchmark} invalid constrained label")
                if outcome.get("correct") is not (label == record["answer_label"]):
                    raise AssertionError(f"{benchmark} correctness flag differs")
            if (
                record["conditions"]["no_message"]["label"]
                != baseline_by_benchmark[benchmark][str(record["example_id"])]
            ):
                raise AssertionError(f"{benchmark} no-message parity failed")

    counts = Counter(str(record["source_task_id"]) for record in hb_records)
    if len(counts) != 20 or set(counts.values()) != {5}:
        raise AssertionError("HiddenBench development task clusters changed")
    return qasc_records, hb_records


def write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def main() -> None:
    args = parse_args()
    gpu = require_authorized_cuda(args.expected_gpu)
    device = torch.device("cuda")

    source_inventory = verify_inventory(args.source_training_run_dir, SOURCE_FILES)
    measurement_inventory = verify_inventory(args.measurement_run_dir, MEASUREMENT_FILES)
    source_run = load_json(args.source_training_run_dir / "run.json")
    source_metrics = load_json(args.source_training_run_dir / "metrics.json")
    source_model = load_json(args.source_training_run_dir / "model_manifest.json")
    source_config = load_yaml(args.source_training_run_dir / "config.yaml")
    config = load_yaml(args.measurement_run_dir / "config.yaml")
    run = load_json(args.measurement_run_dir / "run.json")
    metrics = load_json(args.measurement_run_dir / "metrics.json")
    dataset = load_json(args.measurement_run_dir / "dataset_manifest.json")
    model = load_json(args.measurement_run_dir / "model_manifest.json")
    design = load_json(args.measurement_run_dir / "design.json")
    environment = load_json(args.measurement_run_dir / "environment.json")

    if source_run.get("verdict") != "STOP_PHASE8_TRAINING_GATE":
        raise AssertionError("source training STOP changed")
    if source_metrics.get("training_gate_passed") is not False:
        raise AssertionError("source training gate changed")
    if config.get("stage") != "post_hoc_endpoint_evaluation":
        raise AssertionError("measurement config is not post-hoc")
    source_scientific_config = dict(source_config)
    measurement_scientific_config = dict(config)
    for key in ("protocol_id", "stage", "non_claim"):
        source_scientific_config.pop(key)
    for key in ("protocol_id", "stage", "non_claim", "post_hoc"):
        measurement_scientific_config.pop(key)
    if source_scientific_config != measurement_scientific_config:
        raise AssertionError("post-hoc measurement changed a frozen scientific config field")
    if run.get("stage") != "post_hoc_endpoint_evaluation" or run.get("status") != "completed":
        raise AssertionError("measurement run did not complete as post-hoc")
    if run.get("evidence_tier") != "post_hoc_development_exploratory":
        raise AssertionError("measurement evidence tier changed")
    if metrics.get("verdict") != run.get("verdict"):
        raise AssertionError("measurement verdict artifacts disagree")
    if run.get("verdict") not in {
        "EXPLORATORY_PASS_PHASE8_QASC_HIDDENBENCH_ENDPOINTS",
        "EXPLORATORY_FAIL_PHASE8_QASC_HIDDENBENCH_ENDPOINTS",
    }:
        raise AssertionError("unexpected post-hoc verdict")
    if metrics.get("training_gate_passed") is not False:
        raise AssertionError("post-hoc run rewrote the original training gate")
    assert_close(metrics.get("training"), source_metrics["training"], "inherited training metrics")
    if metrics.get("training_gate_checks") != source_metrics.get("training_gate_checks"):
        raise AssertionError("post-hoc run changed the training checks")

    post_hoc = metrics.get("post_hoc_endpoint_opening")
    if not isinstance(post_hoc, dict) or post_hoc.get("checkpoint_retrained") is not False:
        raise AssertionError("post-hoc provenance is incomplete")
    source_inventory_sha256 = sha256_file(args.source_training_run_dir / "SHA256SUMS")
    if (
        post_hoc.get("source_inventory_sha256") != source_inventory_sha256
        or config["post_hoc"]["source_inventory_sha256"] != source_inventory_sha256
    ):
        raise AssertionError("source inventory provenance differs")
    source_channel_sha256 = sha256_file(
        args.source_training_run_dir / "qasc_dense_latent_prefix_channel.pt"
    )
    measurement_channel_sha256 = sha256_file(
        args.measurement_run_dir / "qasc_dense_latent_prefix_channel.pt"
    )
    if not (
        source_channel_sha256
        == measurement_channel_sha256
        == source_model["channel_sha256"]
        == model["channel_sha256"]
        == config["post_hoc"]["source_channel_sha256"]
    ):
        raise AssertionError("measurement did not reuse the exact source channel")
    if model.get("channel_retrained_in_this_run") is not False:
        raise AssertionError("measurement claims to have retrained the channel")
    if (
        design.get("post_hoc_training_gate_waiver") is not True
        or design.get("training_gate_threshold_changed") is not False
        or design.get("vote_used") is not False
        or design.get("receiver_canvas_replaced") is not False
    ):
        raise AssertionError("post-hoc design contract differs")
    integrity = metrics.get("integrity", {})
    if (
        integrity.get("exact_no_message_parity") is not True
        or integrity.get("canvas_assignment_count") != 0
        or integrity.get("designated_receiver_no_vote") is not True
        or integrity.get("sealed_qasc_test_predictions_opened") is not False
        or dataset["qasc_sealed_test"]["predictions_opened"] is not False
    ):
        raise AssertionError("measurement integrity gate failed")
    for benchmark in ("qasc", "hiddenbench"):
        parity = integrity["decoder_no_prefix_parity"][benchmark]
        if (
            parity.get("hidden_bfloat16_exact") is not True
            or parity.get("constrained_prediction_exact") is not True
            or parity.get("max_hidden_abs_difference") != 0.0
        ):
            raise AssertionError(f"{benchmark} decoder parity failed")
    if not str(environment.get("gpu", "")).startswith("NVIDIA H100"):
        raise AssertionError("measurement was not executed on H100")
    if sha256_file(args.measurement_source_manifest) != sha256_file(
        args.measurement_run_dir / "source-manifest.sha256"
    ):
        raise AssertionError("measurement source receipt differs")
    for input_dir, expected, label in (
        (
            args.qasc_adapter_run_dir,
            config["qasc"]["adapter_inventory_sha256"],
            "QASC adapter",
        ),
        (
            args.qasc_baseline_run_dir,
            config["qasc"]["baseline_inventory_sha256"],
            "QASC baseline",
        ),
        (
            args.hiddenbench_adapter_run_dir,
            config["hiddenbench"]["adapter_inventory_sha256"],
            "HiddenBench adapter",
        ),
        (
            args.hiddenbench_baseline_run_dir,
            config["hiddenbench"]["baseline_inventory_sha256"],
            "HiddenBench baseline",
        ),
    ):
        if sha256_file(input_dir / "SHA256SUMS") != str(expected):
            raise AssertionError(f"{label} inventory differs from the frozen config")
    for input_path, expected in (
        (
            args.qasc_adapter_run_dir / str(config["qasc"]["development_file"]),
            config["qasc"]["development_sha256"],
        ),
        (
            args.qasc_adapter_run_dir / str(config["qasc"]["sealed_test_file"]),
            config["qasc"]["sealed_test_sha256"],
        ),
        (
            args.hiddenbench_adapter_run_dir / str(config["hiddenbench"]["file"]),
            config["hiddenbench"]["file_sha256"],
        ),
    ):
        if sha256_file(input_path) != str(expected):
            raise AssertionError(f"dataset hash differs: {input_path.name}")

    records = read_jsonl(args.measurement_run_dir / "predictions.jsonl")
    qasc_records, hb_records = validate_prediction_records(
        records,
        config=config,
        dataset=dataset,
        qasc_adapter=args.qasc_adapter_run_dir,
        qasc_baseline=args.qasc_baseline_run_dir,
        hiddenbench_adapter=args.hiddenbench_adapter_run_dir,
        hiddenbench_baseline=args.hiddenbench_baseline_run_dir,
    )
    recomputed = recompute_metrics(qasc_records, hb_records, config, device=device)
    assert_close(metrics.get("qasc"), recomputed["qasc"], "qasc metrics")
    assert_close(metrics.get("hiddenbench"), recomputed["hiddenbench"], "hiddenbench metrics")
    gate_checks = recompute_gate(recomputed, config)
    if metrics.get("development_gate_checks") != gate_checks:
        raise AssertionError("stored endpoint gate checks differ from recomputation")
    gate_passed = all(gate_checks.values())
    if metrics.get("development_gate_passed") is not gate_passed:
        raise AssertionError("stored endpoint verdict differs from recomputation")
    expected_verdict = (
        "EXPLORATORY_PASS_PHASE8_QASC_HIDDENBENCH_ENDPOINTS"
        if gate_passed
        else "EXPLORATORY_FAIL_PHASE8_QASC_HIDDENBENCH_ENDPOINTS"
    )
    if run["verdict"] != expected_verdict:
        raise AssertionError("post-hoc verdict does not follow the unchanged endpoint gate")

    output_dir = create_run_directory(args.output_root, args.output_id)
    audit_status = "AUDIT_PASS_PHASE8_POSTHOC_ENDPOINT_ARTIFACTS"
    write_immutable_json(
        output_dir / "audit.json",
        {
            "audit_status": audit_status,
            "measurement_run_id": run["run_id"],
            "measurement_inventory_sha256": sha256_file(args.measurement_run_dir / "SHA256SUMS"),
            "source_training_run_id": source_run["run_id"],
            "source_training_inventory_sha256": source_inventory_sha256,
            "source_training_gate_passed": False,
            "source_channel_sha256": source_channel_sha256,
            "checkpoint_retrained": False,
            "endpoint_gate_passed": gate_passed,
            "endpoint_verdict": expected_verdict,
            "endpoint_gate_checks": gate_checks,
            "qasc": recomputed["qasc"],
            "hiddenbench": recomputed["hiddenbench"],
            "sealed_qasc_test_predictions_opened": False,
            "evidence_tier": "post_hoc_development_exploratory",
            "paper_ready": False,
        },
    )
    write_immutable_json(
        output_dir / "source_inventory.json",
        {
            "source_training": source_inventory,
            "measurement": measurement_inventory,
            "measurement_source_manifest_sha256": sha256_file(args.measurement_source_manifest),
            "audit_source_manifest_sha256": sha256_file(args.audit_source_manifest),
        },
    )
    write_immutable_json(
        output_dir / "environment.json",
        {"pod": os.environ.get("HOSTNAME"), "gpu": gpu, "torch": torch.__version__},
    )
    stdout = json.dumps(
        {
            "audit_status": audit_status,
            "endpoint_gate_passed": gate_passed,
            "endpoint_verdict": expected_verdict,
        },
        sort_keys=True,
    )
    write_text_exclusive(output_dir / "stdout.log", stdout + "\n")
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    write_text_exclusive(output_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    print(stdout, flush=True)
    while not args.release_file.exists():
        time.sleep(5)


if __name__ == "__main__":
    main()
