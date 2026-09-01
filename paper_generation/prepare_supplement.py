#!/usr/bin/env python3
"""Build paper-facing evidence files from immutable run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def filter_jsonl(source: Path, destination: Path, predicate) -> int:
    count = 0
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            row = json.loads(line)
            if predicate(row):
                writer.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-metrics", type=Path, required=True)
    parser.add_argument("--phase5-predictions", type=Path, required=True)
    parser.add_argument("--phase5-channel", type=Path, required=True)
    parser.add_argument("--phase5-driver", type=Path, required=True)
    parser.add_argument("--phase8-metrics", type=Path, required=True)
    parser.add_argument("--phase8-predictions", type=Path, required=True)
    parser.add_argument("--phase8-channel", type=Path, required=True)
    parser.add_argument("--phase6-metrics", type=Path, required=True)
    parser.add_argument("--phase6-predictions", type=Path, action="append", required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite supplement: {args.output_root}")
    shutil.copytree(args.template_root, args.output_root)
    evidence = args.output_root / "evidence"
    checkpoints = args.output_root / "checkpoints"
    evidence.mkdir()
    checkpoints.mkdir()

    phase5 = json.loads(args.phase5_metrics.read_text(encoding="utf-8"))
    tiny_results = {
        "source_artifact_sha256": sha256(args.phase5_metrics),
        "verdict": phase5["verdict"],
        "canvas_assignment_count": phase5["canvas_assignment_count"],
        "training_steps": phase5["training_steps"],
        "sealed_test": phase5["sealed_test"],
    }
    write_json(evidence / "tiny_controlled_results.json", tiny_results)
    tiny_rows = filter_jsonl(
        args.phase5_predictions,
        evidence / "tiny_controlled_predictions.jsonl",
        lambda row: row.get("split") == "sealed-test",
    )
    expected_tiny_rows = phase5["sealed_test"]["examples"] * len(
        phase5["sealed_test"]["accuracy"]
    )
    if tiny_rows != expected_tiny_rows:
        raise RuntimeError(
            f"expected {expected_tiny_rows} Tiny-A2D prediction rows, found {tiny_rows}"
        )

    phase8 = json.loads(args.phase8_metrics.read_text(encoding="utf-8"))
    qasc_results = {
        "source_artifact_sha256": sha256(args.phase8_metrics),
        "qasc": phase8["qasc"],
        "integrity": phase8["integrity"],
        "channel_diagnostics": phase8["training"]["message"],
        "prefix_cosine": phase8["training"]["prefix_cosine"],
    }
    write_json(evidence / "qasc_results.json", qasc_results)
    qasc_rows = filter_jsonl(
        args.phase8_predictions,
        evidence / "qasc_predictions.jsonl",
        lambda row: row.get("benchmark") == "qasc",
    )
    if qasc_rows != 1000:
        raise RuntimeError(f"expected 1000 QASC prediction rows, found {qasc_rows}")

    phase6 = json.loads(args.phase6_metrics.read_text(encoding="utf-8"))
    controlled_results = {
        "source_artifact_sha256": sha256(args.phase6_metrics),
        "verdict": phase6["verdict"],
        "all_seed_gates_passed": phase6["all_seed_gates_passed"],
        "aggregate": phase6["aggregate"],
        "replicates": [
            {
                "channel_training_seeds": row["channel_training_seeds"],
                "sealed_test": row["sealed_test"],
            }
            for row in phase6["per_seed"]
        ],
    }
    write_json(evidence / "controlled_results.json", controlled_results)
    if len(args.phase6_predictions) != 3:
        raise RuntimeError("expected exactly three controlled prediction artifacts")
    for index, source in enumerate(args.phase6_predictions, start=1):
        rows = filter_jsonl(
            source,
            evidence / f"controlled_predictions_rep{index}.jsonl",
            lambda row: row.get("endpoint") == "development_test",
        )
        if rows != 10240:
            raise RuntimeError(
                f"expected 10240 controlled rows for replicate {index}, found {rows}"
            )

    shutil.copy2(args.phase5_channel, checkpoints / "tiny_a2d_channel.pt")
    shutil.copy2(args.phase8_channel, checkpoints / "qasc_dense_latent_prefix_channel.pt")
    shutil.copy2(
        args.phase5_driver,
        args.output_root / "code" / "scripts" / "phase5_hm_synth_tiny.py",
    )


if __name__ == "__main__":
    main()
