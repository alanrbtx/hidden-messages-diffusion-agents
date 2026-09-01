"""Build the frozen Stage 7A QASC-Distributed factual adapter package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import hf_hub_download

from hidden_messages.datasets.qasc import (
    QASC_SOURCE_ID,
    QASC_SOURCE_REVISION,
    build_qasc_training_fact_pool,
    partition_qasc_row,
    select_qasc_distractors,
)
from hidden_messages.utils.checksums import sha256_file
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json

SOURCE_FILES = {
    "train": (
        "data/train-00000-of-00001.parquet",
        "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1",
        8134,
    ),
    "validation": (
        "data/validation-00000-of-00001.parquet",
        "d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae",
        926,
    ),
    "test": (
        "data/test-00000-of-00001.parquet",
        "495cfbd17abc1720b54785cdce68825104aaf60d7b8bdb2acac62157a31eb517",
        920,
    ),
}
DERIVED_VALIDATION_ROWS = 1000
SPLIT_SALT = "qasc-distributed-v1-derived-validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--source-file-root", type=Path)
    parser.add_argument("--dataset-revision", default=QASC_SOURCE_REVISION)
    parser.add_argument("--project-source-revision", required=True)
    parser.add_argument("--project-source-manifest", type=Path, required=True)
    parser.add_argument("--distractors-per-agent", type=int, default=4)
    return parser.parse_args()


def resolve_source_file(
    filename: str,
    *,
    revision: str,
    dataset_cache: Path,
    source_file_root: Path | None,
) -> Path:
    if source_file_root is not None:
        path = source_file_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(
        hf_hub_download(
            repo_id=QASC_SOURCE_ID,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=dataset_cache,
        )
    )


def dataset_rows(path: Path) -> list[dict[str, object]]:
    dataset = Dataset.from_parquet(str(path))
    return [dict(row) for row in dataset]


def write_jsonl(path: Path, records: list[str]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(record + "\n")


def derive_labeled_splits(
    official_train: list[dict[str, object]],
    official_validation: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    ranked = sorted(
        official_train,
        key=lambda row: hashlib.sha256(f"{SPLIT_SALT}\0{row['id']}".encode()).hexdigest(),
    )
    return {
        "train": ranked[DERIVED_VALIDATION_ROWS:],
        "validation": ranked[:DERIVED_VALIDATION_ROWS],
        "test": official_validation,
    }


def main() -> None:
    args = parse_args()
    if args.dataset_revision != QASC_SOURCE_REVISION:
        raise ValueError("dataset revision differs from the frozen Stage 7A source")
    if args.distractors_per_agent != 4:
        raise ValueError("Stage 7A freezes four distractors per agent")
    run_dir = create_run_directory(args.run_root, args.run_id)
    shutil.copyfile(args.project_source_manifest, run_dir / "source-manifest.sha256")

    source_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split, (filename, expected_sha256, expected_rows) in SOURCE_FILES.items():
        path = resolve_source_file(
            filename,
            revision=args.dataset_revision,
            dataset_cache=args.dataset_cache,
            source_file_root=args.source_file_root,
        )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"QASC {split} source checksum mismatch")
        rows = dataset_rows(path)
        if len(rows) != expected_rows:
            raise RuntimeError(f"QASC {split} row count differs from the frozen source")
        source_paths[split] = path
        rows_by_split[split] = rows
        source_records[split] = {
            "filename": filename,
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }

    split_id_sets = {
        split: {str(row["id"]) for row in rows} for split, rows in rows_by_split.items()
    }
    if any(len(ids) != len(rows_by_split[split]) for split, ids in split_id_sets.items()):
        raise RuntimeError("QASC contains duplicate source IDs inside a split")
    split_names = tuple(rows_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_id_sets[left] & split_id_sets[right]:
                raise RuntimeError(f"QASC source IDs overlap between {left} and {right}")

    labeled_rows = derive_labeled_splits(rows_by_split["train"], rows_by_split["validation"])
    training_pool = build_qasc_training_fact_pool(labeled_rows["train"])
    output_records: dict[str, list[str]] = {}
    context_word_gaps: dict[str, list[int]] = {}
    emitted_ids: set[str] = set()
    for split, rows in labeled_rows.items():
        serialized: list[str] = []
        gaps: list[int] = []
        for row in rows:
            distractors = select_qasc_distractors(
                row,
                training_pool,
                count_per_agent=args.distractors_per_agent,
            )
            example = partition_qasc_row(
                row,
                source_split=split,
                distractors=distractors,
                source_revision=args.dataset_revision,
            )
            if example.example_id in emitted_ids:
                raise RuntimeError(f"duplicate emitted ID: {example.example_id}")
            emitted_ids.add(example.example_id)
            if len(example.private_contexts) != 2 or any(
                len(group) != 1 for group in example.support_ids
            ):
                raise RuntimeError("QASC partition violated the two-agent provenance contract")
            gaps.append(
                abs(
                    len(example.private_contexts[0].split())
                    - len(example.private_contexts[1].split())
                )
            )
            serialized.append(example.canonical_json())
        output_records[split] = serialized
        context_word_gaps[split] = gaps
        write_jsonl(run_dir / f"{split}.jsonl", serialized)

    test_source_ids = sorted(str(row["id"]) for row in labeled_rows["test"])
    (run_dir / "sealed_test_ids.txt").write_text(
        "\n".join(test_source_ids) + "\n", encoding="utf-8"
    )
    write_immutable_json(
        run_dir / "source_files.json",
        {
            "dataset_id": QASC_SOURCE_ID,
            "dataset_revision": args.dataset_revision,
            "files": source_records,
        },
    )
    write_immutable_json(
        run_dir / "dataset_card.json",
        {
            "dataset_id": QASC_SOURCE_ID,
            "dataset_revision": args.dataset_revision,
            "license": "CC-BY-4.0",
            "adapter": "qasc-distributed-v1",
            "designated_receiver": 0,
            "num_agents": 2,
            "distractors_per_agent": args.distractors_per_agent,
            "distractor_pool": "deduplicated derived-training gold facts only",
            "split_policy": {
                "train": "7134 official-train rows after SHA-256 holdout",
                "validation": "1000 official-train rows with lowest salted SHA-256 rank",
                "test": "926 official-validation rows, sealed",
                "official_test": "unlabeled submission-only; not emitted",
                "salt": SPLIT_SALT,
            },
            "test_policy": "official validation is sealed; no model evaluation in Stage 7A",
            "counterfactual_status": "not_materialized_pending_separate_protocol",
        },
    )
    write_immutable_json(
        run_dir / "schema_report.json",
        {
            "gate_status": "PASS_PHASE7A_QASC_ADAPTER",
            "split_rows": {split: len(records) for split, records in output_records.items()},
            "official_source_rows": {split: len(rows) for split, rows in rows_by_split.items()},
            "unique_emitted_ids": len(emitted_ids),
            "training_fact_pool_size": len(training_pool),
            "context_word_gap": {
                split: {
                    "mean": statistics.mean(gaps),
                    "max": max(gaps),
                }
                for split, gaps in context_word_gaps.items()
            },
            "source_split_overlap_count": 0,
            "proposed_method_predictions_used": False,
            "sealed_test_evaluated": False,
        },
    )
    write_immutable_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "stage": "engineering_data_audit",
            "protocol_id": "phase7a-qasc-distributed-v1",
            "project_source_revision": args.project_source_revision,
            "dataset_revision": args.dataset_revision,
            "command": sys.argv,
            "verdict": "PASS_PHASE7A_QASC_ADAPTER",
            "article_status": "BLOCKED_PENDING_TERMINAL_GATES",
        },
    )
    write_immutable_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    stdout = json.dumps(
        {
            "source_sha256": {split: record["sha256"] for split, record in source_records.items()},
            "split_rows": {split: len(records) for split, records in output_records.items()},
            "verdict": "PASS_PHASE7A_QASC_ADAPTER",
        },
        sort_keys=True,
    )
    (run_dir / "stdout.log").write_text(stdout + "\n", encoding="utf-8")
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(stdout)


if __name__ == "__main__":
    main()
