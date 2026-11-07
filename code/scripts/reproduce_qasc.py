"""Inference-only QASC release check, using the unchanged original evaluation functions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import dllm
import phase8_qasc_hiddenbench_dense_prefix as original
import torch
from phase5_hm_synth_tiny import model_weight_manifest
from phase6_dream_input_prefix import embedding_rms
from phase8_qasc_hiddenbench_dense_prefix import (
    evaluate_condition,
    load_config,
    load_qasc_rows,
    qasc_derangement_order,
    tokenize_cohort,
    verify_decoder_parity,
)

from hidden_messages.communication import DenseLatentPrefixChannel
from hidden_messages.utils.checksums import sha256_file
from hidden_messages.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True,
                        help="HF cache containing the pinned Dream snapshot")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing directories are never overwritten")
    parser.add_argument("--config", type=Path,
                        default=root / "configs/qasc_original_evaluation.yaml")
    parser.add_argument("--data", type=Path, default=root / "data/qasc/validation.jsonl")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "checkpoints/qasc_dense_latent_prefix_channel.pt")
    parser.add_argument("--reference", type=Path,
                        default=root / "evidence/qasc_predictions.jsonl")
    parser.add_argument("--conditions", nargs="+", default=["final_only", "matched"],
                        choices=["no_message", "matched", "final_only", "deranged",
                                 "zero", "random", "self"])
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--record-messages", action="store_true",
                        help="Save sender states and encoded messages at each exchange")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QASC reproduction requires CUDA")
    gpu = torch.cuda.get_device_name(0)
    if args.expected_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"Expected {args.expected_gpu!r}, found {gpu!r}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = load_config(args.config)
    shutil.copyfile(args.config, args.output_dir / "config.yaml")
    qasc = config["qasc"]
    hashes = {}
    for key, path, expected in (
        ("data", args.data, qasc["development_sha256"]),
        ("checkpoint", args.checkpoint, config["post_hoc"]["source_channel_sha256"]),
    ):
        hashes[key] = sha256_file(path)
        if hashes[key] != expected:
            raise RuntimeError(f"{key} SHA-256 differs from the original run")
    hashes["config"] = sha256_file(args.config)
    hashes["reference"] = sha256_file(args.reference)
    rows = load_qasc_rows(args.data, expected_rows=int(qasc["development_rows"]),
                          expected_split="validation")
    references = [json.loads(line) for line in args.reference.read_text().splitlines()]
    if (len(references) != len(rows)
            or len({record["example_id"] for record in references}) != len(rows)):
        raise RuntimeError("Reference must contain one unique record per development example")
    reference = {record["example_id"]: record for record in references}
    if set(reference) != {row.example_id for row in rows}:
        raise RuntimeError("Reference and input example IDs differ")
    for row in rows:
        if (reference[row.example_id]["answer_label"] != row.answer_label
                or reference[row.example_id]["designated_receiver"] != 0):
            raise RuntimeError("Reference labels or designated receiver differ")
    seed_everything(int(config["training"]["seed"]))
    device = torch.device("cuda")
    mc = config["model"]
    snapshot = (args.model_cache / ("models--" + mc["id"].replace("/", "--"))
                / "snapshots" / mc["revision"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Download the pinned Dream snapshot first: {snapshot}")
    hashes["model_weights"], _ = model_weight_manifest(snapshot)
    if hashes["model_weights"] != mc["weight_manifest_sha256"]:
        raise RuntimeError("Dream weight manifest differs from the original run")
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot), dtype=mc["dtype"],
        attn_implementation=mc["requested_attention_implementation"],
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    model.requires_grad_(False)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    decoder = model.get_decoder()
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    cohort = tokenize_cohort(rows, benchmark="qasc", tokenizer=tokenizer, config=config)
    if args.record_messages:
        with (args.output_dir / "token_rows.jsonl").open("x") as handle:
            for row, tokens in zip(rows, cohort.token_rows, strict=True):
                handle.write(json.dumps({"example_id": row.example_id,
                                         "agent_token_ids": tokens}) + "\n")
    parity = verify_decoder_parity(
        model, decoder, embedding, head, tokenizer, cohort,
        output_position_offset=int(mc["output_position_offset"]), device=device,
    )
    cc = config["channel"]
    channel = DenseLatentPrefixChannel(
        int(model.config.hidden_size), len(config["surface"]["qasc_labels"]),
        input_embedding_rms=embedding_rms(embedding),
        message_slots=int(config["surface"]["message_slots"]),
        message_dim=int(cc["message_dim"]), width=int(cc["width"]),
        classifier_width=int(cc["classifier_width"]),
    ).to(device)
    channel.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True),
                            strict=True)
    channel.eval().requires_grad_(False)
    ev = config["evaluation"]
    order = qasc_derangement_order(rows, salt=ev["derangement_salt"])
    outputs = [{"example_id": row.example_id, "answer_label": row.answer_label,
                "designated_receiver": 0, "conditions": {}} for row in rows]
    summary = {
        "stage": "release_validation", "examples": len(rows), "hashes": hashes,
        "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "record_messages": args.record_messages,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
            for directory in (Path(__file__).resolve().parent,
                              Path(__file__).resolve().parents[1] / "src")
            for path in sorted(directory.rglob("*.py"))
        },
        "model_metadata_sha256": {
            path.name: sha256_file(path)
            for path in sorted(snapshot.iterdir())
            if path.is_file() and path.suffix in {".json", ".jinja", ".model", ".txt"}
        },
        "packages": {name: importlib.metadata.version(name) for name in
                     ("transformers", "accelerate", "numpy", "peft", "huggingface-hub")},
        "decoder_parity": parity, "conditions": {},
    }
    mismatches = []
    labels = tuple(config["surface"]["qasc_labels"])
    original_encode = original.encode_all
    current_condition = ""
    exchange = 0

    def record_encode(channel, hidden, *, batch_size, device):
        nonlocal exchange
        messages = original_encode(channel, hidden, batch_size=batch_size, device=device)
        exchange += 1
        torch.save({
            "condition": current_condition, "exchange": exchange,
            "example_ids": [row.example_id for row in rows],
            "agent_order": [0, 1],
            "sender_states": hidden.detach().cpu().reshape(len(rows), 2, -1),
            "encoded_messages": messages.detach().cpu().reshape(
                len(rows), 2, channel.message_slots, channel.message_dim),
            "location": "before intervention, incoming aggregation and projection",
        }, args.output_dir / f"{current_condition}_exchange_{exchange:02d}.pt")
        return messages

    for condition in dict.fromkeys(args.conditions):
        print(f"Evaluating {condition}: {len(rows)} examples", flush=True)
        current_condition, exchange = condition, 0
        if args.record_messages:
            original.encode_all = record_encode
        try:
            predictions, profile = evaluate_condition(
                channel, model, decoder, embedding, head, tokenizer, cohort,
                condition=condition, derangement_order=order,
                batch_examples=int(ev["qasc_batch_examples"]),
                diagnostic_batch_size=int(config["training"]["diagnostic_batch_size"]),
                exchange_rounds=int(cc["exchange_rounds"]),
                output_position_offset=int(mc["output_position_offset"]),
                random_seed=int(ev["random_message_seed"])
                + 100 * ev["condition_order"].index(condition),
                no_message_receiver_batch_examples=8, device=device,
            )
        finally:
            original.encode_all = original_encode
        correct = identical = 0
        for row, output, index in zip(rows, outputs, predictions.tolist(), strict=True):
            label = labels[index]
            expected = reference[row.example_id]["conditions"][condition]["label"]
            correct += label == row.answer_label
            identical += label == expected
            output["conditions"][condition] = {"label": label, "correct": label == row.answer_label}
            if label != expected:
                mismatches.append({"example_id": row.example_id, "condition": condition,
                                   "expected": expected, "observed": label})
        summary["conditions"][condition] = {
            "correct": correct, "accuracy": correct / len(rows),
            "reference_labels_equal": identical, "profile": profile,
        }
        # Preserve completed conditions even if a later condition fails.
        with (args.output_dir / "predictions.jsonl").open("w") as handle:
            for output in outputs:
                handle.write(json.dumps(output, sort_keys=True) + "\n")
        print(f"{condition}: {correct}/{len(rows)} correct, "
              f"{identical}/{len(rows)} labels match the release", flush=True)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["all_requested_predictions_match"] = not mismatches
    (args.output_dir / "mismatches.json").write_text(json.dumps(mismatches, indent=2) + "\n")
    (args.output_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    if mismatches:
        raise SystemExit("Prediction mismatches recorded; this is not an exact reproduction")


if __name__ == "__main__":
    main()
