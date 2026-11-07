"""Train and causally audit a Dream input-prefix channel with a fixed light LoRA."""

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
    build_split,
    gate_passes,
    model_weight_manifest,
    precompute_hidden,
    split_digest,
    tokenize_split,
)
from phase6_dream_input_prefix import (
    assert_no_message_parity,
    assert_pair_alignment,
    embedding_rms,
    evaluate_split,
    prediction_records,
    receiver_logits,
    sender_metadata,
    write_jsonl,
    write_zstd_jsonl,
)
from torch import nn

from hidden_messages.adaptation import (
    attach_lora_to_decoder_layers,
    load_lora_adapter_state_dict,
    lora_named_parameters,
)
from hidden_messages.communication import InputPrefixChannel, batch_message_variance_loss
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
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Dream LoRA input-prefix config must be a mapping")
    return payload


def load_adapter_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in payload.items()
    ):
        raise TypeError("LoRA checkpoint must be a tensor state dictionary")
    load_lora_adapter_state_dict(model, payload)


def prefix_teacher_table(
    tokenizer: Any,
    *,
    answer_values: tuple[str, ...],
    template: str,
    message_slots: int,
) -> dict[int, list[int]]:
    table: dict[int, list[int]] = {}
    for answer in answer_values:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(answer_ids) != 1:
            raise RuntimeError(f"answer {answer!r} is not a stable one-token target")
        phrase = template.format(answer=answer)
        phrase_ids = [int(value) for value in tokenizer.encode(phrase, add_special_tokens=False)]
        if len(phrase_ids) != message_slots:
            raise RuntimeError(
                f"prefix alignment phrase {phrase!r} has {len(phrase_ids)} tokens, "
                f"expected {message_slots}"
            )
        table[int(answer_ids[0])] = phrase_ids
    if len(table) != len(answer_values):
        raise RuntimeError("prefix alignment table has duplicate answer token IDs")
    return table


def _cosine_factor(step_index: int, *, steps: int, warmup_steps: int) -> float:
    if step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    progress = (step_index - warmup_steps) / max(steps - warmup_steps, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def train_channel(
    channel: InputPrefixChannel,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenized: TokenizedSplit,
    hidden_split: HiddenSplit,
    prefix_table: dict[int, list[int]],
    config: dict[str, Any],
    *,
    assistant_prefix_tokens: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    if batch_size < 2 or batch_size % 2:
        raise ValueError("training batch size must be a positive even number")
    if str(training["prefix_alignment_loss"]) != "mean_cosine_distance":
        raise ValueError("unsupported prefix alignment loss")
    pair_batch_size = batch_size // 2
    pair_count = len(hidden_split.examples) // 2
    steps = int(training["steps"])
    output_offset = int(config["model"]["output_position_offset"])
    normalized_time_value = float(training["normalized_time"])
    lambda_pair = float(training["lambda_pair"])
    pair_fraction = float(training["pair_fraction"])
    pair_margin = float(training["pair_margin"])
    lambda_variance = float(training["lambda_message_variance"])
    target_message_std = float(training["target_message_std"])
    lambda_alignment = float(training["lambda_prefix_alignment"])
    optimizer = torch.optim.AdamW(
        channel.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = max(1, round(steps * float(training["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step_index: _cosine_factor(
            step_index,
            steps=steps,
            warmup_steps=warmup_steps,
        ),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training["seed"]))
    permutation = torch.randperm(pair_count, generator=generator)
    cursor = 0
    records: list[dict[str, float | int]] = []
    channel.train()
    decoder.eval()
    for step in range(steps):
        if cursor + pair_batch_size > pair_count:
            permutation = torch.randperm(pair_count, generator=generator)
            cursor = 0
        selected_pairs = permutation[cursor : cursor + pair_batch_size]
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
            teacher_ids = torch.tensor(
                [prefix_table[int(target)] for target in targets.tolist()],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                target_prefixes = input_embedding(teacher_ids)
            prefix_cosine = nn.functional.cosine_similarity(
                prefixes.float(),
                target_prefixes.float(),
                dim=-1,
            ).mean()
            alignment_loss = 1.0 - prefix_cosine
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
                + lambda_pair * pair_loss
                + lambda_variance * variance_loss
                + lambda_alignment * alignment_loss
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
                    "pair_loss": float(pair_loss.item()),
                    "message_variance_loss": float(variance_loss.item()),
                    "prefix_alignment_loss": float(alignment_loss.item()),
                    "prefix_cosine": float(prefix_cosine.item()),
                    "message_std": float(message_std.item()),
                    "prefix_rms": float(prefixes.float().square().mean().sqrt().detach()),
                    "batch_accuracy": float((logits.argmax(dim=-1) == targets).float().mean()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
        del selected_hidden, selected_attention, selected_segments, logits, loss
    channel.eval()
    return records


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    confirmation_run = config["stage"] == "confirmation"
    test_split_name = "confirmation_test" if confirmation_run else "development_test"
    test_pairs_key = f"{test_split_name}_pairs"
    test_seed_key = f"{test_split_name}_seed_start"
    gpu_name = require_authorized_cuda(args.expected_gpu)
    seed_everything(int(config["training"]["seed"]))
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
    if not isinstance(decoder, nn.Module):
        raise TypeError("Dream decoder is not a torch module")
    if not isinstance(input_embedding, nn.Embedding):
        raise TypeError("Dream input embedding is not torch Embedding")
    if not isinstance(output_head, nn.Module):
        raise TypeError("Dream output head is not a torch module")
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
        raise RuntimeError("sealed LoRA checkpoint hash differs from the frozen config")
    lora_copy = run_dir / "lora_adapter.pt"
    shutil.copyfile(source_lora, lora_copy)
    load_adapter_checkpoint(model, lora_copy)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for _, parameter in lora_named_parameters(model)):
        raise AssertionError("fixed LoRA parameters unexpectedly remain trainable")

    dataset = config["dataset"]
    channel_config = config["channel"]
    training = config["training"]
    answer_values = tuple(str(value) for value in dataset["answer_values"])
    assistant_prefix = str(dataset["assistant_prefix"])
    assistant_prefix_tokens = len(tokenizer.encode(assistant_prefix, add_special_tokens=False))
    if assistant_prefix_tokens <= 0:
        raise RuntimeError("assistant prefix must contain at least one token")
    prefix_table = prefix_teacher_table(
        tokenizer,
        answer_values=answer_values,
        template=str(channel_config["prefix_alignment_template"]),
        message_slots=int(channel_config["message_slots"]),
    )

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
        batch_size=int(training["encoding_batch_size"]),
        device=device,
    )
    input_rms = embedding_rms(input_embedding)
    channel = InputPrefixChannel(
        int(model.config.hidden_size),
        input_embedding_rms=input_rms,
        message_slots=int(channel_config["message_slots"]),
        message_dim=int(channel_config["message_dim"]),
        width=int(channel_config["width"]),
        num_heads=int(channel_config["num_heads"]),
        max_agents=int(channel_config["max_agents"]),
    ).to(device)
    training_log = train_channel(
        channel,
        decoder,
        input_embedding,
        output_head,
        train_tokenized,
        train_hidden,
        prefix_table,
        config,
        assistant_prefix_tokens=assistant_prefix_tokens,
        device=device,
    )
    channel_path = run_dir / "channel.pt"
    torch.save(channel.state_dict(), channel_path)
    channel_sha256 = sha256_file(channel_path)
    del train_hidden, train_tokenized, train_examples
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
        batch_size=int(training["encoding_batch_size"]),
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
        seed=int(config["evaluation"]["seed"]),
        device=device,
    )
    gate = config["evaluation"]["gate"]
    validation_passed = gate_passes(validation_metrics, gate)
    engineering_smoke = config["stage"] == "engineering_smoke"
    open_test = validation_passed and not engineering_smoke
    test_metrics: dict[str, Any] | None = None
    test_predictions: dict[str, torch.Tensor] | None = None
    test_digest: str | None = None
    test_tokenized: TokenizedSplit | None = None
    test_hidden: HiddenSplit | None = None
    if open_test:
        test_examples = build_split(
            pair_count=int(dataset[test_pairs_key]),
            seed_start=int(dataset[test_seed_key]),
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
            batch_size=int(training["encoding_batch_size"]),
            device=device,
        )
        test_metrics, test_predictions = evaluate_split(
            channel,
            decoder,
            input_embedding,
            output_head,
            test_tokenized,
            test_hidden,
            config,
            assistant_prefix_tokens=assistant_prefix_tokens,
            seed=int(config["evaluation"]["seed"]) + 1,
            device=device,
        )
    pilot_passed = bool(test_metrics is not None and gate_passes(test_metrics, gate))
    if engineering_smoke:
        verdict = "engineering_smoke_completed"
    elif not validation_passed:
        verdict = "stop_validation"
    elif confirmation_run and pilot_passed:
        verdict = "confirmation_seed_passed"
    elif confirmation_run:
        verdict = "confirmation_seed_failed"
    elif pilot_passed:
        verdict = "go_multiseed_confirmation"
    else:
        verdict = "stop_development_test"

    model_digest, weight_records = model_weight_manifest(snapshot)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Dream decoder layers are not a ModuleList")
    effective_attention_classes = sorted(
        {type(getattr(layer, "self_attn", None)).__name__ for layer in layers}
    )
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
            "mechanism": "sender_hidden_to_receiver_input_embedding_prefix_with_fixed_lora",
            "prefix_insertion": "before_visible_assistant_prefix",
            "prefix_alignment_loss": training["prefix_alignment_loss"],
            "prefix_alignment_token_ids": {str(key): value for key, value in prefix_table.items()},
            "assistant_prefix_tokens": assistant_prefix_tokens,
            "input_embedding_rms": input_rms,
            "no_message_parity": parity,
            "sender_encoder_does_not_consult_receiver": True,
            "receiver_canvas_replaced": False,
            "pretrained_backbone_frozen": True,
            "task_lora_frozen": True,
            "language_model_head_frozen": True,
            "training_answers_fully_masked_single_token": True,
            f"{test_split_name}_opened_only_after_validation_pass": open_test,
            "gate": gate,
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
            test_split_name: {
                "pairs": int(dataset[test_pairs_key]),
                "seed_start": int(dataset[test_seed_key]),
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
            "requested_attention_implementation": model_config[
                "requested_attention_implementation"
            ],
            "effective_attention_classes": effective_attention_classes,
            "lora_checkpoint_sha256": sha256_file(lora_copy),
            "lora_attachments": [attachment.qualified_name for attachment in attachments],
            "lora_frozen": True,
            "pretrained_parameters_frozen": True,
            "language_model_head_frozen": True,
            "channel_checkpoint_sha256": channel_sha256,
        },
    )
    write_immutable_json(
        run_dir / "seed_manifest.json",
        {
            "training_seed": training["seed"],
            "train_generation_seed_start": dataset["train_seed_start"],
            "validation_generation_seed_start": dataset["validation_seed_start"],
            f"{test_split_name}_generation_seed_start": dataset[test_seed_key],
            "evaluation_seed": config["evaluation"]["seed"],
            "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
        },
    )
    write_immutable_json(
        run_dir / "metrics.json",
        {
            "verdict": verdict,
            "gate": gate,
            "validation_gate_passed": validation_passed,
            (
                "confirmation_seed_gate_passed"
                if confirmation_run
                else "development_pilot_gate_passed"
            ): pilot_passed,
            "validation": validation_metrics,
            test_split_name: test_metrics,
            "canvas_assignment_count": 0,
            "training_steps": training["steps"],
            "non_claim": config["non_claim"],
        },
    )
    write_jsonl(run_dir / "training_log.jsonl", training_log)
    endpoint_name = test_split_name if open_test else "validation"
    endpoint_tokenized = test_tokenized if open_test else validation_tokenized
    endpoint_predictions = test_predictions if open_test else validation_predictions
    if endpoint_tokenized is None or endpoint_predictions is None:
        raise AssertionError("selected endpoint artifacts are missing")
    message_bytes = int(channel_config["message_slots"]) * int(channel_config["message_dim"]) * 2
    write_jsonl(
        run_dir / "predictions.jsonl",
        prediction_records(
            endpoint_name,
            endpoint_tokenized,
            endpoint_predictions,
            tokenizer,
            model_id=str(model_config["id"]),
            training_seed=int(training["seed"]),
            checkpoint_hash=channel_sha256,
            message_bytes=message_bytes,
        ),
    )
    trajectories: list[dict[str, Any]] = []
    for index, example in enumerate(endpoint_tokenized.examples):
        for method in ("matched", "wrong_fact"):
            final_token = endpoint_predictions[method][index : index + 1]
            initial_canvas = endpoint_tokenized.input_ids[index, 0]
            trajectories.append(
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
    write_zstd_jsonl(run_dir / "trajectories.jsonl.zst", trajectories)
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
            f"{test_split_name}_hidden_shape": (
                list(test_hidden.hidden_states.shape) if test_hidden is not None else None
            ),
        },
    )
    _write_text_exclusive(run_dir / "git_commit.txt", args.source_revision + "\n")
    _write_text_exclusive(
        run_dir / "stdout.log",
        json.dumps(
            {
                "validation_gate_passed": validation_passed,
                "development_pilot_gate_passed": pilot_passed,
                "verdict": verdict,
            },
            sort_keys=True,
        )
        + "\n",
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    _write_text_exclusive(run_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    print((run_dir / "stdout.log").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
