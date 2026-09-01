"""Train a QASC dense-prefix channel and audit QASC/HiddenBench communication causally."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dllm
import torch
import yaml  # type: ignore[import-untyped]
from phase5_hm_synth_tiny import model_weight_manifest
from phase6_dream_balanced_oracle_prefix_audit import prefix_table
from phase6_dream_dense_latent_two_round import (
    classify_messages,
    dense_message_diagnostics,
    encode_all,
    projected_prefix_cosine,
    target_prefix_embeddings,
    train_prefix_decoder,
    train_representation,
)
from phase6_dream_input_prefix import decoder_hidden, embedding_rms, insert_prefix_embeddings
from torch import nn

from hidden_messages.communication import DenseLatentPrefixChannel
from hidden_messages.utils.checksums import sha256_file, sha256_tensor
from hidden_messages.utils.manifests import create_run_directory, write_immutable_json
from hidden_messages.utils.reproducibility import require_authorized_cuda, seed_everything


@dataclass(frozen=True, slots=True)
class MethodRow:
    example_id: str
    source_task_id: str
    permutation_index: int
    question: str
    answer_label: str
    valid_label_count: int
    private_contexts: tuple[str, ...]
    source_split: str


@dataclass(frozen=True, slots=True)
class TokenizedCohort:
    rows: tuple[MethodRow, ...]
    token_rows: tuple[tuple[tuple[int, ...], ...], ...]
    label_ids: tuple[int, ...]
    assistant_prefix_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--qasc-adapter-run-dir", type=Path, required=True)
    parser.add_argument("--qasc-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--hiddenbench-adapter-run-dir", type=Path, required=True)
    parser.add_argument("--hiddenbench-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--source-channel-run-dir", type=Path)
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Phase 8 config must be a mapping")
    return payload


def _hash_key(*values: object) -> str:
    text = "|".join(str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()


def _read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _load_prediction_labels(path: Path, *, benchmark: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            example_id = str(payload["example_id"])
            if benchmark == "qasc":
                label = payload["conditions"]["agent0_local"]["constrained_label"]
            elif benchmark == "hiddenbench":
                label = payload["base_conditions"]["receiver_local"]["constrained_label"]
            else:
                raise ValueError(f"unknown benchmark: {benchmark}")
            labels[example_id] = str(label)
    return labels


def load_qasc_rows(path: Path, *, expected_rows: int, expected_split: str) -> list[MethodRow]:
    rows: list[MethodRow] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"QASC line {line_number} lacks a terminal newline")
            payload = json.loads(line)
            contexts = payload.get("private_contexts")
            answer_options = payload.get("answer_options")
            example_id = payload.get("example_id")
            if (
                not isinstance(example_id, str)
                or not example_id
                or example_id in seen
                or payload.get("source_split") != expected_split
                or payload.get("designated_receiver") != 0
                or payload.get("answer_label") not in tuple("ABCDEFGH")
                or not isinstance(payload.get("question"), str)
                or not isinstance(contexts, list)
                or len(contexts) != 2
                or any(not isinstance(value, str) or not value for value in contexts)
                or not isinstance(answer_options, list)
                or len(answer_options) != 8
            ):
                raise ValueError(f"QASC line {line_number} violates the Phase 8 schema")
            seen.add(example_id)
            rows.append(
                MethodRow(
                    example_id=example_id,
                    source_task_id=str(payload["source_example_id"]),
                    permutation_index=0,
                    question=str(payload["question"]),
                    answer_label=str(payload["answer_label"]),
                    valid_label_count=8,
                    private_contexts=tuple(str(value) for value in contexts),
                    source_split=expected_split,
                )
            )
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} QASC rows, observed {len(rows)}")
    return rows


def load_hiddenbench_rows(path: Path, *, expected_rows: int) -> list[MethodRow]:
    rows: list[MethodRow] = []
    seen: set[str] = set()
    task_counts: Counter[str] = Counter()
    task_sizes: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        contexts = payload.get("private_contexts")
        options = payload.get("answer_options")
        metadata = payload.get("metadata")
        example_id = payload.get("example_id")
        source_task_id = payload.get("source_example_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id in seen
            or not isinstance(source_task_id, str)
            or not source_task_id.isdigit()
            or payload.get("source_split") != "official"
            or payload.get("designated_receiver") != 0
            or payload.get("answer_label") not in tuple("ABCD")
            or not isinstance(payload.get("question"), str)
            or not isinstance(contexts, list)
            or len(contexts) not in {3, 4}
            or any(not isinstance(value, str) or not value for value in contexts)
            or not isinstance(options, list)
            or len(options) not in {3, 4}
            or not isinstance(metadata, dict)
            or metadata.get("num_agents") != len(contexts)
            or metadata.get("cluster_task_id") != source_task_id
            or metadata.get("permutation_index") not in range(5)
        ):
            raise ValueError(f"HiddenBench line {line_number} violates the Phase 8 schema")
        seen.add(example_id)
        if source_task_id in task_sizes and task_sizes[source_task_id] != len(contexts):
            raise ValueError("HiddenBench task changed group size")
        task_sizes[source_task_id] = len(contexts)
        task_counts[source_task_id] += 1
        rows.append(
            MethodRow(
                example_id=example_id,
                source_task_id=source_task_id,
                permutation_index=int(metadata["permutation_index"]),
                question=str(payload["question"]),
                answer_label=str(payload["answer_label"]),
                valid_label_count=len(options),
                private_contexts=tuple(str(value) for value in contexts),
                source_split="official",
            )
        )
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} HiddenBench rows, observed {len(rows)}")
    if len(task_counts) != 65 or set(task_counts.values()) != {5}:
        raise ValueError("HiddenBench must contain 65 five-permutation task clusters")
    return rows


def hiddenbench_development_task_ids(
    rows: Sequence[MethodRow],
    *,
    salt: str,
    n4_tasks: int,
    n3_tasks: int,
) -> tuple[str, ...]:
    task_sizes = {row.source_task_id: len(row.private_contexts) for row in rows}
    selected: list[str] = []
    for group_size, count in ((4, n4_tasks), (3, n3_tasks)):
        candidates = sorted(
            (task for task, size in task_sizes.items() if size == group_size),
            key=lambda task: _hash_key(salt, f"N{group_size}", task),
        )
        if len(candidates) < count:
            raise ValueError(f"HiddenBench N={group_size} development stratum is undersized")
        selected.extend(candidates[:count])
    return tuple(sorted(selected, key=int))


def select_rows(rows: Sequence[MethodRow], task_ids: Iterable[str]) -> list[MethodRow]:
    selected = set(task_ids)
    return [row for row in rows if row.source_task_id in selected]


def engineering_smoke_rows(rows: Sequence[MethodRow], *, count: int) -> list[MethodRow]:
    if count < 2 or len(rows) < count:
        raise ValueError("engineering smoke requires at least two rows")
    selected = list(rows[:count])
    if len({row.answer_label for row in selected}) == 1:
        original_label = selected[0].answer_label
        replacement = next(
            (row for row in rows[count:] if row.answer_label != original_label), None
        )
        if replacement is None:
            raise ValueError("engineering smoke cannot find a second target label")
        selected[-1] = replacement
    return selected


def qasc_derangement_order(rows: Sequence[MethodRow], *, salt: str) -> torch.Tensor:
    order: list[int] = []
    for index, row in enumerate(rows):
        candidates = [
            candidate
            for candidate, other in enumerate(rows)
            if candidate != index and other.answer_label != row.answer_label
        ]
        if not candidates:
            raise ValueError("QASC derangement lacks a different-target candidate")
        order.append(min(candidates, key=lambda value: _hash_key(salt, row.example_id, value)))
    return torch.tensor(order, dtype=torch.long)


def hiddenbench_derangement_order(rows: Sequence[MethodRow], *, salt: str) -> torch.Tensor:
    row_lookup = {
        (row.source_task_id, row.permutation_index): index for index, row in enumerate(rows)
    }
    task_sizes = {row.source_task_id: len(row.private_contexts) for row in rows}
    task_mapping: dict[str, str] = {}
    for group_size in sorted(set(task_sizes.values())):
        tasks = sorted(
            (task for task, size in task_sizes.items() if size == group_size),
            key=lambda task: _hash_key(salt, f"N{group_size}", task),
        )
        if len(tasks) < 2:
            raise ValueError("HiddenBench derangement requires two tasks per native group size")
        task_mapping.update(
            {task: tasks[(index + 1) % len(tasks)] for index, task in enumerate(tasks)}
        )
    return torch.tensor(
        [row_lookup[(task_mapping[row.source_task_id], row.permutation_index)] for row in rows],
        dtype=torch.long,
    )


def _stable_label_ids(tokenizer: Any, labels: tuple[str, ...]) -> tuple[int, ...]:
    token_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != label:
            raise RuntimeError(f"answer label {label!r} is not a stable single token")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("answer labels do not map to distinct token IDs")
    return tuple(token_ids)


def tokenize_cohort(
    rows: Sequence[MethodRow],
    *,
    benchmark: str,
    tokenizer: Any,
    config: dict[str, Any],
) -> TokenizedCohort:
    surface = config["surface"]
    if benchmark == "qasc":
        labels = tuple(str(value) for value in surface["qasc_labels"])
        template = str(surface["qasc_prompt_template"])
        max_length = int(surface["max_sequence_length_qasc"])
        evidence_key = "evidence"
    elif benchmark == "hiddenbench":
        labels = tuple(str(value) for value in surface["hiddenbench_labels"])
        template = str(surface["hiddenbench_prompt_template"])
        max_length = int(surface["max_sequence_length_hiddenbench"])
        evidence_key = "private_information"
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("Dream tokenizer has no mask token")
    assistant_prefix = str(surface["assistant_prefix"])
    prefix_ids = tuple(
        int(value) for value in tokenizer.encode(assistant_prefix, add_special_tokens=False)
    )
    if not prefix_ids:
        raise RuntimeError("Dream assistant prefix tokenized to an empty sequence")
    nested: list[tuple[tuple[int, ...], ...]] = []
    for row in rows:
        agents: list[tuple[int, ...]] = []
        for context in row.private_contexts:
            user_text = template.format(question=row.question, **{evidence_key: context})
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            tokens = tuple(int(value) for value in prompt) + prefix_ids + (int(mask_id),)
            if len(tokens) > max_length:
                raise RuntimeError(
                    f"{benchmark} prompt length {len(tokens)} exceeds {max_length} "
                    f"for {row.example_id}"
                )
            agents.append(tokens)
        nested.append(tuple(agents))
    return TokenizedCohort(
        rows=tuple(rows),
        token_rows=tuple(nested),
        label_ids=_stable_label_ids(tokenizer, labels),
        assistant_prefix_tokens=len(prefix_ids),
    )


def _pad_batch(
    token_rows: Sequence[Sequence[int]], tokenizer: Any, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Dream tokenizer has no pad or eos token")
    width = max(len(row) for row in token_rows)
    input_ids = torch.full((len(token_rows), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids, dtype=torch.bool)
    lengths = torch.empty(len(token_rows), dtype=torch.long, device=device)
    for index, row in enumerate(token_rows):
        values = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, : len(row)] = values
        attention[index, : len(row)] = True
        lengths[index] = len(row)
    return input_ids, attention, lengths


@torch.inference_mode()
def forward_selected(
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    token_rows: Sequence[Sequence[int]],
    *,
    output_position_offset: int,
    assistant_prefix_tokens: int,
    prefix_embeddings: torch.Tensor | None,
    label_ids: Sequence[int],
    valid_label_counts: Sequence[int],
    device: torch.device,
    reference_full_model: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention, lengths = _pad_batch(token_rows, tokenizer, device=device)
    output_positions = lengths - 1 + output_position_offset
    if bool((output_positions < 0).any()):
        raise RuntimeError("output offset moved before the prompt")
    if prefix_embeddings is None and reference_full_model:
        captured: list[torch.Tensor] = []

        def capture_hidden(_module: nn.Module, hook_inputs: tuple[Any, ...]) -> None:
            if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
                raise RuntimeError("Dream output head did not receive hidden states")
            captured.append(hook_inputs[0])

        handle = output_head.register_forward_pre_hook(capture_hidden)
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one Dream hidden capture, observed {len(captured)}")
        hidden = captured[0]
        rows = torch.arange(hidden.shape[0], device=device)
        selected_hidden = hidden[rows, output_positions]
        selected_logits = outputs.logits[rows, output_positions]
    else:
        embeddings = input_embedding(input_ids)
        if prefix_embeddings is not None:
            insertion_positions = lengths - 1 - assistant_prefix_tokens
            embeddings, attention, output_positions = insert_prefix_embeddings(
                embeddings,
                attention,
                insertion_positions,
                output_positions,
                prefix_embeddings.to(device),
            )
        hidden = decoder_hidden(decoder, embeddings, attention)
        rows = torch.arange(hidden.shape[0], device=device)
        selected_hidden = hidden[rows, output_positions]
        selected_logits = output_head(selected_hidden)
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    label_logits = selected_logits.float().index_select(-1, label_tensor)
    label_range = torch.arange(len(label_ids), device=device)
    counts = torch.tensor(valid_label_counts, dtype=torch.long, device=device).unsqueeze(1)
    label_logits = label_logits.masked_fill(label_range.unsqueeze(0) >= counts, -torch.inf)
    predictions = label_logits.argmax(dim=-1)
    return selected_hidden.to(device="cpu", dtype=torch.bfloat16), predictions.cpu()


@torch.inference_mode()
def verify_decoder_parity(
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    cohort: TokenizedCohort,
    *,
    output_position_offset: int,
    device: torch.device,
) -> dict[str, Any]:
    rows = [cohort.token_rows[index][0] for index in range(min(2, len(cohort.rows)))]
    counts = [cohort.rows[index].valid_label_count for index in range(len(rows))]
    reference_hidden, reference_predictions = forward_selected(
        model,
        decoder,
        input_embedding,
        output_head,
        tokenizer,
        rows,
        output_position_offset=output_position_offset,
        assistant_prefix_tokens=cohort.assistant_prefix_tokens,
        prefix_embeddings=None,
        label_ids=cohort.label_ids,
        valid_label_counts=counts,
        device=device,
        reference_full_model=True,
    )
    input_ids, attention, lengths = _pad_batch(rows, tokenizer, device=device)
    positions = lengths - 1 + output_position_offset
    hidden = decoder_hidden(decoder, input_embedding(input_ids), attention)
    selected = hidden[torch.arange(len(rows), device=device), positions]
    label_tensor = torch.tensor(cohort.label_ids, dtype=torch.long, device=device)
    candidate_logits = output_head(selected).float().index_select(-1, label_tensor)
    label_range = torch.arange(len(cohort.label_ids), device=device)
    candidate_logits = candidate_logits.masked_fill(
        label_range.unsqueeze(0) >= torch.tensor(counts, device=device).unsqueeze(1), -torch.inf
    )
    candidate_predictions = candidate_logits.argmax(dim=-1).cpu()
    hidden_difference = (reference_hidden.float() - selected.float().cpu()).abs()
    exact_hidden = torch.equal(reference_hidden, selected.to(device="cpu", dtype=torch.bfloat16))
    exact_predictions = torch.equal(reference_predictions, candidate_predictions)
    if not exact_hidden or not exact_predictions:
        raise AssertionError("Dream no-prefix decoder path failed exact parity")
    return {
        "hidden_bfloat16_exact": exact_hidden,
        "constrained_prediction_exact": exact_predictions,
        "max_hidden_abs_difference": float(hidden_difference.max().item()),
    }


def flatten_tokens(cohort: TokenizedCohort) -> list[tuple[int, ...]]:
    return [tokens for agents in cohort.token_rows for tokens in agents]


@torch.inference_mode()
def collect_states(
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    cohort: TokenizedCohort,
    *,
    batch_examples: int,
    output_position_offset: int,
    device: torch.device,
) -> torch.Tensor:
    agent_count = len(cohort.token_rows[0])
    if any(len(agents) != agent_count for agents in cohort.token_rows):
        raise ValueError("state collection cohort must have a fixed agent count")
    flat = flatten_tokens(cohort)
    valid = [row.valid_label_count for row in cohort.rows for _ in range(agent_count)]
    batch_size = batch_examples * agent_count
    outputs: list[torch.Tensor] = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        hidden, _ = forward_selected(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            flat[start:stop],
            output_position_offset=output_position_offset,
            assistant_prefix_tokens=cohort.assistant_prefix_tokens,
            prefix_embeddings=None,
            label_ids=cohort.label_ids,
            valid_label_counts=valid[start:stop],
            device=device,
        )
        outputs.append(hidden)
    return torch.cat(outputs)


@torch.inference_mode()
def _project_all(
    channel: DenseLatentPrefixChannel,
    messages: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    prefixes: list[torch.Tensor] = []
    for start in range(0, messages.shape[0], batch_size):
        stop = min(start + batch_size, messages.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            value = channel.project(messages[start:stop].to(device))
        prefixes.append(value.to(device="cpu", dtype=torch.bfloat16))
    return torch.cat(prefixes)


def _moment_matched_random(messages: torch.Tensor, *, seed: int) -> torch.Tensor:
    values = messages.float()
    mean = values.mean(dim=(0, 1), keepdim=True)
    std = values.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(values.shape, generator=generator)
    random = (random - random.mean(dim=(0, 1), keepdim=True)) / random.std(
        dim=(0, 1), unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    return (random * std + mean).to(torch.bfloat16)


def _incoming_messages(messages: torch.Tensor, *, condition: str) -> torch.Tensor:
    row_count, agent_count = messages.shape[:2]
    if agent_count < 2:
        raise ValueError("communication requires at least two agents")
    if condition == "self":
        return messages
    incoming: list[torch.Tensor] = []
    for receiver in range(agent_count):
        senders = [index for index in range(agent_count) if index != receiver]
        incoming.append(messages[:, senders].float().mean(dim=1).to(torch.bfloat16))
    result = torch.stack(incoming, dim=1)
    if result.shape[:2] != (row_count, agent_count):
        raise AssertionError("incoming-message aggregation changed row or receiver count")
    return result


@torch.inference_mode()
def _forward_cohort(
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    cohort: TokenizedCohort,
    *,
    batch_examples: int,
    output_position_offset: int,
    prefixes: torch.Tensor | None,
    receiver_only: bool,
    reference_full_model: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if receiver_only:
        tokens = [agents[0] for agents in cohort.token_rows]
        counts = [row.valid_label_count for row in cohort.rows]
        flat_prefixes = prefixes
        agent_count = 1
    else:
        agent_count = len(cohort.token_rows[0])
        tokens = flatten_tokens(cohort)
        counts = [row.valid_label_count for row in cohort.rows for _ in range(agent_count)]
        flat_prefixes = None if prefixes is None else prefixes.flatten(0, 1)
    batch_size = batch_examples * agent_count
    states: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for start in range(0, len(tokens), batch_size):
        stop = min(start + batch_size, len(tokens))
        hidden, labels = forward_selected(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            tokens[start:stop],
            output_position_offset=output_position_offset,
            assistant_prefix_tokens=cohort.assistant_prefix_tokens,
            prefix_embeddings=None if flat_prefixes is None else flat_prefixes[start:stop],
            label_ids=cohort.label_ids,
            valid_label_counts=counts[start:stop],
            device=device,
            reference_full_model=reference_full_model,
        )
        states.append(hidden)
        predictions.append(labels)
    return torch.cat(states), torch.cat(predictions)


@torch.inference_mode()
def evaluate_condition(
    channel: DenseLatentPrefixChannel,
    model: nn.Module,
    decoder: nn.Module,
    input_embedding: nn.Embedding,
    output_head: nn.Module,
    tokenizer: Any,
    cohort: TokenizedCohort,
    *,
    condition: str,
    derangement_order: torch.Tensor,
    batch_examples: int,
    diagnostic_batch_size: int,
    exchange_rounds: int,
    output_position_offset: int,
    random_seed: int,
    no_message_receiver_batch_examples: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    agent_count = len(cohort.token_rows[0])
    row_count = len(cohort.rows)
    if condition == "final_only":
        states, _ = _forward_cohort(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            cohort,
            batch_examples=batch_examples,
            output_position_offset=output_position_offset,
            prefixes=None,
            receiver_only=False,
            reference_full_model=False,
            device=device,
        )
        messages = encode_all(
            channel, states, batch_size=diagnostic_batch_size, device=device
        ).reshape(row_count, agent_count, channel.message_slots, channel.message_dim)
        incoming = _incoming_messages(messages, condition="matched")[:, 0]
        final_receiver_prefix = _project_all(
            channel, incoming, batch_size=diagnostic_batch_size, device=device
        )
        _, predictions = _forward_cohort(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            cohort,
            batch_examples=batch_examples,
            output_position_offset=output_position_offset,
            prefixes=final_receiver_prefix,
            receiver_only=True,
            reference_full_model=False,
            device=device,
        )
        return predictions, {
            "model_agent_forwards_per_example": agent_count + 1,
            "communication_rounds": 1,
            "transmitted_bytes_per_example": agent_count
            * (agent_count - 1)
            * channel.message_slots
            * channel.message_dim
            * 2,
            "message_sha256": sha256_tensor(messages),
        }

    prefixes: torch.Tensor | None = None
    round_hashes: list[str] = []
    for round_index in range(exchange_rounds):
        states, _ = _forward_cohort(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            cohort,
            batch_examples=batch_examples,
            output_position_offset=output_position_offset,
            prefixes=prefixes,
            receiver_only=False,
            reference_full_model=False,
            device=device,
        )
        messages = encode_all(
            channel, states, batch_size=diagnostic_batch_size, device=device
        ).reshape(row_count, agent_count, channel.message_slots, channel.message_dim)
        if condition == "deranged":
            messages = messages.index_select(0, derangement_order)
        elif condition == "zero":
            messages = torch.zeros_like(messages)
        elif condition == "random":
            messages = _moment_matched_random(messages, seed=random_seed + round_index)
        elif condition not in {"matched", "self", "no_message"}:
            raise ValueError(f"unknown recurrent condition: {condition}")
        if condition == "no_message":
            prefixes = None
        else:
            incoming = _incoming_messages(messages, condition=condition)
            prefixes = _project_all(
                channel,
                incoming.flatten(0, 1),
                batch_size=diagnostic_batch_size,
                device=device,
            ).reshape(row_count, agent_count, channel.message_slots, channel.model_dim)
            round_hashes.append(sha256_tensor(messages))
    recurrent_receiver_prefix = None if prefixes is None else prefixes[:, 0]
    _, predictions = _forward_cohort(
        model,
        decoder,
        input_embedding,
        output_head,
        tokenizer,
        cohort,
        batch_examples=(
            no_message_receiver_batch_examples if condition == "no_message" else batch_examples
        ),
        output_position_offset=output_position_offset,
        prefixes=recurrent_receiver_prefix,
        receiver_only=True,
        reference_full_model=condition == "no_message",
        device=device,
    )
    return predictions, {
        "model_agent_forwards_per_example": exchange_rounds * agent_count + 1,
        "communication_rounds": 0 if condition == "no_message" else exchange_rounds,
        "transmitted_bytes_per_example": 0
        if condition == "no_message"
        else exchange_rounds
        * agent_count
        * (agent_count - 1)
        * channel.message_slots
        * channel.message_dim
        * 2,
        "round_message_sha256": round_hashes,
    }


def paired_bootstrap_interval_cuda(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    differences = (left.float() - right.float()).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    estimates: list[torch.Tensor] = []
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = torch.randint(
            0,
            differences.numel(),
            (count, differences.numel()),
            generator=generator,
            device=device,
        )
        estimates.append(differences[indices].mean(dim=1))
    quantiles = torch.quantile(torch.cat(estimates), torch.tensor([0.025, 0.975], device=device))
    return [float(value) for value in quantiles.cpu().tolist()]


def task_mean_differences(
    left: torch.Tensor, right: torch.Tensor, rows: Sequence[MethodRow]
) -> tuple[list[str], torch.Tensor]:
    if left.shape != right.shape or left.ndim != 1 or left.numel() != len(rows):
        raise ValueError("clustered contrast inputs are not aligned")
    task_ids = sorted({row.source_task_id for row in rows}, key=int)
    values: list[torch.Tensor] = []
    for task_id in task_ids:
        indices = [index for index, row in enumerate(rows) if row.source_task_id == task_id]
        if len(indices) != 5:
            raise ValueError("HiddenBench task cluster does not contain five permutations")
        values.append((left[indices].float() - right[indices].float()).mean())
    return task_ids, torch.stack(values)


def cluster_bootstrap_interval_cuda(
    values: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    values = values.float().to(device)
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


def _accuracy(correct: torch.Tensor, mask: torch.Tensor | None = None) -> float | None:
    selected = correct if mask is None else correct[mask]
    if selected.numel() == 0:
        return None
    return float(selected.float().mean().item())


def qasc_metrics(
    predictions: dict[str, torch.Tensor],
    rows: Sequence[MethodRow],
    strict_ids: set[str],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    targets = torch.tensor([ord(row.answer_label) - ord("A") for row in rows])
    correct = {condition: values == targets for condition, values in predictions.items()}
    strict = torch.tensor([row.example_id in strict_ids for row in rows], dtype=torch.bool)
    evaluation = config["evaluation"]
    replicates = int(evaluation["bootstrap_replicates"])
    seed = int(evaluation["qasc_bootstrap_seed"])
    contrasts: dict[str, Any] = {}
    for offset, reference in enumerate(("no_message", "deranged", "final_only")):
        key = f"matched_minus_{reference}"
        difference = correct["matched"].float() - correct[reference].float()
        contrasts[key] = float(difference.mean().item())
        contrasts[f"{key}_paired_bootstrap_ci95"] = paired_bootstrap_interval_cuda(
            correct["matched"],
            correct[reference],
            replicates=replicates,
            seed=seed + offset,
            device=device,
        )
    return {
        "examples": len(rows),
        "union_required_examples": int(strict.sum().item()),
        "accuracy": {condition: _accuracy(values) for condition, values in correct.items()},
        "union_required_accuracy": {
            condition: _accuracy(values, strict) for condition, values in correct.items()
        },
        **contrasts,
    }


def hiddenbench_metrics(
    predictions: dict[str, torch.Tensor],
    rows: Sequence[MethodRow],
    strict_ids: set[str],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    targets = torch.tensor([ord(row.answer_label) - ord("A") for row in rows])
    correct = {condition: values == targets for condition, values in predictions.items()}
    strict = torch.tensor([row.example_id in strict_ids for row in rows], dtype=torch.bool)
    n4 = torch.tensor([len(row.private_contexts) == 4 for row in rows], dtype=torch.bool)
    n3 = ~n4
    evaluation = config["evaluation"]
    replicates = int(evaluation["bootstrap_replicates"])
    seed = int(evaluation["hiddenbench_bootstrap_seed"])
    contrasts: dict[str, Any] = {}
    for offset, reference in enumerate(("no_message", "deranged", "final_only")):
        key = f"matched_minus_{reference}"
        _, task_values = task_mean_differences(correct["matched"], correct[reference], rows)
        contrasts[key] = float(task_values.mean().item())
        contrasts[f"{key}_task_cluster_bootstrap_ci95"] = cluster_bootstrap_interval_cuda(
            task_values,
            replicates=replicates,
            seed=seed + offset,
            device=device,
        )
    n4_rows = [row for row, include in zip(rows, n4.tolist(), strict=True) if include]
    _, n4_values = task_mean_differences(correct["matched"][n4], correct["no_message"][n4], n4_rows)
    return {
        "examples": len(rows),
        "source_tasks": len({row.source_task_id for row in rows}),
        "context_required_examples": int(strict.sum().item()),
        "context_required_source_tasks": len(
            {
                row.source_task_id
                for row, include in zip(rows, strict.tolist(), strict=True)
                if include
            }
        ),
        "accuracy": {condition: _accuracy(values) for condition, values in correct.items()},
        "context_required_accuracy": {
            condition: _accuracy(values, strict) for condition, values in correct.items()
        },
        "n4_accuracy": {condition: _accuracy(values, n4) for condition, values in correct.items()},
        "n3_accuracy": {condition: _accuracy(values, n3) for condition, values in correct.items()},
        "n4_matched_minus_no_message": float(n4_values.mean().item()),
        "n4_matched_minus_no_message_task_cluster_bootstrap_ci95": cluster_bootstrap_interval_cuda(
            n4_values,
            replicates=replicates,
            seed=seed + 10,
            device=device,
        ),
        **contrasts,
    }


def development_gate(
    metrics: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, bool], bool]:
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
    checks = {
        **{f"qasc_{key}": value for key, value in qasc_checks.items()},
        **{f"hiddenbench_{key}": value for key, value in hiddenbench_checks.items()},
        "qasc_pass": all(qasc_checks.values()),
        "hiddenbench_pass": all(hiddenbench_checks.values()),
        "exact_no_message_parity": bool(metrics["integrity"]["exact_no_message_parity"]),
        "independent_agent_canvases": metrics["integrity"]["canvas_assignment_count"] == 0,
        "designated_receiver_no_vote": bool(metrics["integrity"]["designated_receiver_no_vote"]),
    }
    return checks, all(checks.values())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _validate_input_inventory(path: Path, expected_sha256: str, label: str) -> None:
    if sha256_file(path / "SHA256SUMS") != expected_sha256:
        raise RuntimeError(f"{label} inventory differs from the frozen Phase 8 config")


def validate_run_inventory(run_dir: Path) -> str:
    """Verify a flat immutable run inventory and return its own SHA-256."""
    inventory_path = run_dir / "SHA256SUMS"
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        inventory_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if "  " not in line:
            raise ValueError(f"invalid inventory line {line_number}")
        digest, name = line.split("  ", maxsplit=1)
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError(f"invalid inventory digest on line {line_number}")
        if Path(name).name != name or name == "SHA256SUMS":
            raise ValueError(f"invalid inventory path on line {line_number}")
        if name in entries:
            raise ValueError(f"duplicate inventory path: {name}")
        entries[name] = digest
    actual_names = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_names != set(entries):
        raise RuntimeError("run artifact set differs from SHA256SUMS")
    for name, expected in entries.items():
        if sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"run artifact checksum mismatch: {name}")
    return sha256_file(inventory_path)


def compute_training_gate_checks(
    training_metrics: dict[str, Any], training_config: dict[str, Any]
) -> dict[str, bool]:
    gate = training_config["gate"]
    return {
        "train_classifier_accuracy": training_metrics["train_classifier_accuracy"]
        >= float(gate["train_classifier_accuracy_min"]),
        "development_classifier_accuracy": training_metrics["development_classifier_accuracy"]
        >= float(gate["development_classifier_accuracy_min"]),
        "prefix_cosine": training_metrics["prefix_cosine"] >= float(gate["prefix_cosine_min"]),
        "message_feature_std": training_metrics["message"]["feature_std_mean"]
        >= float(gate["message_feature_std_min"]),
    }


def load_post_hoc_channel(
    channel: DenseLatentPrefixChannel,
    *,
    source_run_dir: Path,
    destination_run_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    post_hoc = config["post_hoc"]
    inventory_sha256 = validate_run_inventory(source_run_dir)
    if inventory_sha256 != str(post_hoc["source_inventory_sha256"]):
        raise RuntimeError("source training-run inventory differs from post-hoc config")

    source_run = json.loads((source_run_dir / "run.json").read_text(encoding="utf-8"))
    source_metrics = json.loads((source_run_dir / "metrics.json").read_text(encoding="utf-8"))
    source_model = json.loads((source_run_dir / "model_manifest.json").read_text(encoding="utf-8"))
    source_revision = (source_run_dir / "git_commit.txt").read_text(encoding="utf-8").strip()
    if source_run.get("run_id") != str(post_hoc["source_training_run_id"]):
        raise RuntimeError("unexpected source training run ID")
    if source_run.get("verdict") != "STOP_PHASE8_TRAINING_GATE":
        raise RuntimeError("source run is not the registered Phase 8 training-gate STOP")
    if source_revision != str(post_hoc["source_revision"]):
        raise RuntimeError("source training revision differs from post-hoc config")
    if source_metrics.get("training_gate_passed") is not False:
        raise RuntimeError("source training gate was not preserved as failed")
    if "qasc" in source_metrics or "hiddenbench" in source_metrics:
        raise RuntimeError("source run unexpectedly contains opened scientific endpoints")

    training_metrics = source_metrics["training"]
    training_checks = compute_training_gate_checks(training_metrics, config["training"])
    if training_checks != source_metrics.get("training_gate_checks"):
        raise RuntimeError("source training checks do not reproduce under the unchanged thresholds")
    if all(training_checks.values()):
        raise RuntimeError("post-hoc waiver is unnecessary because the source training gate passed")

    source_channel = source_run_dir / "qasc_dense_latent_prefix_channel.pt"
    channel_sha256 = sha256_file(source_channel)
    if channel_sha256 != str(post_hoc["source_channel_sha256"]):
        raise RuntimeError("source channel differs from post-hoc config")
    if channel_sha256 != source_model.get("channel_sha256"):
        raise RuntimeError("source channel differs from its model manifest")
    state_dict = torch.load(source_channel, map_location=device, weights_only=True)
    channel.load_state_dict(state_dict, strict=True)
    shutil.copyfile(source_channel, destination_run_dir / source_channel.name)
    for log_name in ("representation_training_log.jsonl", "prefix_training_log.jsonl"):
        shutil.copyfile(source_run_dir / log_name, destination_run_dir / log_name)

    provenance = {
        "authorization": "post_hoc_endpoint_opening_after_observed_training_gate_failure",
        "evidence_tier": "post_hoc_development_exploratory",
        "source_training_run_id": source_run["run_id"],
        "source_training_verdict": source_run["verdict"],
        "source_training_gate_passed": False,
        "source_inventory_sha256": inventory_sha256,
        "source_revision": source_revision,
        "source_channel_sha256": channel_sha256,
        "checkpoint_retrained": False,
        "training_gate_threshold_changed": False,
    }
    return training_metrics, training_checks, provenance


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    gpu_name = require_authorized_cuda(args.expected_gpu)
    observed_gpu_uuid = os.popen("nvidia-smi --query-gpu=uuid --format=csv,noheader").read().strip()
    if observed_gpu_uuid != args.gpu_uuid:
        raise RuntimeError("runtime GPU UUID differs from the pod entry receipt")
    training_seed = int(config["training"]["seed"])
    seed_everything(training_seed)
    device = torch.device("cuda")
    run_dir = create_run_directory(args.run_root, args.run_id)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    shutil.copyfile(args.source_manifest, run_dir / "source-manifest.sha256")

    qasc_config = config["qasc"]
    hb_config = config["hiddenbench"]
    _validate_input_inventory(
        args.qasc_adapter_run_dir, str(qasc_config["adapter_inventory_sha256"]), "QASC adapter"
    )
    _validate_input_inventory(
        args.qasc_baseline_run_dir, str(qasc_config["baseline_inventory_sha256"]), "QASC baseline"
    )
    _validate_input_inventory(
        args.hiddenbench_adapter_run_dir,
        str(hb_config["adapter_inventory_sha256"]),
        "HiddenBench adapter",
    )
    _validate_input_inventory(
        args.hiddenbench_baseline_run_dir,
        str(hb_config["baseline_inventory_sha256"]),
        "HiddenBench baseline",
    )
    qasc_train_path = args.qasc_adapter_run_dir / str(qasc_config["train_file"])
    qasc_dev_path = args.qasc_adapter_run_dir / str(qasc_config["development_file"])
    qasc_test_path = args.qasc_adapter_run_dir / str(qasc_config["sealed_test_file"])
    hb_path = args.hiddenbench_adapter_run_dir / str(hb_config["file"])
    for path, expected in (
        (qasc_train_path, qasc_config["train_sha256"]),
        (qasc_dev_path, qasc_config["development_sha256"]),
        (qasc_test_path, qasc_config["sealed_test_sha256"]),
        (hb_path, hb_config["file_sha256"]),
    ):
        if sha256_file(path) != str(expected):
            raise RuntimeError(f"dataset checksum mismatch: {path.name}")
    qasc_train = load_qasc_rows(
        qasc_train_path, expected_rows=int(qasc_config["train_rows"]), expected_split="train"
    )
    qasc_dev = load_qasc_rows(
        qasc_dev_path,
        expected_rows=int(qasc_config["development_rows"]),
        expected_split="validation",
    )
    hiddenbench_all = load_hiddenbench_rows(hb_path, expected_rows=int(hb_config["rows"]))
    hb_task_ids = hiddenbench_development_task_ids(
        hiddenbench_all,
        salt=str(hb_config["split_salt"]),
        n4_tasks=int(hb_config["development_n4_tasks"]),
        n3_tasks=int(hb_config["development_n3_tasks"]),
    )
    hiddenbench_dev = select_rows(hiddenbench_all, hb_task_ids)
    if len(hiddenbench_dev) != 5 * (
        int(hb_config["development_n4_tasks"]) + int(hb_config["development_n3_tasks"])
    ):
        raise AssertionError("HiddenBench development split lost permutations")

    stage = str(config["stage"])
    limits = config.get("engineering_limits", {}) if stage == "engineering_smoke" else {}
    if limits:
        qasc_train = qasc_train[: int(limits["qasc_train_rows"])]
        qasc_dev = engineering_smoke_rows(qasc_dev, count=int(limits["qasc_development_rows"]))
        smoke_group_size = int(limits["hiddenbench_group_size"])
        smoke_candidates = [
            task_id
            for task_id in hb_task_ids
            if any(
                row.source_task_id == task_id and len(row.private_contexts) == smoke_group_size
                for row in hiddenbench_dev
            )
        ]
        smoke_tasks = tuple(smoke_candidates[: int(limits["hiddenbench_tasks"])])
        if len(smoke_tasks) != int(limits["hiddenbench_tasks"]):
            raise RuntimeError("engineering HiddenBench stratum is undersized")
        hiddenbench_dev = select_rows(hiddenbench_dev, smoke_tasks)
        if len({len(row.private_contexts) for row in hiddenbench_dev}) != 1:
            raise RuntimeError("engineering HiddenBench slice must have one native group size")

    evaluation = config["evaluation"]
    qasc_order = qasc_derangement_order(qasc_dev, salt=str(evaluation["derangement_salt"]))
    hb_orders_by_size = {
        group_size: hiddenbench_derangement_order(
            [row for row in hiddenbench_dev if len(row.private_contexts) == group_size],
            salt=str(evaluation["derangement_salt"]),
        )
        for group_size in sorted({len(row.private_contexts) for row in hiddenbench_dev})
    }

    model_config = config["model"]
    snapshot = (
        args.model_cache
        / ("models--" + str(model_config["id"]).replace("/", "--"))
        / "snapshots"
        / str(model_config["revision"])
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned Dream snapshot is absent: {snapshot}")
    model_digest, weight_records = model_weight_manifest(snapshot)
    if model_digest != str(model_config["weight_manifest_sha256"]):
        raise RuntimeError("Dream weight manifest differs from the frozen Phase 8 config")
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

    qasc_train_tokens = tokenize_cohort(
        qasc_train, benchmark="qasc", tokenizer=tokenizer, config=config
    )
    qasc_dev_tokens = tokenize_cohort(
        qasc_dev, benchmark="qasc", tokenizer=tokenizer, config=config
    )
    hb_groups = {
        group_size: tokenize_cohort(
            [row for row in hiddenbench_dev if len(row.private_contexts) == group_size],
            benchmark="hiddenbench",
            tokenizer=tokenizer,
            config=config,
        )
        for group_size in sorted({len(row.private_contexts) for row in hiddenbench_dev})
    }
    output_offset = int(model_config["output_position_offset"])
    parity = {
        "qasc": verify_decoder_parity(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            qasc_dev_tokens,
            output_position_offset=output_offset,
            device=device,
        ),
        "hiddenbench": verify_decoder_parity(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            next(iter(hb_groups.values())),
            output_position_offset=output_offset,
            device=device,
        ),
    }

    training_config = config["training"]
    surface = config["surface"]
    qasc_labels = tuple(str(value) for value in surface["qasc_labels"])
    teacher_table = prefix_table(
        tokenizer,
        labels=qasc_labels,
        template=str(surface["prefix_alignment_template"]),
    )
    prototypes = target_prefix_embeddings(
        teacher_table, qasc_labels, input_embedding, device=device
    )
    channel_config = config["channel"]
    if prototypes.shape[1] != int(surface["message_slots"]):
        raise RuntimeError("teacher-prefix token count differs from frozen message slots")
    seed_everything(training_seed)
    channel = DenseLatentPrefixChannel(
        int(model.config.hidden_size),
        len(qasc_labels),
        input_embedding_rms=embedding_rms(input_embedding),
        message_slots=int(surface["message_slots"]),
        message_dim=int(channel_config["message_dim"]),
        width=int(channel_config["width"]),
        classifier_width=int(channel_config["classifier_width"]),
    ).to(device)
    channel_path = run_dir / "qasc_dense_latent_prefix_channel.pt"
    post_hoc_provenance: dict[str, Any] | None = None
    if stage == "post_hoc_endpoint_evaluation":
        if args.source_channel_run_dir is None:
            raise RuntimeError("post-hoc endpoint evaluation requires --source-channel-run-dir")
        training_metrics, training_checks, post_hoc_provenance = load_post_hoc_channel(
            channel,
            source_run_dir=args.source_channel_run_dir,
            destination_run_dir=run_dir,
            config=config,
            device=device,
        )
        training_passed = False
    else:
        if args.source_channel_run_dir is not None:
            raise RuntimeError("--source-channel-run-dir is only valid for post-hoc evaluation")
        train_states = collect_states(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            qasc_train_tokens,
            batch_examples=int(training_config["state_collection_batch_examples"]),
            output_position_offset=output_offset,
            device=device,
        )
        dev_states = collect_states(
            model,
            decoder,
            input_embedding,
            output_head,
            tokenizer,
            qasc_dev_tokens,
            batch_examples=int(training_config["state_collection_batch_examples"]),
            output_position_offset=output_offset,
            device=device,
        )
        train_targets = torch.tensor(
            [ord(row.answer_label) - ord("A") for row in qasc_train for _ in row.private_contexts]
        )
        dev_targets = torch.tensor(
            [ord(row.answer_label) - ord("A") for row in qasc_dev for _ in row.private_contexts]
        )
        representation_steps = int(training_config["representation_steps"])
        prefix_steps = int(training_config["prefix_steps"])
        representation_log = train_representation(
            channel,
            train_states,
            train_targets,
            config,
            seed=training_seed,
            steps=representation_steps,
            device=device,
        )
        prefix_log = train_prefix_decoder(
            channel,
            train_states,
            train_targets,
            prototypes,
            config,
            seed=int(training_config["prefix_seed"]),
            steps=prefix_steps,
            device=device,
        )
        diagnostic_batch_size = int(training_config["diagnostic_batch_size"])
        train_messages = encode_all(
            channel, train_states, batch_size=diagnostic_batch_size, device=device
        )
        dev_messages = encode_all(
            channel, dev_states, batch_size=diagnostic_batch_size, device=device
        )
        training_metrics = {
            "train_examples": len(qasc_train),
            "development_examples": len(qasc_dev),
            "train_state_sha256": sha256_tensor(train_states),
            "development_state_sha256": sha256_tensor(dev_states),
            "train_classifier_accuracy": float(
                (
                    classify_messages(
                        channel, train_messages, batch_size=diagnostic_batch_size, device=device
                    )
                    == train_targets
                )
                .float()
                .mean()
                .item()
            ),
            "development_classifier_accuracy": float(
                (
                    classify_messages(
                        channel, dev_messages, batch_size=diagnostic_batch_size, device=device
                    )
                    == dev_targets
                )
                .float()
                .mean()
                .item()
            ),
            "prefix_cosine": projected_prefix_cosine(
                channel,
                train_messages,
                train_targets,
                prototypes,
                batch_size=diagnostic_batch_size,
                device=device,
            ),
            "message": dense_message_diagnostics(
                train_messages, train_targets, num_classes=len(qasc_labels)
            ),
        }
        training_checks = compute_training_gate_checks(training_metrics, training_config)
        training_passed = all(training_checks.values())
        torch.save(channel.state_dict(), channel_path)
        write_jsonl(run_dir / "representation_training_log.jsonl", representation_log)
        write_jsonl(run_dir / "prefix_training_log.jsonl", prefix_log)
        del train_states, dev_states, train_messages, dev_messages
        gc.collect()
        torch.cuda.empty_cache()

    diagnostic_batch_size = int(training_config["diagnostic_batch_size"])

    conditions = tuple(str(value) for value in evaluation["condition_order"])
    all_predictions: dict[str, dict[str, torch.Tensor]] = {"qasc": {}, "hiddenbench": {}}
    profiles: dict[str, dict[str, Any]] = {"qasc": {}, "hiddenbench": {}}
    should_evaluate = (
        stage in {"engineering_smoke", "post_hoc_endpoint_evaluation"} or training_passed
    )
    no_message_parity = False
    if should_evaluate:
        qasc_baseline_labels = _load_prediction_labels(
            args.qasc_baseline_run_dir / "predictions.jsonl", benchmark="qasc"
        )
        hb_baseline_labels = _load_prediction_labels(
            args.hiddenbench_baseline_run_dir / "predictions.jsonl", benchmark="hiddenbench"
        )
        qasc_baseline_config = load_config(args.qasc_baseline_run_dir / "config.yaml")
        hb_baseline_config = load_config(args.hiddenbench_baseline_run_dir / "config.yaml")
        qasc_no_message_batch = int(qasc_baseline_config["evaluation"]["batch_size"])
        hb_no_message_batch = int(hb_baseline_config["evaluation"]["batch_size"])
        if qasc_no_message_batch <= 0 or hb_no_message_batch <= 0:
            raise RuntimeError("frozen baseline batch size must be positive")
        for condition_index, condition in enumerate(conditions):
            torch.cuda.synchronize()
            qasc_started = time.perf_counter()
            qasc_prediction, qasc_profile = evaluate_condition(
                channel,
                model,
                decoder,
                input_embedding,
                output_head,
                tokenizer,
                qasc_dev_tokens,
                condition=condition,
                derangement_order=qasc_order,
                batch_examples=int(evaluation["qasc_batch_examples"]),
                diagnostic_batch_size=diagnostic_batch_size,
                exchange_rounds=int(channel_config["exchange_rounds"]),
                output_position_offset=output_offset,
                random_seed=int(evaluation["random_message_seed"]) + 100 * condition_index,
                no_message_receiver_batch_examples=qasc_no_message_batch,
                device=device,
            )
            torch.cuda.synchronize()
            qasc_profile["wall_seconds"] = time.perf_counter() - qasc_started
            all_predictions["qasc"][condition] = qasc_prediction
            profiles["qasc"][condition] = qasc_profile
            hb_by_id: dict[str, int] = {}
            hb_profile_groups: dict[str, Any] = {}
            for group_size, cohort in hb_groups.items():
                torch.cuda.synchronize()
                hb_started = time.perf_counter()
                prediction, profile = evaluate_condition(
                    channel,
                    model,
                    decoder,
                    input_embedding,
                    output_head,
                    tokenizer,
                    cohort,
                    condition=condition,
                    derangement_order=hb_orders_by_size[group_size],
                    batch_examples=int(evaluation["hiddenbench_batch_examples"]),
                    diagnostic_batch_size=diagnostic_batch_size,
                    exchange_rounds=int(channel_config["exchange_rounds"]),
                    output_position_offset=output_offset,
                    random_seed=int(evaluation["random_message_seed"])
                    + 1000 * group_size
                    + 100 * condition_index,
                    no_message_receiver_batch_examples=hb_no_message_batch,
                    device=device,
                )
                torch.cuda.synchronize()
                profile["wall_seconds"] = time.perf_counter() - hb_started
                hb_by_id.update(
                    {
                        row.example_id: int(value)
                        for row, value in zip(cohort.rows, prediction.tolist(), strict=True)
                    }
                )
                hb_profile_groups[f"N{group_size}"] = profile
            all_predictions["hiddenbench"][condition] = torch.tensor(
                [hb_by_id[row.example_id] for row in hiddenbench_dev], dtype=torch.long
            )
            profiles["hiddenbench"][condition] = hb_profile_groups
            if condition == "no_message":
                qasc_no_message = [qasc_labels[int(value)] for value in qasc_prediction]
                hb_labels = tuple(str(value) for value in surface["hiddenbench_labels"])
                hb_no_message = [
                    hb_labels[int(value)] for value in all_predictions["hiddenbench"]["no_message"]
                ]
                qasc_mismatches = [
                    row.example_id
                    for row, predicted in zip(qasc_dev, qasc_no_message, strict=True)
                    if predicted != qasc_baseline_labels[row.example_id]
                ]
                hb_mismatches = [
                    row.example_id
                    for row, predicted in zip(hiddenbench_dev, hb_no_message, strict=True)
                    if predicted != hb_baseline_labels[row.example_id]
                ]
                if qasc_mismatches or hb_mismatches:
                    raise AssertionError(
                        "compute-matched no-message predictions differ from frozen baselines: "
                        f"qasc={len(qasc_mismatches)}, hiddenbench={len(hb_mismatches)}, "
                        f"first_qasc={qasc_mismatches[:1]}, first_hiddenbench={hb_mismatches[:1]}"
                    )
                no_message_parity = True

    metrics: dict[str, Any] = {
        "training": training_metrics,
        "training_gate_checks": training_checks,
        "training_gate_passed": training_passed,
        "post_hoc_endpoint_opening": post_hoc_provenance,
        "integrity": {
            "decoder_no_prefix_parity": parity,
            "exact_no_message_parity": no_message_parity,
            "canvas_assignment_count": 0,
            "designated_receiver_no_vote": True,
            "independent_agent_token_rows": True,
            "sealed_qasc_test_predictions_opened": False,
        },
    }
    gate_checks: dict[str, bool] = {}
    development_passed = False
    if stage == "engineering_smoke":
        metrics["engineering_prediction_hashes"] = {
            benchmark: {
                condition: sha256_tensor(values)
                for condition, values in benchmark_predictions.items()
            }
            for benchmark, benchmark_predictions in all_predictions.items()
        }
        verdict = "PASS_PHASE8_ENGINEERING_SMOKE"
    elif not training_passed and stage != "post_hoc_endpoint_evaluation":
        verdict = "STOP_PHASE8_TRAINING_GATE"
    else:
        qasc_strict_ids = _read_ids(
            args.qasc_baseline_run_dir / str(qasc_config["union_required_ids_file"])
        )
        hb_strict_ids = _read_ids(
            args.hiddenbench_baseline_run_dir / str(hb_config["context_required_ids_file"])
        )
        metrics["qasc"] = qasc_metrics(
            all_predictions["qasc"], qasc_dev, qasc_strict_ids, config, device=device
        )
        metrics["hiddenbench"] = hiddenbench_metrics(
            all_predictions["hiddenbench"], hiddenbench_dev, hb_strict_ids, config, device=device
        )
        gate_checks, development_passed = development_gate(metrics, config)
        if stage == "post_hoc_endpoint_evaluation":
            verdict = (
                "EXPLORATORY_PASS_PHASE8_QASC_HIDDENBENCH_ENDPOINTS"
                if development_passed
                else "EXPLORATORY_FAIL_PHASE8_QASC_HIDDENBENCH_ENDPOINTS"
            )
        else:
            verdict = (
                "GO_PHASE8_QASC_HIDDENBENCH_THREE_SEED_CONFIRMATION"
                if development_passed
                else "STOP_PHASE8_QASC_HIDDENBENCH_DENSE_PREFIX_DEVELOPMENT"
            )
        prediction_records: list[dict[str, Any]] = []
        for benchmark, rows in (("qasc", qasc_dev), ("hiddenbench", hiddenbench_dev)):
            labels = qasc_labels if benchmark == "qasc" else tuple(surface["hiddenbench_labels"])
            registered_ids = qasc_strict_ids if benchmark == "qasc" else hb_strict_ids
            registered_name = "union_required" if benchmark == "qasc" else "context_required"
            for index, row in enumerate(rows):
                prediction_records.append(
                    {
                        "benchmark": benchmark,
                        "example_id": row.example_id,
                        "source_task_id": row.source_task_id,
                        "permutation_index": row.permutation_index,
                        "group_size": len(row.private_contexts),
                        "answer_label": row.answer_label,
                        "designated_receiver": 0,
                        registered_name: row.example_id in registered_ids,
                        "conditions": {
                            condition: {
                                "label": str(labels[int(values[index])]),
                                "correct": str(labels[int(values[index])]) == row.answer_label,
                            }
                            for condition, values in all_predictions[benchmark].items()
                        },
                    }
                )
        write_jsonl(run_dir / "predictions.jsonl", prediction_records)

    metrics.update(
        {
            "verdict": verdict,
            "development_gate_checks": gate_checks,
            "development_gate_passed": development_passed,
            "non_claim": config["non_claim"],
        }
    )
    write_immutable_json(run_dir / "metrics.json", metrics)
    write_immutable_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "protocol_id": config["protocol_id"],
            "stage": stage,
            "evidence_tier": (
                "post_hoc_development_exploratory"
                if stage == "post_hoc_endpoint_evaluation"
                else stage
            ),
            "source_revision": args.source_revision,
            "status": "completed",
            "verdict": verdict,
        },
    )
    write_immutable_json(
        run_dir / "dataset_manifest.json",
        {
            "qasc_train": {
                "rows": len(qasc_train),
                "sha256": sha256_file(qasc_train_path),
                "gradient_updates": True,
            },
            "qasc_development": {
                "rows": len(qasc_dev),
                "sha256": sha256_file(qasc_dev_path),
                "gradient_updates": False,
            },
            "qasc_sealed_test": {
                "rows": int(qasc_config["sealed_test_rows"]),
                "sha256": sha256_file(qasc_test_path),
                "predictions_opened": False,
            },
            "hiddenbench": {
                "all_rows": int(hb_config["rows"]),
                "development_rows": len(hiddenbench_dev),
                "development_task_ids": list(hb_task_ids) if not limits else list(smoke_tasks),
                "sha256": sha256_file(hb_path),
                "gradient_updates": False,
                "split_salt": hb_config["split_salt"],
            },
            "full_benchmark_rows_retained": True,
            "proposed_predictions_define_no_subsets": True,
        },
    )
    model_manifest = {
        "model_id": model_config["id"],
        "revision": model_config["revision"],
        "weight_manifest_sha256": model_digest,
        "weights": weight_records,
        "channel_sha256": sha256_file(channel_path),
        "backbone_frozen": True,
        "output_head_frozen": True,
        "channel_training_source": "qasc_train_only",
        "channel_retrained_in_this_run": stage != "post_hoc_endpoint_evaluation",
    }
    if post_hoc_provenance is not None:
        model_manifest["source_training_run_id"] = post_hoc_provenance["source_training_run_id"]
        model_manifest["source_channel_sha256"] = post_hoc_provenance["source_channel_sha256"]
    write_immutable_json(run_dir / "model_manifest.json", model_manifest)
    write_immutable_json(
        run_dir / "design.json",
        {
            "transport": "signed_dense_bfloat16_slots",
            "message_shape": [channel.message_slots, channel.message_dim],
            "exchange_rounds": channel_config["exchange_rounds"],
            "incoming_aggregation": channel_config["incoming_aggregation"],
            "persistent_prefix": channel_config["persistent_prefix"],
            "primary_output": evaluation["primary_output"],
            "vote_used": False,
            "receiver_canvas_replaced": False,
            "agent_contexts_shared": False,
            "auxiliary_gold_labels_used_for_qasc_training": True,
            "hard_label_or_token_payload": False,
            "post_hoc_training_gate_waiver": stage == "post_hoc_endpoint_evaluation",
            "training_gate_threshold_changed": False,
            "non_claim": config["non_claim"],
        },
    )
    write_immutable_json(
        run_dir / "profile.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "condition_profiles": profiles,
            "message_bytes_per_sender_per_round": channel.message_slots * channel.message_dim * 2,
        },
    )
    write_immutable_json(
        run_dir / "environment.json",
        {
            "pod": os.environ.get("HOSTNAME"),
            "gpu": gpu_name,
            "gpu_uuid": observed_gpu_uuid,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    write_text_exclusive(run_dir / "git_commit.txt", args.source_revision + "\n")
    stdout = json.dumps(
        {
            "training_gate_passed": training_passed,
            "post_hoc_endpoint_opening": post_hoc_provenance is not None,
            "development_gate_passed": development_passed,
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
