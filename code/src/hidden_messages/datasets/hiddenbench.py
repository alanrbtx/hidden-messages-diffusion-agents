"""Deterministic official HiddenBench adapter with fixed task-clustered permutations."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence

from hidden_messages.datasets.distributed_qa import DistributedQAExample

HIDDENBENCH_SOURCE_ID = "YuxuanLi1225/HiddenBench"
HIDDENBENCH_SOURCE_REVISION = "1e3c25b1fd798c6717f4df0463edd3825c8e37f9"
HIDDENBENCH_VARIANTS_PER_TASK = 5
HIDDENBENCH_PRIVATE_SALT = "hiddenbench-private-permutations-v1"
HIDDENBENCH_OPTION_SALT = "hiddenbench-option-permutations-v1"


def _require_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"HiddenBench field {key!r} must be a non-empty string")
    return value.strip()


def _require_task_id(row: Mapping[str, object]) -> int:
    value = row.get("id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("HiddenBench id must be a positive integer")
    return value


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"HiddenBench field {field!r} must be a sequence")
    strings = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(strings) != len(value):
        raise ValueError(f"HiddenBench field {field!r} must contain non-empty strings")
    return strings


def _ranked_permutations(size: int, *, salt: str, task_id: int) -> list[tuple[int, ...]]:
    permutations = list(itertools.permutations(range(size)))
    return sorted(
        permutations,
        key=lambda permutation: hashlib.sha256(
            f"{salt}\0{task_id}\0{','.join(str(value) for value in permutation)}".encode()
        ).hexdigest(),
    )


def select_private_permutations(task_id: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    """Cover every source private fact at receiver 0, then fill by salted rank."""

    if group_size not in {3, 4}:
        raise ValueError("official HiddenBench primary corpus requires group size 3 or 4")
    ranked = _ranked_permutations(
        group_size,
        salt=HIDDENBENCH_PRIVATE_SALT,
        task_id=task_id,
    )
    selected = [
        next(permutation for permutation in ranked if permutation[0] == source_index)
        for source_index in range(group_size)
    ]
    selected.extend(permutation for permutation in ranked if permutation not in selected)
    return tuple(selected[:HIDDENBENCH_VARIANTS_PER_TASK])


def select_option_permutations(
    task_id: int,
    option_count: int,
    *,
    correct_option_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Cover every answer-label position once, then fill by an independent salted rank."""

    if option_count not in {3, 4} or correct_option_index not in range(option_count):
        raise ValueError("HiddenBench option permutation inputs are invalid")
    ranked = _ranked_permutations(
        option_count,
        salt=HIDDENBENCH_OPTION_SALT,
        task_id=task_id,
    )
    selected = [
        next(
            permutation
            for permutation in ranked
            if permutation.index(correct_option_index) == target_position
        )
        for target_position in range(option_count)
    ]
    selected.extend(permutation for permutation in ranked if permutation not in selected)
    return tuple(selected[:HIDDENBENCH_VARIANTS_PER_TASK])


def partition_hiddenbench_task(
    row: Mapping[str, object],
    *,
    permutation_index: int,
    source_revision: str = HIDDENBENCH_SOURCE_REVISION,
) -> DistributedQAExample:
    task_id = _require_task_id(row)
    name = _require_string(row, "name")
    description = _require_string(row, "description")
    shared_information = _string_sequence(row.get("shared_information"), field="shared_information")
    hidden_information = _string_sequence(row.get("hidden_information"), field="hidden_information")
    possible_answers = _string_sequence(row.get("possible_answers"), field="possible_answers")
    correct_answer = _require_string(row, "correct_answer")
    if len(hidden_information) not in {3, 4}:
        raise ValueError("HiddenBench official group size must be 3 or 4")
    if len(possible_answers) not in {3, 4} or len(set(possible_answers)) != len(possible_answers):
        raise ValueError("HiddenBench must contain three or four unique answer options")
    if correct_answer not in possible_answers:
        raise ValueError("HiddenBench correct answer must be one of possible_answers")
    if permutation_index not in range(HIDDENBENCH_VARIANTS_PER_TASK):
        raise ValueError("HiddenBench permutation index must be in the frozen range 0..4")

    correct_index = possible_answers.index(correct_answer)
    private_permutation = select_private_permutations(task_id, len(hidden_information))[
        permutation_index
    ]
    option_permutation = select_option_permutations(
        task_id,
        len(possible_answers),
        correct_option_index=correct_index,
    )[permutation_index]
    labels = tuple("ABCD"[: len(possible_answers)])
    permuted_options = tuple(
        (label, possible_answers[source_index])
        for label, source_index in zip(labels, option_permutation, strict=True)
    )
    answer_label = labels[option_permutation.index(correct_index)]
    private_contexts = tuple(
        "Private information for this agent:\n- " + hidden_information[source_index]
        for source_index in private_permutation
    )
    support_ids = tuple(
        (f"hiddenbench:{task_id}:private:{source_index}",) for source_index in private_permutation
    )
    shared_prompt = "\n".join(f"- {fact}" for fact in shared_information)
    option_prompt = " ".join(f"({label}) {text}" for label, text in permuted_options)
    identity = f"{HIDDENBENCH_SOURCE_ID}\0{source_revision}\0{task_id}\0{permutation_index}"
    example_id = f"hiddenbench-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return DistributedQAExample(
        example_id=example_id,
        source_id=HIDDENBENCH_SOURCE_ID,
        source_revision=source_revision,
        source_split="official",
        source_example_id=str(task_id),
        question=(
            f"HiddenBench task: {name}\n{description}\nShared information:\n{shared_prompt}\n"
            f"Options: {option_prompt}\nReturn only the option label."
        ),
        answer_options=permuted_options,
        answer_label=answer_label,
        private_contexts=private_contexts,
        designated_receiver=0,
        support_ids=support_ids,
        metadata={
            "adapter": "hiddenbench-official-v1",
            "cluster_task_id": str(task_id),
            "permutation_index": permutation_index,
            "num_agents": len(hidden_information),
            "private_permutation": list(private_permutation),
            "option_permutation": list(option_permutation),
            "original_correct_answer": correct_answer,
            "rationale_present": "rationale" in row,
        },
    )
