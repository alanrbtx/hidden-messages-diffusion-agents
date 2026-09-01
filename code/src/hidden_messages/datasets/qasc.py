"""Deterministic factual partitioning for the QASC-Distributed adapter."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from heapq import nsmallest

from hidden_messages.datasets.distributed_qa import DistributedQAExample

QASC_SOURCE_ID = "allenai/qasc"
QASC_SOURCE_REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
QASC_OPTION_LABELS = tuple("ABCDEFGH")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class QASCFactCandidate:
    fact_id: str
    source_example_id: str
    source_role: str
    text: str


def _require_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"QASC field {key!r} must be a non-empty string")
    return value.strip()


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"QASC field {field!r} must be a sequence")
    strings = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(strings) != len(value):
        raise ValueError(f"QASC field {field!r} must contain only non-empty strings")
    return strings


def parse_qasc_choices(row: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    choices = row.get("choices")
    if not isinstance(choices, Mapping):
        raise ValueError("QASC choices must be a mapping")
    labels = _string_sequence(choices.get("label"), field="choices.label")
    texts = _string_sequence(choices.get("text"), field="choices.text")
    if labels != QASC_OPTION_LABELS or len(texts) != len(QASC_OPTION_LABELS):
        raise ValueError("QASC must contain the ordered options A through H")
    if len(set(texts)) != len(texts):
        raise ValueError("QASC option texts must be unique")
    return tuple(zip(labels, texts, strict=True))


def build_qasc_training_fact_pool(
    rows: Sequence[Mapping[str, object]],
) -> tuple[QASCFactCandidate, ...]:
    candidates: list[QASCFactCandidate] = []
    seen_texts: set[str] = set()
    for row in rows:
        source_example_id = _require_string(row, "id")
        for source_role in ("fact1", "fact2"):
            text = _require_string(row, source_role)
            normalized = " ".join(text.casefold().split())
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            candidates.append(
                QASCFactCandidate(
                    fact_id=f"qasc:{source_example_id}:{source_role}",
                    source_example_id=source_example_id,
                    source_role=source_role,
                    text=text,
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.fact_id))


@cache
def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.casefold()))


def _stable_tie_break(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def select_qasc_distractors(
    row: Mapping[str, object],
    pool: Sequence[QASCFactCandidate],
    *,
    count_per_agent: int = 4,
    salt: str = "qasc-distributed-v1",
) -> tuple[tuple[QASCFactCandidate, ...], tuple[QASCFactCandidate, ...]]:
    """Select disjoint train-pool distractors by question overlap and fact length."""

    if count_per_agent < 0:
        raise ValueError("count_per_agent must be non-negative")
    source_example_id = _require_string(row, "id")
    question_tokens = _tokens(_require_string(row, "question"))
    choices = parse_qasc_choices(row)
    answer_token_sets = tuple(_tokens(text) for _, text in choices)
    gold_facts = (_require_string(row, "fact1"), _require_string(row, "fact2"))
    gold_normalized = {" ".join(fact.casefold().split()) for fact in gold_facts}
    selected: list[tuple[QASCFactCandidate, ...]] = []
    used_ids: set[str] = set()

    for agent_id, gold_fact in enumerate(gold_facts):
        target_length = len(_TOKEN_PATTERN.findall(gold_fact.casefold()))
        scored: list[tuple[int, int, str, QASCFactCandidate]] = []
        for candidate in pool:
            candidate_tokens = _tokens(candidate.text)
            if candidate.source_example_id == source_example_id or candidate.fact_id in used_ids:
                continue
            if " ".join(candidate.text.casefold().split()) in gold_normalized:
                continue
            if any(
                option_tokens and option_tokens <= candidate_tokens
                for option_tokens in answer_token_sets
            ):
                continue
            overlap = len(question_tokens & candidate_tokens)
            length_gap = abs(len(candidate_tokens) - target_length)
            tie_break = _stable_tie_break(salt, source_example_id, str(agent_id), candidate.fact_id)
            scored.append((-overlap, length_gap, tie_break, candidate))
        chosen = tuple(
            item[3]
            for item in nsmallest(
                count_per_agent,
                scored,
                key=lambda item: (item[0], item[1], item[2]),
            )
        )
        if len(chosen) != count_per_agent:
            raise ValueError(f"insufficient leakage-safe QASC distractors for {source_example_id}")
        used_ids.update(candidate.fact_id for candidate in chosen)
        selected.append(chosen)
    return selected[0], selected[1]


def partition_qasc_row(
    row: Mapping[str, object],
    *,
    source_split: str,
    distractors: tuple[tuple[QASCFactCandidate, ...], tuple[QASCFactCandidate, ...]],
    source_revision: str = QASC_SOURCE_REVISION,
) -> DistributedQAExample:
    source_example_id = _require_string(row, "id")
    question = _require_string(row, "question")
    choices = parse_qasc_choices(row)
    answer_label = _require_string(row, "answerKey")
    if answer_label not in QASC_OPTION_LABELS:
        raise ValueError("QASC answerKey must be one of A through H")
    gold_facts = (_require_string(row, "fact1"), _require_string(row, "fact2"))
    if gold_facts[0].casefold() == gold_facts[1].casefold():
        raise ValueError("QASC gold facts must be distinct")
    if len(distractors) != 2:
        raise ValueError("QASC requires exactly two agent distractor groups")
    distractor_ids = [candidate.fact_id for group in distractors for candidate in group]
    if len(distractor_ids) != len(set(distractor_ids)):
        raise ValueError("QASC distractors must be disjoint across agents")

    option_prompt = " ".join(f"({label}) {text}" for label, text in choices)
    contexts: list[str] = []
    support_ids: list[tuple[str, ...]] = []
    for agent_id, (gold_fact, group) in enumerate(zip(gold_facts, distractors, strict=True)):
        evidence = (gold_fact, *(candidate.text for candidate in group))
        contexts.append("Evidence:\n" + "\n".join(f"- {fact}" for fact in evidence))
        support_ids.append((f"qasc:{source_example_id}:fact{agent_id + 1}",))

    identity = f"{QASC_SOURCE_ID}\0{source_revision}\0{source_split}\0{source_example_id}"
    example_id = f"qasc-dist-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    answer_text = dict(choices)[answer_label]
    return DistributedQAExample(
        example_id=example_id,
        source_id=QASC_SOURCE_ID,
        source_revision=source_revision,
        source_split=source_split,
        source_example_id=source_example_id,
        question=f"{question}\nOptions: {option_prompt}\nReturn only the option label.",
        answer_options=choices,
        answer_label=answer_label,
        private_contexts=tuple(contexts),
        designated_receiver=0,
        support_ids=tuple(support_ids),
        metadata={
            "adapter": "qasc-distributed-v1",
            "answer_text": answer_text,
            "num_agents": 2,
            "distractors_per_agent": len(distractors[0]),
            "agent0_distractor_ids": [candidate.fact_id for candidate in distractors[0]],
            "agent1_distractor_ids": [candidate.fact_id for candidate in distractors[1]],
        },
    )
