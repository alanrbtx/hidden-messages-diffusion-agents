"""Deterministic two-agent partitioning for official MuSiQue-Answerable."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hidden_messages.datasets.distributed_qa import (
    DistributedOpenQAExample,
    EvidenceFragment,
)

MUSIQUE_SOURCE_ID = "StonyBrookNLP/musique"
MUSIQUE_SOURCE_REVISION = (
    "v1.0+sha256:98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
)
MUSIQUE_PARTITION_SALT = "musique-distributed-v1"


@dataclass(frozen=True, slots=True)
class MuSiQueParagraph:
    index: int
    title: str
    text: str
    is_support: bool


@dataclass(frozen=True, slots=True)
class MuSiQueHop:
    hop_id: int
    question: str
    answer: str
    paragraph_index: int


def _require_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MuSiQue field {key!r} must be a non-empty string")
    return value.strip()


def _require_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"MuSiQue field {key!r} must be an integer")
    return value


def _mapping_sequence(value: object, *, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"MuSiQue field {field!r} must be a sequence")
    mappings = tuple(item for item in value if isinstance(item, Mapping))
    if len(mappings) != len(value):
        raise ValueError(f"MuSiQue field {field!r} must contain only mappings")
    return mappings


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"MuSiQue field {field!r} must be a sequence")
    strings = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(strings) != len(value):
        raise ValueError(f"MuSiQue field {field!r} must contain only non-empty strings")
    return strings


def parse_musique_paragraphs(row: Mapping[str, object]) -> tuple[MuSiQueParagraph, ...]:
    paragraphs = tuple(
        MuSiQueParagraph(
            index=_require_int(paragraph, "idx"),
            title=_require_string(paragraph, "title"),
            text=_require_string(paragraph, "paragraph_text"),
            is_support=paragraph.get("is_supporting") is True,
        )
        for paragraph in _mapping_sequence(row.get("paragraphs"), field="paragraphs")
    )
    if len(paragraphs) < 2 or len({paragraph.index for paragraph in paragraphs}) != len(paragraphs):
        raise ValueError("MuSiQue paragraphs must have at least two unique indices")
    return paragraphs


def parse_musique_hops(row: Mapping[str, object]) -> tuple[MuSiQueHop, ...]:
    hops = tuple(
        MuSiQueHop(
            hop_id=_require_int(step, "id"),
            question=_require_string(step, "question"),
            answer=_require_string(step, "answer"),
            paragraph_index=_require_int(step, "paragraph_support_idx"),
        )
        for step in _mapping_sequence(
            row.get("question_decomposition"), field="question_decomposition"
        )
    )
    if len(hops) not in {2, 3, 4}:
        raise ValueError("MuSiQue-Answerable must contain two, three, or four hops")
    if len({hop.hop_id for hop in hops}) != len(hops):
        raise ValueError("MuSiQue hop IDs must be unique")
    if len({hop.paragraph_index for hop in hops}) != len(hops):
        raise ValueError("MuSiQue support paragraph indices must be unique")
    return hops


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _word_count(paragraph: MuSiQueParagraph) -> int:
    return len(paragraph.title.split()) + len(paragraph.text.split())


def assign_musique_paragraphs(
    source_example_id: str,
    paragraphs: tuple[MuSiQueParagraph, ...],
    hops: tuple[MuSiQueHop, ...],
    *,
    salt: str = MUSIQUE_PARTITION_SALT,
) -> tuple[tuple[MuSiQueParagraph, ...], tuple[MuSiQueParagraph, ...]]:
    """Split supports across agents, then balance all remaining paragraphs by length."""

    paragraph_by_index = {paragraph.index: paragraph for paragraph in paragraphs}
    support_indices = {paragraph.index for paragraph in paragraphs if paragraph.is_support}
    hop_support_indices = {hop.paragraph_index for hop in hops}
    if support_indices != hop_support_indices:
        raise ValueError("MuSiQue support flags differ from decomposition provenance")

    assigned: list[list[MuSiQueParagraph]] = [[], []]
    for hop_position, hop in enumerate(hops):
        assigned[hop_position % 2].append(paragraph_by_index[hop.paragraph_index])
    if not all(assigned) or any(len(group) == len(hops) for group in assigned):
        raise ValueError("MuSiQue supports must be distributed across both agents")

    target_counts = ((len(paragraphs) + 1) // 2, len(paragraphs) // 2)
    word_totals = [sum(_word_count(item) for item in group) for group in assigned]
    distractors = sorted(
        (paragraph for paragraph in paragraphs if not paragraph.is_support),
        key=lambda paragraph: (
            -_word_count(paragraph),
            _stable_digest(salt, source_example_id, "distractor", str(paragraph.index)),
        ),
    )
    for paragraph in distractors:
        candidates = [
            agent_id for agent_id in (0, 1) if len(assigned[agent_id]) < target_counts[agent_id]
        ]
        if not candidates:
            raise RuntimeError("MuSiQue paragraph assignment exceeded both target capacities")
        word_count = _word_count(paragraph)
        chosen = min(
            candidates,
            key=lambda agent_id: (
                abs((word_totals[agent_id] + word_count) - word_totals[1 - agent_id]),
                len(assigned[agent_id]),
                _stable_digest(
                    salt,
                    source_example_id,
                    str(paragraph.index),
                    str(agent_id),
                ),
            ),
        )
        assigned[chosen].append(paragraph)
        word_totals[chosen] += word_count

    if tuple(len(group) for group in assigned) != target_counts:
        raise RuntimeError("MuSiQue paragraph assignment did not reach frozen target counts")
    ordered = tuple(
        tuple(
            sorted(
                group,
                key=lambda paragraph: _stable_digest(
                    salt,
                    source_example_id,
                    "order",
                    str(agent_id),
                    str(paragraph.index),
                ),
            )
        )
        for agent_id, group in enumerate(assigned)
    )
    return ordered[0], ordered[1]


def _fragment(
    source_example_id: str,
    paragraph: MuSiQueParagraph,
    hop_ids: tuple[int, ...],
) -> EvidenceFragment:
    return EvidenceFragment(
        fragment_id=f"musique:{source_example_id}:paragraph:{paragraph.index}",
        source_document_id=str(paragraph.index),
        title=paragraph.title,
        text=paragraph.text,
        is_support=paragraph.is_support,
        hop_ids=hop_ids,
    )


def _render_context(fragments: tuple[EvidenceFragment, ...]) -> str:
    rendered = [
        f"[{fragment.source_document_id}] {fragment.title}\n{fragment.text}"
        for fragment in fragments
    ]
    return "Evidence paragraphs:\n\n" + "\n\n".join(rendered)


def partition_musique_row(
    row: Mapping[str, object],
    *,
    source_split: str,
    source_revision: str = MUSIQUE_SOURCE_REVISION,
) -> DistributedOpenQAExample:
    if row.get("answerable") is not True:
        raise ValueError("MuSiQue-Answerable row must have answerable=true")
    source_example_id = _require_string(row, "id")
    question = _require_string(row, "question")
    answer = _require_string(row, "answer")
    aliases = _string_sequence(row.get("answer_aliases"), field="answer_aliases")
    paragraphs = parse_musique_paragraphs(row)
    hops = parse_musique_hops(row)
    groups = assign_musique_paragraphs(source_example_id, paragraphs, hops)
    hop_ids_by_paragraph = {hop.paragraph_index: (hop.hop_id,) for hop in hops}
    private_fragments = tuple(
        tuple(
            _fragment(
                source_example_id,
                paragraph,
                hop_ids_by_paragraph.get(paragraph.index, ()),
            )
            for paragraph in group
        )
        for group in groups
    )
    support_ids = tuple(
        tuple(fragment.fragment_id for fragment in group if fragment.is_support)
        for group in private_fragments
    )
    all_fragment_ids = [fragment.fragment_id for group in private_fragments for fragment in group]
    if len(all_fragment_ids) != len(set(all_fragment_ids)) or len(all_fragment_ids) != len(
        paragraphs
    ):
        raise RuntimeError("MuSiQue paragraph partition is not exhaustive and disjoint")

    identity = f"{MUSIQUE_SOURCE_ID}\0{source_revision}\0{source_split}\0{source_example_id}"
    example_id = f"musique-dist-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    context_word_counts = [
        sum(len(fragment.title.split()) + len(fragment.text.split()) for fragment in group)
        for group in private_fragments
    ]
    return DistributedOpenQAExample(
        example_id=example_id,
        source_id=MUSIQUE_SOURCE_ID,
        source_revision=source_revision,
        source_split=source_split,
        source_example_id=source_example_id,
        question=f"{question}\nReturn only a short answer.",
        answer=answer,
        answer_aliases=aliases,
        private_contexts=tuple(_render_context(group) for group in private_fragments),
        private_fragments=private_fragments,
        designated_receiver=0,
        support_ids=support_ids,
        metadata={
            "adapter": "musique-distributed-v1",
            "num_agents": 2,
            "num_hops": len(hops),
            "paragraph_count": len(paragraphs),
            "agent0_paragraph_ids": [
                fragment.source_document_id for fragment in private_fragments[0]
            ],
            "agent1_paragraph_ids": [
                fragment.source_document_id for fragment in private_fragments[1]
            ],
            "agent0_hop_ids": [hop.hop_id for index, hop in enumerate(hops) if index % 2 == 0],
            "agent1_hop_ids": [hop.hop_id for index, hop in enumerate(hops) if index % 2 == 1],
            "context_word_gap": abs(context_word_counts[0] - context_word_counts[1]),
        },
    )
