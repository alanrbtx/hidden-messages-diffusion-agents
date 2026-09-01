"""Canonical records for natural distributed-evidence QA adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

MetadataValue = bool | int | float | str | list[int] | list[str]


@dataclass(frozen=True, slots=True)
class DistributedQAExample:
    """One frozen partition with a predesignated receiver and explicit provenance."""

    example_id: str
    source_id: str
    source_revision: str
    source_split: str
    source_example_id: str
    question: str
    answer_options: tuple[tuple[str, str], ...]
    answer_label: str
    private_contexts: tuple[str, ...]
    designated_receiver: int
    support_ids: tuple[tuple[str, ...], ...]
    metadata: dict[str, MetadataValue]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class EvidenceFragment:
    """One source paragraph with explicit support and hop provenance."""

    fragment_id: str
    source_document_id: str
    title: str
    text: str
    is_support: bool
    hop_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DistributedOpenQAExample:
    """One open-answer partition with a predesignated receiver."""

    example_id: str
    source_id: str
    source_revision: str
    source_split: str
    source_example_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    private_contexts: tuple[str, ...]
    private_fragments: tuple[tuple[EvidenceFragment, ...], ...]
    designated_receiver: int
    support_ids: tuple[tuple[str, ...], ...]
    metadata: dict[str, MetadataValue]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
