"""Validated run manifests and no-overwrite artifact directories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStage(StrEnum):
    ENGINEERING_SMOKE = "engineering_smoke"
    DEVELOPMENT = "development"
    POST_HOC = "post_hoc"
    PROSPECTIVE_CONFIRMATION = "prospective_confirmation"
    RELEASE_VALIDATION = "release_validation"


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    stage: EvidenceStage
    source_commit: str
    command: list[str]
    seeds: list[int]
    expected_outputs: list[str]
    dataset_identity: str | None = None
    model_identity: str | None = None
    protocol_identity: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def create_run_directory(root: Path, run_id: str) -> Path:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_immutable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
