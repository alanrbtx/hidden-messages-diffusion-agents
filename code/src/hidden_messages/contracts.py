"""Core immutable contracts shared by samplers, agents, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch


class MessageIntervention(StrEnum):
    """Receiver-boundary interventions used by the registered causal audit."""

    MATCHED = "matched"
    ZERO = "zero"
    RANDOM_MOMENT_MATCHED = "random_moment_matched"
    DERANGED = "deranged"
    SELF = "self"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True, slots=True)
class HiddenMessage:
    """One sender's message for one example at one normalized denoising time."""

    sender_id: int
    example_id: str
    normalized_time: float
    slots: torch.Tensor
    dtype_bytes: int
    source_layer: int
    checksum: str

    def __post_init__(self) -> None:
        if self.sender_id < 0:
            raise ValueError("sender_id must be non-negative")
        if not 0.0 <= self.normalized_time <= 1.0:
            raise ValueError("normalized_time must be in [0, 1]")
        if self.slots.ndim != 2:
            raise ValueError("slots must have shape [K, d_msg]")
        if self.dtype_bytes != self.slots.element_size():
            raise ValueError("dtype_bytes must equal slots.element_size()")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Sparse, claim-auditable snapshot of one agent's denoising state."""

    step: int
    normalized_time: float
    canvas_checksum: str
    masked_tokens: int
    changed_tokens: int
    mean_confidence_pre: float | None = None
    mean_confidence_post: float | None = None
    beneficial_revisions: int | None = None
    harmful_revisions: int | None = None


@dataclass(slots=True)
class AgentState:
    """Mutable state owned by exactly one agent trajectory."""

    agent_id: int
    role_id: int
    private_context_ids: torch.Tensor
    canvas_ids: torch.Tensor
    rng_state: torch.Tensor
    message_cache: dict[int, HiddenMessage] = field(default_factory=dict)
    trajectory_log: list[StepRecord] = field(default_factory=list)

    def assert_independent_from(self, other: AgentState) -> None:
        """Fail if mutable tensors alias another agent's storage."""

        for name in ("private_context_ids", "canvas_ids", "rng_state"):
            ours = getattr(self, name)
            theirs = getattr(other, name)
            if ours.data_ptr() == theirs.data_ptr():
                raise AssertionError(f"agent tensor aliases another trajectory: {name}")
