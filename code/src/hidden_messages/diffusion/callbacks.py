"""Read-only denoising-step callback API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class DenoisingStep:
    block_index: int
    step_index: int
    normalized_time: float
    canvas: torch.Tensor
    hidden_states: torch.Tensor
    pre_logits: torch.Tensor
    attention_mask: torch.Tensor
    canvas_mask: torch.Tensor


class DenoisingCallback(Protocol):
    def __call__(self, state: DenoisingStep) -> torch.Tensor:
        """Return fused logits; direct canvas mutation is outside the interface."""


class IdentityCallback:
    def __call__(self, state: DenoisingStep) -> torch.Tensor:
        return state.pre_logits
