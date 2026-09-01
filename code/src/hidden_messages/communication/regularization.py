"""Regularizers that keep learned hidden messages content-sensitive."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def batch_message_variance_loss(
    messages_by_sender: Sequence[torch.Tensor],
    *,
    target_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize per-feature collapse across examples for each sender independently."""
    if not messages_by_sender:
        raise ValueError("messages_by_sender cannot be empty")
    if target_std <= 0:
        raise ValueError("target_std must be positive")
    reference_shape = messages_by_sender[0].shape
    if len(reference_shape) != 3:
        raise ValueError("each message tensor must have shape [B, K, d_msg]")
    if reference_shape[0] < 2:
        raise ValueError("message variance requires at least two examples")
    if any(message.shape != reference_shape for message in messages_by_sender):
        raise ValueError("all sender message tensors must share one shape")

    sender_stds = torch.stack(
        [message.float().std(dim=0, unbiased=False) for message in messages_by_sender]
    )
    loss = nn.functional.relu(target_std - sender_stds).mean()
    return loss, sender_stds.mean()
