"""Perceiver-style compression of sender hidden states into fixed message slots."""

from __future__ import annotations

import torch
from torch import nn


class MessageCompressor(nn.Module):
    """Compress `[B, L, d_model]` states into `[B, K, d_msg]` messages."""

    def __init__(
        self,
        model_dim: int,
        *,
        message_slots: int = 4,
        message_dim: int = 128,
        width: int = 256,
        num_heads: int = 4,
        num_segments: int = 3,
        max_agents: int = 8,
        uncertainty_dim: int = 3,
    ) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if min(model_dim, message_slots, message_dim, width, max_agents) <= 0:
            raise ValueError("all dimensions must be positive")

        self.model_dim = model_dim
        self.message_slots = message_slots
        self.message_dim = message_dim
        self.uncertainty_dim = uncertainty_dim

        self.hidden_projection = nn.Linear(model_dim, width)
        self.segment_embedding = nn.Embedding(num_segments, width)
        self.sender_embedding = nn.Embedding(max_agents, width)
        self.uncertainty_projection = nn.Linear(uncertainty_dim, width, bias=False)
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.query_slots = nn.Parameter(torch.empty(message_slots, width))
        self.attention = nn.MultiheadAttention(
            width,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, message_dim),
        )
        self.output_norm = nn.LayerNorm(message_dim)
        nn.init.normal_(self.query_slots, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        segment_ids: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_ids: torch.Tensor,
        uncertainty: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if hidden_dim != self.model_dim:
            raise ValueError(f"expected hidden dim {self.model_dim}, got {hidden_dim}")
        if segment_ids.shape != (batch_size, sequence_length):
            raise ValueError("segment_ids must have shape [B, L]")
        if uncertainty.shape != (batch_size, sequence_length, self.uncertainty_dim):
            raise ValueError("uncertainty must have shape [B, L, uncertainty_dim]")
        if normalized_time.shape not in {(batch_size,), (batch_size, 1)}:
            raise ValueError("normalized_time must have shape [B] or [B, 1]")
        if sender_ids.shape != (batch_size,):
            raise ValueError("sender_ids must have shape [B]")
        if attention_mask is not None and attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [B, L]")

        time = normalized_time.reshape(batch_size, 1).to(dtype=hidden_states.dtype)
        key_value = self.hidden_projection(hidden_states)
        key_value = key_value + self.segment_embedding(segment_ids)
        key_value = key_value + self.uncertainty_projection(uncertainty)
        key_value = key_value + self.timestep_embedding(time).unsqueeze(1)

        queries = self.query_slots.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.sender_embedding(sender_ids).unsqueeze(1)
        queries = queries + self.timestep_embedding(time).unsqueeze(1)

        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        slots, _ = self.attention(
            query=queries,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(self.bottleneck(slots))
