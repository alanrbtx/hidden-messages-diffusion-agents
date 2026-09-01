"""Canvas-only, timestep- and uncertainty-conditioned message fusion."""

from __future__ import annotations

import torch
from torch import nn


class MessageFusion(nn.Module):
    """Fuse incoming slots while leaving non-canvas positions bitwise unchanged."""

    def __init__(
        self,
        model_dim: int,
        *,
        message_dim: int = 128,
        width: int = 256,
        num_heads: int = 4,
        max_agents: int = 8,
        uncertainty_dim: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.model_dim = model_dim
        self.message_dim = message_dim
        self.uncertainty_dim = uncertainty_dim

        self.query_norm = nn.LayerNorm(model_dim)
        self.query_projection = nn.Linear(model_dim, width)
        self.message_projection = nn.Linear(message_dim, width)
        self.sender_embedding = nn.Embedding(max_agents, width)
        self.age_projection = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(
            width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(width, model_dim)
        self.gate = nn.Sequential(
            nn.Linear(uncertainty_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

        # Exact identity at initialization. Gradients first open the output projection, then the
        # rest of the channel; no-message parity does not rely on an approximately zero sigmoid.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        gate_output = self.gate[-1]
        if not isinstance(gate_output, nn.Linear):
            raise TypeError("gate output must be a linear layer")
        nn.init.zeros_(gate_output.weight)
        assert gate_output.bias is not None
        nn.init.constant_(gate_output.bias, -4.0)

    def forward(
        self,
        receiver_hidden: torch.Tensor,
        *,
        incoming_slots: torch.Tensor,
        sender_ids: torch.Tensor,
        message_ages: torch.Tensor,
        normalized_time: torch.Tensor,
        uncertainty: torch.Tensor,
        canvas_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = receiver_hidden.shape
        if hidden_dim != self.model_dim:
            raise ValueError(f"expected hidden dim {self.model_dim}, got {hidden_dim}")
        if incoming_slots.ndim != 5:
            raise ValueError("incoming_slots must have shape [B, R, S, K, d_msg]")
        if incoming_slots.shape[0] != batch_size or incoming_slots.shape[-1] != self.message_dim:
            raise ValueError("incoming_slots batch/message dimensions do not match")

        receiver_count = incoming_slots.shape[1]
        if receiver_count != 1:
            raise ValueError("MessageFusion handles one receiver axis at a time; expected R=1")
        edge_count, slot_count = incoming_slots.shape[2:4]
        if sender_ids.shape != (batch_size, 1, edge_count):
            raise ValueError("sender_ids must have shape [B, 1, E]")
        if message_ages.shape != (batch_size, 1, edge_count):
            raise ValueError("message_ages must have shape [B, 1, E]")
        if uncertainty.shape != (batch_size, sequence_length, self.uncertainty_dim):
            raise ValueError("uncertainty must have shape [B, L, uncertainty_dim]")
        if canvas_mask.shape != (batch_size, sequence_length):
            raise ValueError("canvas_mask must have shape [B, L]")
        if normalized_time.shape not in {(batch_size,), (batch_size, 1)}:
            raise ValueError("normalized_time must have shape [B] or [B, 1]")

        messages = incoming_slots[:, 0].reshape(
            batch_size,
            edge_count * slot_count,
            self.message_dim,
        )
        sender_features = self.sender_embedding(sender_ids[:, 0])
        sender_features = (
            sender_features.unsqueeze(2)
            .expand(-1, -1, slot_count, -1)
            .reshape(batch_size, edge_count * slot_count, -1)
        )
        age_features = self.age_projection(message_ages[:, 0].unsqueeze(-1))
        age_features = (
            age_features.unsqueeze(2)
            .expand(-1, -1, slot_count, -1)
            .reshape(batch_size, edge_count * slot_count, -1)
        )

        queries = self.query_projection(self.query_norm(receiver_hidden))
        key_value = self.message_projection(messages) + sender_features + age_features
        attended, _ = self.attention(
            query=queries,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        delta = self.output_projection(attended)

        time = normalized_time.reshape(batch_size, 1, 1).to(dtype=receiver_hidden.dtype)
        time = time.expand(-1, sequence_length, -1)
        gate_value = torch.sigmoid(self.gate(torch.cat([uncertainty, time], dim=-1)))
        gate_value = gate_value * canvas_mask.unsqueeze(-1).to(dtype=gate_value.dtype)
        fused = receiver_hidden + gate_value * delta
        return fused, gate_value
