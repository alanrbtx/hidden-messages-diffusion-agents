"""Sender-conditioned continuous prefixes in the receiver input-embedding space."""

from __future__ import annotations

import torch
from torch import nn

from hidden_messages.communication.compressor import MessageCompressor


class InputPrefixChannel(nn.Module):
    """Compress sender states and map the slots to calibrated input embeddings."""

    def __init__(
        self,
        model_dim: int,
        *,
        input_embedding_rms: float,
        message_slots: int = 8,
        message_dim: int = 256,
        width: int = 512,
        num_heads: int = 8,
        max_agents: int = 8,
    ) -> None:
        super().__init__()
        if input_embedding_rms <= 0:
            raise ValueError("input_embedding_rms must be positive")
        self.model_dim = model_dim
        self.message_slots = message_slots
        self.message_dim = message_dim
        self.compressor = MessageCompressor(
            model_dim,
            message_slots=message_slots,
            message_dim=message_dim,
            width=width,
            num_heads=num_heads,
            max_agents=max_agents,
        )
        self.prefix_projection = nn.Sequential(
            nn.Linear(message_dim, width, bias=False),
            nn.SiLU(),
            nn.Linear(width, model_dim, bias=False),
        )
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "input_embedding_rms",
            torch.tensor(float(input_embedding_rms), dtype=torch.float32),
        )

    def encode(
        self,
        hidden_states: torch.Tensor,
        *,
        segment_ids: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_ids: torch.Tensor,
        uncertainty: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return compact sender slots without consulting any receiver state."""

        return self.compressor(
            hidden_states,
            segment_ids=segment_ids,
            normalized_time=normalized_time,
            sender_ids=sender_ids,
            uncertainty=uncertainty,
            attention_mask=attention_mask,
        )

    def project(self, incoming_slots: torch.Tensor) -> torch.Tensor:
        """Map `[B, K, d_msg]` slots to input-like `[B, K, d_model]` prefixes."""

        if incoming_slots.ndim != 3:
            raise ValueError("incoming_slots must have shape [B, K, d_msg]")
        if incoming_slots.shape[1:] != (self.message_slots, self.message_dim):
            raise ValueError(
                "incoming_slots must match the configured message slot and dimension sizes"
            )
        normalized = nn.functional.layer_norm(incoming_slots, (self.message_dim,))
        projected = self.prefix_projection(normalized)
        inverse_rms = torch.rsqrt(
            projected.float().square().mean(dim=-1, keepdim=True).add(1e-12)
        ).to(dtype=projected.dtype)
        scale = self.log_scale.clamp(min=-2.0, max=2.0).exp()
        return projected * inverse_rms * self.input_embedding_rms.to(projected.dtype) * scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        segment_ids: torch.Tensor,
        normalized_time: torch.Tensor,
        sender_ids: torch.Tensor,
        uncertainty: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact messages and their receiver input-space realization."""

        messages = self.encode(
            hidden_states,
            segment_ids=segment_ids,
            normalized_time=normalized_time,
            sender_ids=sender_ids,
            uncertainty=uncertainty,
            attention_mask=attention_mask,
        )
        return messages, self.project(messages)
