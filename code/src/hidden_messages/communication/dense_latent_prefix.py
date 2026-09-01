"""Dense sender-state messages with a learned receiver-side prefix decoder."""

from __future__ import annotations

import torch
from torch import nn


class DenseLatentPrefixChannel(nn.Module):
    """Encode one sender state into signed latent slots and decode them as prefixes.

    Class labels supervise the representation during training, but the transmitted payload is the
    dense `[B, K, d_msg]` tensor. Neither class probabilities nor token IDs are receiver inputs.
    """

    def __init__(
        self,
        model_dim: int,
        num_classes: int,
        *,
        input_embedding_rms: float,
        message_slots: int = 4,
        message_dim: int = 128,
        width: int = 512,
        classifier_width: int = 512,
    ) -> None:
        super().__init__()
        if (
            min(
                model_dim,
                num_classes,
                message_slots,
                message_dim,
                width,
                classifier_width,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")
        if num_classes < 2:
            raise ValueError("at least two supervised classes are required")
        if input_embedding_rms <= 0:
            raise ValueError("input_embedding_rms must be positive")

        self.model_dim = model_dim
        self.num_classes = num_classes
        self.message_slots = message_slots
        self.message_dim = message_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, width),
            nn.SiLU(),
            nn.Linear(width, message_slots * message_dim),
        )
        self.message_norm = nn.LayerNorm(message_dim)
        flattened_dim = message_slots * message_dim
        self.message_classifier = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, classifier_width),
            nn.SiLU(),
            nn.Linear(classifier_width, num_classes),
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

    def encode(self, sender_state: torch.Tensor) -> torch.Tensor:
        """Return signed dense slots from one sender state per example."""

        if sender_state.ndim != 2 or sender_state.shape[1] != self.model_dim:
            raise ValueError("sender_state must have shape [B, model_dim]")
        raw = self.state_encoder(sender_state)
        slots = raw.reshape(sender_state.shape[0], self.message_slots, self.message_dim)
        return self.message_norm(slots)

    def classify_messages(self, messages: torch.Tensor) -> torch.Tensor:
        """Return auxiliary class logits from already encoded messages."""

        if messages.ndim != 3 or messages.shape[1:] != (
            self.message_slots,
            self.message_dim,
        ):
            raise ValueError("messages must have shape [B, K, d_msg]")
        return self.message_classifier(messages.flatten(start_dim=1))

    def classify(self, sender_state: torch.Tensor) -> torch.Tensor:
        """Return auxiliary class logits without changing the transmitted representation."""

        return self.classify_messages(self.encode(sender_state))

    def project(self, messages: torch.Tensor) -> torch.Tensor:
        """Decode dense messages into calibrated continuous receiver prefix embeddings."""

        if messages.ndim != 3 or messages.shape[1:] != (
            self.message_slots,
            self.message_dim,
        ):
            raise ValueError("messages must have shape [B, K, d_msg]")
        projected = self.prefix_projection(messages)
        inverse_rms = torch.rsqrt(
            projected.float().square().mean(dim=-1, keepdim=True).add(1e-12)
        ).to(dtype=projected.dtype)
        scale = self.log_scale.clamp(min=-2.0, max=2.0).exp()
        return projected * inverse_rms * self.input_embedding_rms.to(projected.dtype) * scale

    def freeze_representation(self) -> None:
        """Freeze sender encoding and its auxiliary classifier before prefix-decoder fitting."""

        for module in (self.state_encoder, self.message_norm, self.message_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, sender_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.encode(sender_state)
        return messages, self.project(messages)
