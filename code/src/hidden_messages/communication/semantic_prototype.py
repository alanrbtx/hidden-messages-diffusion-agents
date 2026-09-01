"""Supervised continuous semantic bottleneck for interface-diagnostic communication."""

from __future__ import annotations

import torch
from torch import nn


class SemanticPrototypeChannel(nn.Module):
    """Map one sender state to a soft distribution over fixed continuous prefixes."""

    def __init__(
        self,
        model_dim: int,
        prototype_embeddings: torch.Tensor,
        *,
        width: int = 512,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if prototype_embeddings.ndim != 3:
            raise ValueError("prototype_embeddings must have shape [C, K, D]")
        if prototype_embeddings.shape[-1] != model_dim:
            raise ValueError("prototype model dimension does not match model_dim")
        if prototype_embeddings.shape[0] < 2 or prototype_embeddings.shape[1] < 1:
            raise ValueError("at least two classes and one prefix token are required")
        if width < 1:
            raise ValueError("width must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model_dim = model_dim
        self.num_classes = int(prototype_embeddings.shape[0])
        self.prefix_length = int(prototype_embeddings.shape[1])
        self.temperature = float(temperature)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, width),
            nn.SiLU(),
            nn.Linear(width, self.num_classes),
        )
        self.prototype_embeddings: torch.Tensor
        self.register_buffer("prototype_embeddings", prototype_embeddings.detach().clone())

    def classify(self, sender_state: torch.Tensor) -> torch.Tensor:
        """Return unnormalized semantic scores for one state per example."""

        if sender_state.ndim != 2 or sender_state.shape[1] != self.model_dim:
            raise ValueError("sender_state must have shape [B, model_dim]")
        return self.classifier(sender_state)

    def encode(self, sender_state: torch.Tensor) -> torch.Tensor:
        """Return a continuous simplex-valued message."""

        return torch.softmax(self.classify(sender_state).float() / self.temperature, dim=-1)

    def project(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Map semantic probabilities to a soft mixture of continuous prefix prototypes."""

        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError("probabilities must have shape [B, num_classes]")
        return torch.einsum(
            "bc,ckd->bkd",
            probabilities.to(dtype=self.prototype_embeddings.dtype),
            self.prototype_embeddings,
        )

    def forward(self, sender_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the continuous message and its receiver-side prefix realization."""

        probabilities = self.encode(sender_state)
        return probabilities, self.project(probabilities)
