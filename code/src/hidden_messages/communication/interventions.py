"""Shape-preserving receiver-boundary interventions for causal audits."""

from __future__ import annotations

import torch

from hidden_messages.contracts import MessageIntervention


def deterministic_derangement(batch_size: int, *, seed: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("derangement requires at least two examples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shift = int(torch.randint(1, batch_size, (1,), generator=generator).item())
    indices = (torch.arange(batch_size) + shift) % batch_size
    return indices.to(device=device)


def _moment_matched_random(messages: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=messages.device)
    generator.manual_seed(seed)
    reduce_dims = tuple(range(3, messages.ndim))
    mean = messages.mean(dim=reduce_dims, keepdim=True)
    std = messages.std(dim=reduce_dims, keepdim=True, unbiased=False)
    noise = torch.randn(
        messages.shape,
        dtype=messages.dtype,
        device=messages.device,
        generator=generator,
    )
    noise_mean = noise.mean(dim=reduce_dims, keepdim=True)
    noise_std = noise.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-8)
    normalized_noise = (noise - noise_mean) / noise_std
    return normalized_noise * std + mean


def apply_message_intervention(
    matched: torch.Tensor,
    intervention: MessageIntervention,
    *,
    seed: int = 0,
    self_messages: torch.Tensor | None = None,
    counterfactual_messages: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an intervention to `[B, N, E, K, d_msg]` incoming messages."""

    if matched.ndim != 5:
        raise ValueError("matched messages must have shape [B, N, E, K, d_msg]")
    if intervention is MessageIntervention.MATCHED:
        return matched
    if intervention is MessageIntervention.ZERO:
        return torch.zeros_like(matched)
    if intervention is MessageIntervention.RANDOM_MOMENT_MATCHED:
        return _moment_matched_random(matched, seed=seed)
    if intervention is MessageIntervention.DERANGED:
        order = deterministic_derangement(matched.shape[0], seed=seed, device=matched.device)
        return matched.index_select(0, order)
    if intervention is MessageIntervention.SELF:
        if self_messages is None:
            raise ValueError("self_messages are required for SELF intervention")
        if self_messages.ndim != 4 or self_messages.shape[:2] != matched.shape[:2]:
            raise ValueError("self_messages must have shape [B, N, K, d_msg]")
        return self_messages.unsqueeze(2).expand(-1, -1, matched.shape[2], -1, -1)
    if intervention is MessageIntervention.COUNTERFACTUAL:
        if counterfactual_messages is None:
            raise ValueError("counterfactual_messages are required for COUNTERFACTUAL intervention")
        if counterfactual_messages.shape != matched.shape:
            raise ValueError("counterfactual_messages must match the incoming tensor shape")
        return counterfactual_messages
    raise AssertionError(f"unhandled intervention: {intervention}")
