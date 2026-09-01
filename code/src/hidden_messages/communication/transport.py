"""Synchronous all-to-all transport with explicit sender identities and byte accounting."""

from __future__ import annotations

import torch


def all_to_all_graph_mask(
    num_agents: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the directed all-to-all adjacency matrix with self edges disabled."""

    if num_agents < 2:
        raise ValueError("all-to-all transport requires at least two agents")
    mask = torch.ones((num_agents, num_agents), dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    return mask


def all_to_all_messages(messages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Jacobi-style incoming messages without self edges.

    Args:
        messages: Pre-fusion sender messages shaped `[B, N, K, d_msg]`.

    Returns:
        Incoming slots `[B, N, N-1, K, d_msg]` and sender IDs `[B, N, N-1]`.
    """

    if messages.ndim != 4:
        raise ValueError("messages must have shape [B, N, K, d_msg]")
    batch_size, agent_count, _, _ = messages.shape
    if agent_count < 2:
        raise ValueError("all-to-all transport requires at least two agents")

    incoming_by_receiver: list[torch.Tensor] = []
    senders_by_receiver: list[torch.Tensor] = []
    base_ids = torch.arange(agent_count, device=messages.device)
    graph_mask = all_to_all_graph_mask(agent_count, device=messages.device)
    for receiver in range(agent_count):
        sender_ids = base_ids[graph_mask[receiver]]
        incoming_by_receiver.append(messages.index_select(1, sender_ids))
        senders_by_receiver.append(sender_ids.expand(batch_size, -1))
    return torch.stack(incoming_by_receiver, dim=1), torch.stack(senders_by_receiver, dim=1)


def payload_bytes(
    messages: torch.Tensor,
    *,
    communication_rounds: int = 1,
    directed_edges: int | None = None,
) -> int:
    """Return transmitted bytes per example, excluding the batch dimension."""

    if messages.ndim != 4:
        raise ValueError("messages must have shape [B, N, K, d_msg]")
    if communication_rounds < 1:
        raise ValueError("communication_rounds must be positive")
    _, agent_count, slot_count, message_dim = messages.shape
    edge_count = directed_edges if directed_edges is not None else agent_count * (agent_count - 1)
    if edge_count < 0:
        raise ValueError("directed_edges must be non-negative")
    return communication_rounds * edge_count * slot_count * message_dim * messages.element_size()
