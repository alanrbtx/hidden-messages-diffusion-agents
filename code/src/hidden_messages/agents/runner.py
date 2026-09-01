"""Batched independent agent canvases, sampling streams, and trajectory logging."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import torch

from hidden_messages.agents.rng import AgentRNGPool
from hidden_messages.contracts import StepRecord
from hidden_messages.utils.checksums import sha256_tensor

LogitsFunction = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
UpdateMaskFunction = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True, slots=True)
class MultiAgentRunResult:
    """Immutable run metadata around independently owned final canvas tensors."""

    final_canvases: torch.Tensor
    designated_receiver_id: int
    trajectory_logs: tuple[tuple[tuple[StepRecord, ...], ...], ...]

    @property
    def designated_canvases(self) -> torch.Tensor:
        """Return a copy of the predesignated receiver output for every example."""

        return self.final_canvases[:, self.designated_receiver_id].clone()

    def majority_vote_canvases(self) -> torch.Tensor:
        """Select the most frequent complete canvas; ties favor the designated receiver."""

        voted: list[torch.Tensor] = []
        for example_canvases in self.final_canvases:
            keys = [tuple(int(token) for token in canvas.tolist()) for canvas in example_canvases]
            counts = Counter(keys)
            max_count = max(counts.values())
            designated_key = keys[self.designated_receiver_id]
            winner = (
                designated_key
                if counts[designated_key] == max_count
                else next(key for key in keys if counts[key] == max_count)
            )
            voted.append(torch.tensor(winner, dtype=torch.long, device=self.final_canvases.device))
        return torch.stack(voted)


class TrajectoryLogger:
    """Record per-example, per-agent canvas evolution without retaining logits."""

    def __init__(self, batch_size: int, num_agents: int) -> None:
        self._records: list[list[list[StepRecord]]] = [
            [[] for _ in range(num_agents)] for _ in range(batch_size)
        ]

    def log(
        self,
        *,
        step: int,
        normalized_time: float,
        before: torch.Tensor,
        after: torch.Tensor,
        update_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        log_normalizer = torch.logsumexp(logits.float(), dim=-1)
        max_logits = logits.float().amax(dim=-1)
        confidence = torch.exp(max_logits - log_normalizer)
        changed = before != after
        after_cpu = after.detach().to(device="cpu")
        update_mask_cpu = update_mask.detach().to(device="cpu")
        confidence_cpu = confidence.detach().to(device="cpu")
        changed_cpu = changed.detach().to(device="cpu")
        batch_size, num_agents, _ = after.shape
        for example_index in range(batch_size):
            for agent_id in range(num_agents):
                selected = update_mask_cpu[example_index, agent_id]
                mean_confidence = (
                    float(confidence_cpu[example_index, agent_id][selected].mean().item())
                    if bool(selected.any())
                    else None
                )
                self._records[example_index][agent_id].append(
                    StepRecord(
                        step=step,
                        normalized_time=normalized_time,
                        canvas_checksum=sha256_tensor(after_cpu[example_index, agent_id]),
                        masked_tokens=int(selected.sum().item()),
                        changed_tokens=int(changed_cpu[example_index, agent_id].sum().item()),
                        mean_confidence_pre=mean_confidence,
                    )
                )

    def freeze(self) -> tuple[tuple[tuple[StepRecord, ...], ...], ...]:
        return tuple(
            tuple(tuple(agent_records) for agent_records in example_records)
            for example_records in self._records
        )


@dataclass(slots=True)
class MultiAgentBatch:
    canvases: torch.Tensor
    private_context_ids: torch.Tensor
    rng: AgentRNGPool
    designated_receiver_id: int = 0

    @classmethod
    def create(
        cls,
        *,
        canvases: torch.Tensor,
        private_context_ids: torch.Tensor,
        base_seed: int,
        designated_receiver_id: int = 0,
    ) -> MultiAgentBatch:
        if canvases.ndim != 3:
            raise ValueError("canvases must have shape [B, N, L]")
        if private_context_ids.ndim != 3 or private_context_ids.shape[:2] != canvases.shape[:2]:
            raise ValueError("private_context_ids must have shape [B, N, C]")
        if not 0 <= designated_receiver_id < canvases.shape[1]:
            raise ValueError("designated_receiver_id must index the agent dimension")
        owned_canvases = canvases.clone().contiguous()
        owned_contexts = private_context_ids.clone().contiguous()
        return cls(
            canvases=owned_canvases,
            private_context_ids=owned_contexts,
            rng=AgentRNGPool(
                canvases.shape[1],
                base_seed=base_seed,
                device=canvases.device,
            ),
            designated_receiver_id=designated_receiver_id,
        )

    @property
    def num_agents(self) -> int:
        return self.canvases.shape[1]

    def sample_independent_updates(
        self,
        logits: torch.Tensor,
        *,
        update_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Update each canvas from its own logits and PRNG without cross-agent assignment."""

        if logits.shape[:3] != self.canvases.shape:
            raise ValueError("logits must have shape [B, N, L, V]")
        if update_mask.shape != self.canvases.shape:
            raise ValueError("update_mask must have shape [B, N, L]")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        next_canvases = self.canvases.clone()
        for agent_id in range(self.num_agents):
            agent_logits = logits[:, agent_id]
            if temperature > 0:
                uniform = self.rng.rand(
                    agent_id,
                    tuple(agent_logits.shape),
                    dtype=torch.float64,
                ).clamp_(min=1e-12, max=1.0 - 1e-12)
                gumbel = -torch.log(-torch.log(uniform))
                sampled = torch.argmax(
                    agent_logits.to(torch.float64) / temperature + gumbel,
                    dim=-1,
                )
            else:
                sampled = torch.argmax(agent_logits, dim=-1)
            agent_mask = update_mask[:, agent_id]
            next_canvases[:, agent_id] = torch.where(
                agent_mask,
                sampled,
                self.canvases[:, agent_id],
            )
        self.canvases = next_canvases
        return next_canvases

    def run_no_communication(
        self,
        logits_function: LogitsFunction,
        *,
        steps: int,
        update_mask: torch.Tensor | UpdateMaskFunction,
        temperature: float,
    ) -> MultiAgentRunResult:
        """Run independent trajectories in one batched tensor without any message path."""

        if steps < 1:
            raise ValueError("steps must be positive")
        logger = TrajectoryLogger(self.canvases.shape[0], self.num_agents)
        for step in range(steps):
            before = self.canvases.clone()
            logits = logits_function(before, self.private_context_ids, step)
            if logits.ndim != 4 or logits.shape[:3] != before.shape:
                raise ValueError("logits_function must return [B, N, L, V]")
            resolved_mask = update_mask(before, step) if callable(update_mask) else update_mask
            if resolved_mask.shape != before.shape or resolved_mask.dtype != torch.bool:
                raise ValueError("update_mask must be boolean with shape [B, N, L]")
            after = self.sample_independent_updates(
                logits,
                update_mask=resolved_mask,
                temperature=temperature,
            )
            logger.log(
                step=step,
                normalized_time=max(0.0, 1.0 - (step + 1) / steps),
                before=before,
                after=after,
                update_mask=resolved_mask,
                logits=logits,
            )
        return MultiAgentRunResult(
            final_canvases=self.canvases.clone(),
            designated_receiver_id=self.designated_receiver_id,
            trajectory_logs=logger.freeze(),
        )

    def pairwise_hamming_fraction(self) -> torch.Tensor:
        """Return `[B, N, N]` pairwise token-disagreement fractions."""

        left = self.canvases.unsqueeze(2)
        right = self.canvases.unsqueeze(1)
        return (left != right).float().mean(dim=-1)
