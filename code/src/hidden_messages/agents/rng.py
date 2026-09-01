"""Independent reproducible PRNG streams for agent trajectories."""

from __future__ import annotations

import torch


class AgentRNGPool:
    def __init__(self, num_agents: int, *, base_seed: int, device: torch.device | str) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.device = torch.device(device)
        self.generators: list[torch.Generator] = []
        for agent_id in range(num_agents):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(base_seed + 1_000_003 * agent_id)
            self.generators.append(generator)

    def rand(self, agent_id: int, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        generator = self.generators[agent_id]
        return torch.rand(shape, generator=generator, device=self.device, dtype=dtype)

    def states(self) -> list[torch.Tensor]:
        return [generator.get_state().clone() for generator in self.generators]
