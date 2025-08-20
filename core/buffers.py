"""Experience buffers used by various algorithms."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Simple on-policy storage for PPO-style algorithms."""

    size: int
    device: torch.device | str = "cpu"

    obs: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    dones: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)

    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, lam: float
    ) -> None:
        values = self.values + [last_value]
        advantages: List[torch.Tensor] = []
        gae = torch.zeros(1, device=self.device)
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
        self.advantages = torch.stack(advantages).squeeze(1)
        self.returns = self.advantages + torch.stack(self.values).squeeze(1)

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_minibatches(self, batch_size: int) -> Iterable[Dict[str, torch.Tensor]]:
        assert self.advantages is not None and self.returns is not None
        n_samples = len(self.obs)
        indices = np.random.permutation(n_samples)
        data = {
            "obs": torch.stack(self.obs).to(self.device),
            "actions": torch.stack(self.actions).to(self.device),
            "log_probs": torch.stack(self.log_probs).to(self.device),
            "advantages": self.advantages.to(self.device),
            "returns": self.returns.to(self.device),
            "values": torch.stack(self.values).to(self.device),
        }
        for start in range(0, n_samples, batch_size):
            mb_idx = indices[start : start + batch_size]
            yield {k: v[mb_idx] for k, v in data.items()}

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages = None
        self.returns = None


class ReplayBuffer:
    """Minimal replay buffer for off-policy algorithms."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.storage: List[Dict[str, Any]] = []
        self.idx = 0

    def add(self, transition: Dict[str, Any]) -> None:
        if len(self.storage) < self.size:
            self.storage.append(transition)
        else:
            self.storage[self.idx] = transition
        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size: int) -> Dict[str, Any]:
        batch = random.sample(self.storage, batch_size)
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


