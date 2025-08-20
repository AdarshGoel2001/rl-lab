"""Advantage estimation utilities."""

from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""

    values = torch.cat([values, last_value[None]], dim=0)
    gae = 0.0
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1] * (1 - dones[step])
            - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    advantages_t = torch.stack(advantages)
    returns_t = advantages_t + values[:-1]
    return advantages_t, returns_t


class GAE:
    """Callable advantage estimator using :func:`compute_gae`."""

    def __init__(self, gamma: float, lam: float) -> None:
        self.gamma = gamma
        self.lam = lam

    def __call__(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return compute_gae(
            rewards, values, dones, last_value, self.gamma, self.lam
        )

