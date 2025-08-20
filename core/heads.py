"""Policy and value heads built on top of feature encoders."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class CategoricalHead(nn.Module):
    """Policy head for discrete action spaces."""

    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_actions)

    def forward(self, feats: torch.Tensor) -> Categorical:
        logits = self.linear(feats)
        return Categorical(logits=logits)


class DiagGaussianHead(nn.Module):
    """Policy head for continuous actions with diagonal covariance."""

    def __init__(self, in_dim: int, action_dim: int, log_std: float = -0.5) -> None:
        super().__init__()
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std)

    def forward(self, feats: torch.Tensor) -> Normal:
        mean = self.mean(feats)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class ValueHead(nn.Module):
    """Single linear layer producing state value estimates."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.linear(feats).squeeze(-1)


def mlp(in_dim: int, out_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)) -> nn.Sequential:
    """Small helper to build MLPs for heads if needed."""

    layers = []
    last_dim = in_dim
    for hs in hidden_sizes:
        layers.append(nn.Linear(last_dim, hs))
        layers.append(nn.Tanh())
        last_dim = hs
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)

