"""Simple multi-layer perceptron encoder."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: nn.Module | None = nn.Tanh(),
    ) -> None:
        super().__init__()
        layers = []
        last = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            if activation is not None:
                layers.append(activation)
            last = hs
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

