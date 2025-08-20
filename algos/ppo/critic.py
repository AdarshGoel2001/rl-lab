"""Value network wrapper for PPO."""

from __future__ import annotations

from typing import Any

import torch


class PPOCritic(torch.nn.Module):
    """Combines encoder and value head to estimate state values."""

    def __init__(self, encoder: Any, value_head: Any) -> None:
        super().__init__()
        self.encoder = encoder
        self.value_head = value_head

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(obs)
        return self.value_head(feats)


