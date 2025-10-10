"""Base interfaces for reward prediction heads."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class BaseRewardPredictor(nn.Module):
    """Maps latent states to immediate reward predictions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.representation_dim = config.get("representation_dim")
        if self.representation_dim is None:
            raise ValueError("Reward predictor requires 'representation_dim' in config")
        self._build_model()

    def _build_model(self) -> None:  # pragma: no cover - to be overridden
        raise NotImplementedError

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["BaseRewardPredictor"]

