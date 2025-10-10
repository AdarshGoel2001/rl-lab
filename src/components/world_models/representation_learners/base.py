"""Base representation learner for world-model latents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseRepresentationLearner(nn.Module, ABC):
    """Defines the interface for modules mapping encoder features to latents."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self._build_learner()

    @abstractmethod
    def _build_learner(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decoding not implemented for this learner")

    def representation_loss(self, features: torch.Tensor, **_) -> Dict[str, torch.Tensor]:
        zero = torch.zeros(1, device=features.device)
        return {
            "representation_loss": zero,
        }

    @property
    @abstractmethod
    def representation_dim(self) -> int:
        raise NotImplementedError

    @property
    def supports_decoding(self) -> bool:
        try:
            dummy = torch.zeros(1, self.representation_dim, device=self.device)
            self.decode(dummy)
            return True
        except (NotImplementedError, AttributeError):
            return False

    def get_representation_info(self) -> Dict[str, Any]:
        return {
            "representation_dim": self.representation_dim,
            "supports_decoding": self.supports_decoding,
            "learner_type": self.__class__.__name__,
            "device": str(self.device),
        }
