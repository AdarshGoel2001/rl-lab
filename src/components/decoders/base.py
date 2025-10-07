"""Base decoder interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for decoders that map latent representations to observations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self._build_decoder()

    @abstractmethod
    def _build_decoder(self) -> None:
        """Construct decoder network."""

    @abstractmethod
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation into observation reconstruction."""

