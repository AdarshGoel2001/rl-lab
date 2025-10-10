"""Base observation adapter for world-model systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import torch


ObservationType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class BaseObservationAdapter(ABC, torch.nn.Module):
    """Transforms environment observations into latent-ready features."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, observations: ObservationType) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return encoder-ready features and auxiliary metadata."""
        raise NotImplementedError
