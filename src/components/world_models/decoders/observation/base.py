"""Base interfaces for observation reconstruction decoders."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

class BaseObservationDecoder(nn.Module):
    """Common interface for decoding latent states back to observations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.representation_dim = config.get("representation_dim")
        if self.representation_dim is None:
            raise ValueError("Observation decoder requires 'representation_dim'")

        output_dim = config.get("output_dim")
        output_shape = config.get("output_shape")
        if output_shape is not None and output_dim is None:
            output_dim = int(math.prod(output_shape))
        if output_dim is None:
            raise ValueError("Observation decoder requires 'output_dim' or 'output_shape'")

        self.output_dim = int(output_dim)
        self.output_shape: Optional[Sequence[int]] = (
            tuple(int(dim) for dim in output_shape)
            if output_shape is not None
            else None
        )

        self._build_decoder()

    def _build_decoder(self) -> None:  # pragma: no cover - abstract hook
        raise NotImplementedError

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["BaseObservationDecoder"]
