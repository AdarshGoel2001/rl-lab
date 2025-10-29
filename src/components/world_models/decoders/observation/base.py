"""Base interfaces for observation reconstruction decoders."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

class BaseObservationDecoder(nn.Module):
    """Common interface for decoding latent states back to observations."""

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        merged_config: Dict[str, Any] = {}
        if config is not None:
            merged_config.update(config)
        if kwargs:
            merged_config.update(kwargs)
        self.config = merged_config
        self.device = torch.device(self.config.get("device", "cpu"))
        self.representation_dim = self.config.get("representation_dim")
        if self.representation_dim is None:
            raise ValueError("Observation decoder requires 'representation_dim'")

        output_dim = self.config.get("output_dim")
        output_shape = self.config.get("output_shape")
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
