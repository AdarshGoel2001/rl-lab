"""Latent representation abstractions for world-model paradigms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class LatentBatch:
    """Container bundling latent tensors and auxiliary encoder outputs."""

    latent: torch.Tensor
    features: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def detach(self) -> "LatentBatch":
        """Return a detached copy suitable for model rollouts."""
        detached_features = self.features.detach() if isinstance(self.features, torch.Tensor) else None
        detached_extras = {}
        for key, value in self.extras.items():
            if isinstance(value, torch.Tensor):
                detached_extras[key] = value.detach()
            elif hasattr(value, "detach") and callable(getattr(value, "detach")):
                detached_extras[key] = value.detach()
            else:
                detached_extras[key] = value
        return LatentBatch(latent=self.latent.detach(), features=detached_features, extras=detached_extras)
