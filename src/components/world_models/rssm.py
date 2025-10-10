"""Shared utilities for RSSM components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RSSMState:
    """Container for Dreamer-style RSSM latent state."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_mean: Optional[torch.Tensor] = None
    prior_std: Optional[torch.Tensor] = None
    posterior_mean: Optional[torch.Tensor] = None
    posterior_std: Optional[torch.Tensor] = None

    def latent(self) -> torch.Tensor:
        """Return concatenated latent representation used by heads."""
        return torch.cat([self.deterministic, self.stochastic], dim=-1)

    def detach(self) -> "RSSMState":
        """Detach tensors to stop gradients while keeping structure."""
        def _maybe_detach(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.detach() if isinstance(t, torch.Tensor) else None

        return RSSMState(
            deterministic=self.deterministic.detach(),
            stochastic=self.stochastic.detach(),
            prior_mean=_maybe_detach(self.prior_mean),
            prior_std=_maybe_detach(self.prior_std),
            posterior_mean=_maybe_detach(self.posterior_mean),
            posterior_std=_maybe_detach(self.posterior_std),
        )

    @property
    def batch_size(self) -> int:
        return self.deterministic.shape[0]

    @property
    def deterministic_dim(self) -> int:
        return self.deterministic.shape[-1]

    @property
    def stochastic_dim(self) -> int:
        return self.stochastic.shape[-1]

    def split_latent(self, latent: torch.Tensor) -> "RSSMState":
        """Construct a new state using deterministic/stochastic slices of ``latent``."""
        det_dim = self.deterministic_dim
        stochastic = latent[..., det_dim:]
        deterministic = latent[..., :det_dim]
        return RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
        )
