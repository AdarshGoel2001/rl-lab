"""Base interfaces and utilities for representation learners."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypeVar, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


LatentStateT = TypeVar("LatentStateT", bound="LatentState")


@runtime_checkable
class LatentState(Protocol):
    """Lightweight protocol describing the minimal latent-state contract."""

    def to_tensor(self) -> torch.Tensor:
        """Return a flattened tensor view of the latent representation."""

    def to(self: LatentStateT, device: torch.device | str) -> LatentStateT:
        """Move latent tensors to the requested device."""

    def detach(self: LatentStateT) -> LatentStateT:
        """Detach latent tensors from the computation graph."""

    def clone(self: LatentStateT) -> LatentStateT:
        """Deep-copy latent tensors."""


@dataclass
class RSSMState:
    """Deterministic/stochastic latent container used by Dreamer-style models."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self.stochastic], dim=-1)

    def clone(self) -> "RSSMState":
        return RSSMState(
            deterministic=self.deterministic.clone(),
            stochastic=self.stochastic.clone(),
            mean=self.mean.clone(),
            std=self.std.clone(),
        )

    def detach(self) -> "RSSMState":
        return RSSMState(
            deterministic=self.deterministic.detach(),
            stochastic=self.stochastic.detach(),
            mean=self.mean.detach(),
            std=self.std.detach(),
        )

    def to(self, device: torch.device | str) -> "RSSMState":
        return RSSMState(
            deterministic=self.deterministic.to(device),
            stochastic=self.stochastic.to(device),
            mean=self.mean.to(device),
            std=self.std.to(device),
        )

    def distribution(self) -> Independent:
        base = Normal(self.mean, self.std)
        return Independent(base, 1)

    def rsample(self) -> torch.Tensor:
        return self.distribution().rsample()


@dataclass
class LatentStep:
    """Posterior/prior pair returned by representation learner updates."""

    posterior: LatentState
    prior: Optional[LatentState] = None

    def detach(self) -> "LatentStep":
        prior = self.prior.detach() if self.prior is not None else None
        return LatentStep(posterior=self.posterior.detach(), prior=prior)

    def to(self, device: torch.device | str) -> "LatentStep":
        prior = self.prior.to(device) if self.prior is not None else None
        return LatentStep(posterior=self.posterior.to(device), prior=prior)

    @property
    def posterior_dist(self) -> Independent:
        if not hasattr(self.posterior, "distribution"):
            raise AttributeError("LatentState implementation does not expose distribution().")
        return self.posterior.distribution()  # type: ignore[attr-defined]

    @property
    def prior_dist(self) -> Optional[Independent]:
        if self.prior is None:
            return None
        if not hasattr(self.prior, "distribution"):
            raise AttributeError("LatentState implementation does not expose distribution().")
        return self.prior.distribution()  # type: ignore[attr-defined]


@dataclass
class LatentSequence:
    """Temporal rollout of posterior/prior states."""

    posterior: LatentState
    prior: Optional[LatentState]
    last_posterior: LatentState

    def to(self, device: torch.device | str) -> "LatentSequence":
        prior = self.prior.to(device) if self.prior is not None else None
        return LatentSequence(
            posterior=self.posterior.to(device),
            prior=prior,
            last_posterior=self.last_posterior.to(device),
        )

    def detach(self) -> "LatentSequence":
        prior = self.prior.detach() if self.prior is not None else None
        return LatentSequence(
            posterior=self.posterior.detach(),
            prior=prior,
            last_posterior=self.last_posterior.detach(),
        )

    @property
    def posterior_dist(self) -> Independent:
        if not hasattr(self.posterior, "distribution"):
            raise AttributeError("LatentState implementation does not expose distribution().")
        return self.posterior.distribution()  # type: ignore[attr-defined]

    @property
    def prior_dist(self) -> Optional[Independent]:
        if self.prior is None:
            return None
        if not hasattr(self.prior, "distribution"):
            raise AttributeError("LatentState implementation does not expose distribution().")
        return self.prior.distribution()  # type: ignore[attr-defined]


__all__ = [
    "LatentState",
    "RSSMState",
    "LatentStep",
    "LatentSequence",
]
