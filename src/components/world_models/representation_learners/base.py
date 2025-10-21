"""Base interfaces and utilities for representation learners."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


@dataclass
class RSSMState:
    """Deterministic/stochastic latent container used by Dreamer-style models."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

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

    posterior: RSSMState
    prior: Optional[RSSMState] = None

    def detach(self) -> "LatentStep":
        prior = self.prior.detach() if self.prior is not None else None
        return LatentStep(posterior=self.posterior.detach(), prior=prior)

    def to(self, device: torch.device | str) -> "LatentStep":
        prior = self.prior.to(device) if self.prior is not None else None
        return LatentStep(posterior=self.posterior.to(device), prior=prior)

    @property
    def posterior_dist(self) -> Independent:
        return self.posterior.distribution()

    @property
    def prior_dist(self) -> Optional[Independent]:
        if self.prior is None:
            return None
        return self.prior.distribution()


@dataclass
class LatentSequence:
    """Temporal rollout of posterior/prior states."""

    posterior: RSSMState
    prior: Optional[RSSMState]
    last_posterior: RSSMState

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
        return self.posterior.distribution()

    @property
    def prior_dist(self) -> Optional[Independent]:
        if self.prior is None:
            return None
        return self.prior.distribution()


class BaseRepresentationLearner(nn.Module, metaclass=abc.ABCMeta):
    """Common contract for modules that maintain latent state across time."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.device = torch.device(self.config.get("device", "cpu"))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def representation_dim(self) -> int:
        """Return dimensionality of the latent representation."""

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> RSSMState:
        """Return zero-initialised state for the requested batch size."""

    @abc.abstractmethod
    def reset_state(
        self,
        batch_size: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset cached online state buffers."""

    @abc.abstractmethod
    def get_state(self) -> RSSMState:
        """Return cached online posterior state."""

    @abc.abstractmethod
    def set_state(self, state: RSSMState) -> None:
        """Load cached online posterior state."""

    # ------------------------------------------------------------------
    # Online usage
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def observe(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        *,
        detach_posteriors: bool = False,
    ) -> LatentStep:
        """Consume encoder features and update cached state (online usage)."""

    # ------------------------------------------------------------------
    # Batched/offline usage
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def observe_batch(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[LatentStep, RSSMState]:
        """Process a batch step without mutating the cached online state."""

    @abc.abstractmethod
    def observe_sequence(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
    ) -> LatentSequence:
        """Encode an entire sequence of features/actions and return stacked latents."""

    # ------------------------------------------------------------------
    # Imagination / rollout
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def imagine_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> RSSMState:
        """Roll latent state forward without real observations."""

    # ------------------------------------------------------------------
    # Device plumbing
    # ------------------------------------------------------------------
    def to(self, *args: Any, **kwargs: Any) -> "BaseRepresentationLearner":  # type: ignore[override]
        super().to(*args, **kwargs)
        target_device: Optional[torch.device | str] = kwargs.get("device")
        if target_device is None and args:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    target_device = arg
                    break
        if target_device is not None:
            self.device = torch.device(target_device)
        return self


__all__ = ["BaseRepresentationLearner", "RSSMState", "LatentStep", "LatentSequence"]
