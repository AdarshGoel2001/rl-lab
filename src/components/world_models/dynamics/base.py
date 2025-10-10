"""Base dynamics interface for world-model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributions import Distribution


class BaseDynamicsModel(nn.Module, ABC):
    """Abstract base class for dynamics modules predicting latent transitions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.state_dim = config.get("state_dim")
        self.action_dim = config.get("action_dim")
        self._build_model()

    @abstractmethod
    def _build_model(self) -> None:
        """Construct neural layers for the dynamics model."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Distribution:
        """Predict distribution over next latent state given current state/action."""
        raise NotImplementedError

    def predict_sequence(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out mean predictions over an action sequence."""
        current_state = initial_state
        states = []
        for t in range(actions.shape[1]):
            next_state_dist = self.forward(current_state, actions[:, t])
            current_state = next_state_dist.mean
            states.append(current_state)
        return torch.stack(states, dim=1)

    def dynamics_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Default negative log-likelihood loss on next-state predictions."""
        predicted = self.forward(states, actions)
        log_prob = predicted.log_prob(next_states)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        nll = -log_prob.mean()
        return {
            "dynamics_loss": nll,
            "dynamics_nll": nll,
        }

    @property
    def is_deterministic(self) -> bool:
        """Whether the dynamics are deterministic (override if needed)."""
        return False

    def get_dynamics_info(self) -> Dict[str, Any]:
        """Return summary information for logging/introspection."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "is_deterministic": self.is_deterministic,
            "model_type": self.__class__.__name__,
            "device": str(self.device),
        }
