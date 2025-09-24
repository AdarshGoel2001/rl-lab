"""
Base Dynamics Model Interface

This module defines the abstract base class for dynamics models.
Dynamics models predict how the world state evolves given current state and actions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
from torch.distributions import Distribution


class BaseDynamicsModel(nn.Module, ABC):
    """
    Abstract base class for all dynamics models.

    Dynamics models predict state transitions: (state, action) -> next_state_distribution.
    They are essential components for model-based RL and world model paradigms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dynamics model with configuration.

        Args:
            config: Dictionary containing hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.state_dim = config.get('state_dim')
        self.action_dim = config.get('action_dim')
        self._build_model()

    @abstractmethod
    def _build_model(self):
        """
        Build the dynamics model architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Distribution:
        """
        Predict next state distribution given current state and action.

        Args:
            state: Current state tensor, shape (batch_size, state_dim)
            action: Action tensor, shape (batch_size, action_dim)

        Returns:
            Distribution over next states
        """
        pass

    def predict_sequence(self,
                        initial_state: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout sequence of predictions given initial state and action sequence.

        Args:
            initial_state: Initial state, shape (batch_size, state_dim)
            actions: Action sequence, shape (batch_size, horizon, action_dim)

        Returns:
            Predicted state sequence, shape (batch_size, horizon, state_dim)
        """
        batch_size, horizon, _ = actions.shape
        states = []
        current_state = initial_state

        for t in range(horizon):
            next_state_dist = self.forward(current_state, actions[:, t])
            current_state = next_state_dist.mean  # Use mean prediction
            states.append(current_state)

        return torch.stack(states, dim=1)

    def dynamics_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute dynamics learning loss.

        Args:
            states: Current states, shape (batch_size, state_dim)
            actions: Actions, shape (batch_size, action_dim)
            next_states: True next states, shape (batch_size, state_dim)

        Returns:
            Dictionary of loss components
        """
        predicted_dist = self.forward(states, actions)

        # Negative log likelihood loss
        nll_loss = -predicted_dist.log_prob(next_states).mean()

        return {
            'dynamics_loss': nll_loss,
            'dynamics_nll': nll_loss
        }

    @property
    def is_deterministic(self) -> bool:
        """
        Whether this dynamics model is deterministic.

        Override in subclasses. Default is stochastic.

        Returns:
            False by default (stochastic dynamics)
        """
        return False

    def get_dynamics_info(self) -> Dict[str, Any]:
        """
        Get information about this dynamics model.

        Returns:
            Dictionary with dynamics model information
        """
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'is_deterministic': self.is_deterministic,
            'model_type': self.__class__.__name__,
            'device': str(self.device)
        }