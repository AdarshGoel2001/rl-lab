"""
Deterministic MLP Dynamics Model

A simple deterministic dynamics model using multi-layer perceptrons.
Predicts next state as a deterministic function of current state and action.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.distributions import Normal

from .base import BaseDynamicsModel
from ...utils.registry import register_dynamics_model


@register_dynamics_model("deterministic_mlp")
class DeterministicMLPDynamics(BaseDynamicsModel):
    """
    Deterministic dynamics model using MLP.

    Learns state transitions as: next_state = f(state, action)
    where f is a neural network.
    """

    def _build_model(self):
        """Build the deterministic dynamics model."""
        if self.state_dim is None:
            raise ValueError("DeterministicMLPDynamics requires 'state_dim' in config")
        if self.action_dim is None:
            raise ValueError("DeterministicMLPDynamics requires 'action_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)

        # Activation function mapping
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation_fn = activation_map[activation]

        # Input dimension is state + action
        input_dim = self.state_dim + self.action_dim

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_fn)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Output layer (predict next state)
        layers.append(nn.Linear(current_dim, self.state_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Normal:
        """
        Predict next state distribution given current state and action.

        Args:
            state: Current state tensor, shape (batch_size, state_dim)
            action: Action tensor, shape (batch_size, action_dim)

        Returns:
            Normal distribution over next states (deterministic mean, small variance)
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)

        # Predict next state
        next_state_mean = self.network(state_action)

        # For deterministic dynamics, use small fixed variance
        variance = float(self.config.get('output_variance', 1e-4))
        next_state_std = torch.full_like(next_state_mean, variance ** 0.5)

        return Normal(next_state_mean, next_state_std)

    def predict_sequence(self,
                        initial_state: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout deterministic sequence of predictions.

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
            # Predict next state (use mean for deterministic rollout)
            next_state_dist = self.forward(current_state, actions[:, t])
            current_state = next_state_dist.mean
            states.append(current_state)

        return torch.stack(states, dim=1)

    @property
    def is_deterministic(self) -> bool:
        """This is a deterministic dynamics model."""
        return True

    def dynamics_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute dynamics learning loss.

        For deterministic dynamics, we use MSE loss on the predicted means.

        Args:
            states: Current states, shape (batch_size, state_dim)
            actions: Actions, shape (batch_size, action_dim)
            next_states: True next states, shape (batch_size, state_dim)

        Returns:
            Dictionary of loss components
        """
        predicted_dist = self.forward(states, actions)
        predicted_next_states = predicted_dist.mean

        # MSE loss between predicted and true next states
        mse_loss = nn.functional.mse_loss(predicted_next_states, next_states)

        return {
            'dynamics_loss': mse_loss,
            'dynamics_mse': mse_loss
        }
