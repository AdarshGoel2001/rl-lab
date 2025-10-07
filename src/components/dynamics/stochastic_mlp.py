"""
Stochastic MLP Dynamics Model

A stochastic dynamics model using multi-layer perceptrons.
Predicts next state as a Gaussian distribution.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .base import BaseDynamicsModel
from ...utils.registry import register_dynamics_model


@register_dynamics_model("stochastic_mlp")
class StochasticMLPDynamics(BaseDynamicsModel):
    """
    Stochastic dynamics model using MLP.

    Learns state transitions as: next_state ~ N(μ(state, action), σ(state, action))
    where μ and σ are neural networks.
    """

    def _build_model(self):
        """Build the stochastic dynamics model."""
        if self.state_dim is None:
            raise ValueError("StochasticMLPDynamics requires 'state_dim' in config")
        if self.action_dim is None:
            raise ValueError("StochasticMLPDynamics requires 'action_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)
        self.learn_std = self.config.get('learn_std', True)
        self.fixed_std = self.config.get('fixed_std', 1.0)
        self.min_std = self.config.get('min_std', 0.1)
        self.max_std = self.config.get('max_std', 10.0)

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

        # Build shared trunk (optional)
        use_shared_trunk = self.config.get('use_shared_trunk', True)

        if use_shared_trunk:
            # Shared trunk for both mean and std
            trunk_layers = []
            current_dim = input_dim

            # Use all but last hidden layer for trunk
            trunk_hidden_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else []

            for hidden_dim in trunk_hidden_dims:
                trunk_layers.append(nn.Linear(current_dim, hidden_dim))
                trunk_layers.append(self.activation_fn)

                if dropout_rate > 0:
                    trunk_layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

            self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
            trunk_output_dim = current_dim

            # Mean head
            mean_layers = []
            if len(hidden_dims) > 0:
                mean_layers.append(nn.Linear(trunk_output_dim, hidden_dims[-1]))
                mean_layers.append(self.activation_fn)
                mean_layers.append(nn.Linear(hidden_dims[-1], self.state_dim))
            else:
                mean_layers.append(nn.Linear(trunk_output_dim, self.state_dim))

            self.mean_network = nn.Sequential(*mean_layers)

            # Std head (if learned)
            if self.learn_std:
                std_layers = []
                if len(hidden_dims) > 0:
                    std_layers.append(nn.Linear(trunk_output_dim, hidden_dims[-1]))
                    std_layers.append(self.activation_fn)
                    std_layers.append(nn.Linear(hidden_dims[-1], self.state_dim))
                else:
                    std_layers.append(nn.Linear(trunk_output_dim, self.state_dim))

                self.std_network = nn.Sequential(*std_layers)
            else:
                self.std_network = None

        else:
            # Separate networks for mean and std
            self.trunk = None

            # Mean network
            mean_layers = []
            current_dim = input_dim

            for hidden_dim in hidden_dims:
                mean_layers.append(nn.Linear(current_dim, hidden_dim))
                mean_layers.append(self.activation_fn)

                if dropout_rate > 0:
                    mean_layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

            mean_layers.append(nn.Linear(current_dim, self.state_dim))
            self.mean_network = nn.Sequential(*mean_layers)

            # Std network (if learned)
            if self.learn_std:
                std_layers = []
                current_dim = input_dim

                for hidden_dim in hidden_dims:
                    std_layers.append(nn.Linear(current_dim, hidden_dim))
                    std_layers.append(self.activation_fn)
                    current_dim = hidden_dim

                std_layers.append(nn.Linear(current_dim, self.state_dim))
                self.std_network = nn.Sequential(*std_layers)
            else:
                self.std_network = None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Normal:
        """
        Predict next state distribution given current state and action.

        Args:
            state: Current state tensor, shape (batch_size, state_dim)
            action: Action tensor, shape (batch_size, action_dim)

        Returns:
            Normal distribution over next states
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)

        # Pass through trunk if using shared architecture
        if self.trunk is not None:
            features = self.trunk(state_action)
        else:
            features = state_action

        # Predict mean
        next_state_mean = self.mean_network(features)

        # Predict or use fixed std
        if self.learn_std:
            next_state_log_std = self.std_network(features)
            # Apply softplus to ensure positivity, then clamp
            next_state_std = F.softplus(next_state_log_std)
            next_state_std = torch.clamp(next_state_std, self.min_std, self.max_std)
        else:
            next_state_std = torch.full_like(next_state_mean, self.fixed_std)

        return Normal(next_state_mean, next_state_std)

    @property
    def is_deterministic(self) -> bool:
        """This is a stochastic dynamics model."""
        return False

    def dynamics_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute dynamics learning loss.

        For stochastic dynamics, we use negative log-likelihood loss.

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

        # Also compute MSE for monitoring
        mse_loss = F.mse_loss(predicted_dist.mean, next_states)

        # Compute mean predicted std for monitoring
        mean_std = predicted_dist.stddev.mean()

        return {
            'dynamics_loss': nll_loss,
            'dynamics_nll': nll_loss,
            'dynamics_mse': mse_loss,
            'dynamics_mean_std': mean_std
        }

    def get_dynamics_info(self) -> Dict[str, Any]:
        """Get information about this dynamics model."""
        info = super().get_dynamics_info()
        info.update({
            'learn_std': self.learn_std,
            'fixed_std': self.fixed_std if not self.learn_std else None,
            'min_std': self.min_std,
            'max_std': self.max_std,
            'hidden_dims': self.config.get('hidden_dims', [256, 256]),
            'use_shared_trunk': self.config.get('use_shared_trunk', True)
        })
        return info
