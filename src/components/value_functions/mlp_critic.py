"""
MLP Value Function (Critic)

Value function based on the original CriticMLP design.
Preserves all functionality from the original network.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseValueFunction


class CriticMLPValueFunction(BaseValueFunction):
    """
    MLP-based value function based on original CriticMLP.

    Preserves the original design with linear output activation
    and single value output for state value estimation.
    """

    def _build_value_function(self):
        """Build value function using original CriticMLP logic."""
        if self.representation_dim is None:
            raise ValueError("CriticMLPValueFunction requires 'representation_dim' in config")

        # Original defaults from CriticMLP
        hidden_dims = self.config.get('hidden_dims', [64, 64])
        activation = self.config.get('activation', 'relu')
        batch_norm = self.config.get('batch_norm', False)
        dropout = self.config.get('dropout', 0.0)
        layer_norm = self.config.get('layer_norm', False)

        layers = []
        prev_dim = self.representation_dim

        # Hidden layers - using original MLP logic exactly
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Layer normalization
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            activation_fn = self._get_activation_function(activation)
            layers.append(activation_fn)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Store hidden layers
        self.layers = nn.Sequential(*layers)

        # Output layer - single value for state value function
        self.output_layer = nn.Linear(prev_dim, 1)

        # Linear output activation (from original CriticMLP default)
        self.output_activation = nn.Identity()

        # Initialize weights using original method
        self._initialize_weights()

    def _get_activation_function(self, activation_name: str):
        """Get activation function - from original BaseNetwork."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'linear': nn.Identity()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def _initialize_weights(self):
        """Initialize weights using original initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self,
                representation: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using original MLP logic.

        Args:
            representation: State representation, shape (batch_size, representation_dim)
            action: Ignored for state-value function

        Returns:
            Value estimates, shape (batch_size, 1)
        """
        # Forward through hidden layers - from original
        x = self.layers(representation)

        # Output layer
        x = self.output_layer(x)

        # Output activation (Identity for linear output)
        x = self.output_activation(x)

        return x

    def value_loss(self,
                  representations: torch.Tensor,
                  targets: torch.Tensor,
                  actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute value function learning loss.

        Args:
            representations: State representations
            targets: Target values (returns, TD targets, etc.)
            actions: Ignored for state-value function

        Returns:
            Dictionary of loss components
        """
        values = self.forward(representations, actions)
        values = values.squeeze(-1)  # Remove last dimension (batch_size, 1) -> (batch_size,)

        # MSE loss
        mse_loss = nn.functional.mse_loss(values, targets)

        # Optional: Huber loss for robustness
        use_huber = self.config.get('use_huber_loss', False)
        if use_huber:
            huber_delta = self.config.get('huber_delta', 1.0)
            huber_loss = nn.functional.smooth_l1_loss(values, targets, beta=huber_delta)
            return {
                'value_loss': huber_loss,
                'value_mse': mse_loss,
                'value_huber': huber_loss
            }
        else:
            return {
                'value_loss': mse_loss,
                'value_mse': mse_loss
            }

    @property
    def is_q_function(self) -> bool:
        """This is a V-function (state values), not Q-function."""
        return False


# Also create a simple MLP critic for backwards compatibility
class MLPCritic(BaseValueFunction):
    """Simple MLP critic (alias for CriticMLPValueFunction)."""

    def _build_value_function(self):
        """Build simple MLP critic."""
        if self.representation_dim is None:
            raise ValueError("MLPCritic requires 'representation_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')

        layers = []
        current_dim = self.representation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self._get_activation_function(activation))
            current_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation_function(self, activation_name: str):
        """Get activation function."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, representation: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.network(representation)