"""
MLP Encoder

Multi-layer perceptron encoder reimplemented from existing MLP network.
Preserves all the functionality and initialization from the original design.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np

from .base import BaseEncoder
from ...utils.registry import register_encoder


@register_encoder("mlp")
class MLPEncoder(BaseEncoder):
    """
    MLP encoder based on the existing MLP network design.

    Preserves all functionality from the original MLP network including:
    - Configurable depth and width
    - Multiple activation functions
    - Batch normalization and dropout support
    - Automatic input/output dimension handling
    """

    def _build_encoder(self):
        """Build the MLP encoder architecture using original network logic."""
        input_dim = self.config.get('input_dim')
        if input_dim is None:
            raise ValueError("MLP encoder requires 'input_dim' in config")

        # Handle tuple input dimensions (flatten to single dimension) - from original
        if isinstance(input_dim, (tuple, list)):
            input_size = int(np.prod(input_dim))
        else:
            input_size = int(input_dim)

        # Use original defaults
        hidden_dims = self.config.get('hidden_dims', [64, 64])
        activation = self.config.get('activation', 'relu')
        batch_norm = self.config.get('batch_norm', False)
        dropout = self.config.get('dropout', 0.0)
        layer_norm = self.config.get('layer_norm', False)

        layers = []
        prev_dim = input_size

        # Hidden layers - using original logic exactly
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

        self.layers = nn.Sequential(*layers)
        self._output_dim = prev_dim

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

        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}")

        return activation_map[activation_name]

    def _initialize_weights(self):
        """Initialize network weights using original initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass using original logic.

        Args:
            observations: Input tensor or dict

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        if isinstance(observations, dict):
            # Handle dict observations - concatenate all values
            if 'observation' in observations:
                obs = observations['observation']
            else:
                obs_list = [v.view(v.shape[0], -1) for v in observations.values()]
                obs = torch.cat(obs_list, dim=-1)
        else:
            obs = observations

        # Flatten input if needed - from original
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)

        return self.layers(obs)

    @property
    def output_dim(self) -> int:
        """Get the dimensionality of the encoder's output features."""
        return self._output_dim