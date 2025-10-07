"""MLP decoder mapping latent state back to observation space."""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from .base import BaseDecoder
from ...utils.registry import register_decoder


@register_decoder("mlp")
class MLPDecoder(BaseDecoder):
    """Simple feed-forward decoder for low-dimensional observations."""

    def _build_decoder(self) -> None:
        latent_dim = self.config.get('latent_dim')
        if latent_dim is None:
            raise ValueError("MLPDecoder requires 'latent_dim' in config")

        output_dim = self.config.get('output_dim')
        if output_dim is None:
            raise ValueError("MLPDecoder requires 'output_dim' in config")

        if isinstance(output_dim, (tuple, list)):
            self._output_shape = tuple(int(x) for x in output_dim)
            final_dim = int(np.prod(self._output_shape))
        else:
            self._output_shape = (int(output_dim),)
            final_dim = int(output_dim)

        hidden_dims = self.config.get('hidden_dims', [64, 64])
        activation_name = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)

        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'linear': nn.Identity(),
        }
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation '{activation_name}' for MLPDecoder")

        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_map[activation_name])
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, final_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        recon = self.network(latent)
        if len(self._output_shape) > 1:
            return recon.view(-1, *self._output_shape)
        return recon

