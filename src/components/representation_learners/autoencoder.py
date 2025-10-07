"""MLP autoencoder-based representation learner."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRepresentationLearner
from ...utils.registry import register_representation_learner


def _build_mlp(input_dim: int,
               hidden_dims: List[int],
               output_dim: int,
               activation: str,
               dropout: float = 0.0) -> nn.Sequential:
    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'linear': nn.Identity(),
    }
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation '{activation}' for autoencoder")

    layers = []
    prev_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(activation_map[activation])
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


@register_representation_learner("mlp_autoencoder")
class MLPAutoencoderRepresentationLearner(BaseRepresentationLearner):
    """Representation learner that trains an MLP autoencoder on encoder features."""

    def _build_learner(self) -> None:
        feature_dim = self.config.get('input_dim')
        if feature_dim is None:
            raise ValueError("MLP autoencoder requires 'input_dim' in config")

        latent_dim = self.config.get('representation_dim', feature_dim)
        hidden_dims = self.config.get('hidden_dims', [64])
        activation = self.config.get('activation', 'relu')
        dropout = float(self.config.get('dropout', 0.0) or 0.0)

        self.encoder_net = _build_mlp(feature_dim, hidden_dims, latent_dim, activation, dropout)
        decoder_hidden = list(reversed(hidden_dims))
        self.decoder_net = _build_mlp(latent_dim, decoder_hidden, feature_dim, activation, dropout)

        self._feature_dim = feature_dim
        self._representation_dim = latent_dim
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(features)

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(representation)

    def representation_loss(self,
                            features: torch.Tensor,
                            **kwargs: Any) -> Dict[str, torch.Tensor]:
        # Reconstruction handled in paradigm; return zero placeholder here.
        return {
            'representation_loss': torch.tensor(0.0, device=features.device)
        }

    @property
    def representation_dim(self) -> int:
        return self._representation_dim
