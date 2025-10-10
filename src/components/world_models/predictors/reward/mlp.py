"""MLP-based reward prediction heads."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .....utils.registry import register_reward_predictor
from .base import BaseRewardPredictor


@register_reward_predictor("mlp")
class MLPRewardPredictor(BaseRewardPredictor):
    """Predicts scalar rewards from latent states using an MLP."""

    def _build_model(self) -> None:
        hidden_dims = self.config.get("hidden_dims", [128, 128])
        activation = self.config.get("activation", "elu")

        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        activation_fn = activation_map.get(activation, nn.ReLU())

        layers = []
        current_dim = self.representation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn)
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).squeeze(-1)


__all__ = ["MLPRewardPredictor"]

