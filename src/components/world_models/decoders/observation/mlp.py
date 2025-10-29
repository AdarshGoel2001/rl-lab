"""Feedforward observation decoders for vector observations."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseObservationDecoder


class MLPObservationDecoder(BaseObservationDecoder):
    """Projects latent vectors back to flat observation space via MLP."""

    def _build_decoder(self) -> None:
        hidden_dims = self.config.get("hidden_dims", [128, 128])
        activation = self.config.get("activation", "elu")
        output_activation = self.config.get("output_activation")

        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
        }
        hidden_activation = activation_map.get(activation, nn.ELU())
        final_activation = (
            activation_map.get(output_activation)
            if output_activation in activation_map
            else None
        )

        layers = []
        current_dim = self.representation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(hidden_activation)
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.output_dim))
        if final_activation is not None:
            layers.append(final_activation)

        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        reconstruction = self.network(latent)
        if self.output_shape is not None:
            reconstruction = reconstruction.view(latent.shape[0], *self.output_shape)
        return reconstruction


__all__ = ["MLPObservationDecoder"]

