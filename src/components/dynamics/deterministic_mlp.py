"""Deterministic MLP dynamics module for TD-MPC."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

class DeterministicMLPDynamics(nn.Module):
    """Predicts next latent deterministically via a feedforward network."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] | Iterable[int] = (512, 512),
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = {"latent_dim": latent_dim, "action_dim": action_dim, "hidden_dims": list(hidden_dims), **kwargs}
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.activation_name = activation
        self.net = self._build_network(hidden_dims)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        deterministic: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        state_tensor = state
        action_tensor = action
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if state_tensor.shape[0] != action_tensor.shape[0]:
            raise ValueError("State and action batch dimensions must match for deterministic dynamics.")
        inputs = torch.cat([state_tensor, action_tensor], dim=-1)
        return self.net(inputs)

    def _build_network(self, hidden_dims: Sequence[int] | Iterable[int]) -> nn.Module:
        layers: list[nn.Module] = []
        input_dim = self.latent_dim + self.action_dim
        activation = self._activation()

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.latent_dim))
        return nn.Sequential(*layers)

    def _activation(self):
        name = self.activation_name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "tanh":
            return nn.Tanh
        if name in {"gelu"}:
            return nn.GELU
        if name in {"swish", "silu"}:
            return nn.SiLU
        return nn.ELU
