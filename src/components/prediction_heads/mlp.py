"""Simple MLP prediction head for rewards, continuations, and values."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Apply an MLP over the last dimension while preserving leading dims."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = dict(config or {})
        if kwargs:
            merged.update(kwargs)

        self.config = merged
        self.input_dim = int(merged["input_dim"])
        self.output_dim = int(merged.get("output_dim", 1))
        self.hidden_dim = int(merged.get("hidden_dim", 128))
        self.num_layers = int(merged.get("num_layers", 2))
        self.activation_name = str(merged.get("activation", "elu"))

        layers: list[nn.Module] = []
        current_dim = self.input_dim
        for _ in range(max(self.num_layers - 1, 0)):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            layers.append(self._activation())
            current_dim = self.hidden_dim
        layers.append(nn.Linear(current_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, inputs.shape[-1])
        outputs = self.net(flat)
        return outputs.reshape(*leading_shape, self.output_dim)

    def _activation(self) -> nn.Module:
        name = self.activation_name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name in {"silu", "swish"}:
            return nn.SiLU()
        if name == "tanh":
            return nn.Tanh()
        return nn.ELU()


__all__ = ["MLPHead"]
