"""Dreamer value critic."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


def _as_plain_dict(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if config is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return dict(OmegaConf.to_container(config, resolve=True))
    except ImportError:
        pass
    return dict(config)


def _hidden_dims(config: Mapping[str, Any]) -> list[int]:
    dims = config.get("hidden_dims")
    if dims is not None:
        return [int(dim) for dim in dims]
    return [int(config.get("hidden_dim", 300))] * int(config.get("num_layers", 3))


class DreamerCritic(nn.Module):
    """MLP value model over flattened RSSM latent features."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = _as_plain_dict(config)
        merged.update(kwargs)

        input_dim = merged.get("input_dim", merged.get("latent_dim"))
        if input_dim is None:
            raise ValueError("DreamerCritic requires input_dim or latent_dim.")
        self.input_dim = int(input_dim)

        layers: list[nn.Module] = []
        last_dim = self.input_dim
        for hidden_dim in _hidden_dims(merged):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

        device = merged.get("device")
        if device is not None:
            self.to(torch.device(device))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


__all__ = ["DreamerCritic"]
