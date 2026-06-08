"""Dreamer continuous-control actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import AffineTransform, Independent, Normal, TanhTransform, TransformedDistribution


@dataclass
class DreamerActorOutput:
    """Actor distribution parameters and transformed action distribution."""

    mean: torch.Tensor
    std: torch.Tensor
    distribution: TransformedDistribution
    raw_mean: torch.Tensor

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


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


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(tensor.expm1().clamp_min(1e-8).log())


class DreamerActor(nn.Module):
    """Tanh-squashed diagonal Gaussian actor for Dreamer-style control."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = _as_plain_dict(config)
        merged.update(kwargs)

        input_dim = merged.get("input_dim", merged.get("latent_dim"))
        if input_dim is None:
            raise ValueError("DreamerActor requires input_dim or latent_dim.")
        self.input_dim = int(input_dim)
        self.action_dim = int(merged["action_dim"])
        self.min_std = float(merged.get("min_std", 1e-4))
        self.init_std = float(merged.get("init_std", 5.0))
        self.mean_scale = float(merged.get("mean_scale", 5.0))

        layers: list[nn.Module] = []
        last_dim = self.input_dim
        for hidden_dim in _hidden_dims(merged):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last_dim, self.action_dim)
        self.std_head = nn.Linear(last_dim, self.action_dim)

        nn.init.constant_(self.std_head.bias, _inverse_softplus(max(self.init_std - self.min_std, 1e-4)))

        action_low = torch.as_tensor(merged.get("action_low", [-1.0] * self.action_dim), dtype=torch.float32)
        action_high = torch.as_tensor(merged.get("action_high", [1.0] * self.action_dim), dtype=torch.float32)
        if action_low.numel() == 1:
            action_low = action_low.repeat(self.action_dim)
        if action_high.numel() == 1:
            action_high = action_high.repeat(self.action_dim)
        if action_low.shape != (self.action_dim,) or action_high.shape != (self.action_dim,):
            raise ValueError("action_low and action_high must match action_dim.")

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        device = merged.get("device")
        if device is not None:
            self.to(torch.device(device))

    def forward(self, latent: torch.Tensor) -> DreamerActorOutput:
        features = self.trunk(latent)
        raw_mean = self.mean_head(features)
        raw_std = self.std_head(features)

        mean = self.mean_scale * torch.tanh(raw_mean / self.mean_scale)
        std = F.softplus(raw_std) + self.min_std

        base = Independent(Normal(mean, std), 1)
        scale = (self.action_high - self.action_low) / 2.0
        loc = (self.action_high + self.action_low) / 2.0
        distribution = TransformedDistribution(
            base,
            [
                TanhTransform(cache_size=1),
                AffineTransform(loc=loc, scale=scale),
            ],
        )
        bounded_mean = loc + scale * torch.tanh(mean)
        return DreamerActorOutput(mean=bounded_mean, std=std, distribution=distribution, raw_mean=mean)

    def _squash_raw_action(self, raw_action: Tensor) -> Tensor:
        scale = (self.action_high - self.action_low) / 2.0
        loc = (self.action_high + self.action_low) / 2.0
        return loc + scale * torch.tanh(raw_action)

    def _raw_log_prob(self, raw_action: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        normal = Normal(mean, std)
        log_prob = normal.log_prob(raw_action)
        log_two = torch.log(raw_action.new_tensor(2.0))
        log_tanh_det = 2.0 * (log_two - raw_action - F.softplus(-2.0 * raw_action))
        log_scale = torch.log(((self.action_high - self.action_low) / 2.0).clamp_min(1e-8))
        return (log_prob - log_tanh_det - log_scale).sum(dim=-1, keepdim=True)

    def act(
        self,
        latent: torch.Tensor,
        deterministic: bool = False,
        return_log_prob: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(latent)
        if deterministic:
            raw_action = outputs.raw_mean
        else:
            raw_action = Normal(outputs.raw_mean, outputs.std).rsample()
        action = self._squash_raw_action(raw_action)
        action = action.clamp(self.action_low, self.action_high)

        if not return_log_prob:
            return action

        log_prob = self._raw_log_prob(raw_action, outputs.raw_mean, outputs.std)
        return action, log_prob


__all__ = ["DreamerActor", "DreamerActorOutput"]
