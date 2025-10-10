"""Recurrent state-space dynamics model for Dreamer-style agents."""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution, constraints

from .base import BaseDynamicsModel
from ....utils.registry import register_dynamics_model
from ..rssm import RSSMState


class RSSMTransitionDistribution(Distribution):
    """Distribution over next RSSM latent state."""

    arg_constraints = {}
    support = Normal.support  # type: ignore[attr-defined]

    has_rsample = True

    def __init__(
        self,
        deterministic: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        if deterministic.shape[:-1] != mean.shape[:-1]:
            raise ValueError("Deterministic and stochastic tensors must share batch shape")
        batch_shape = mean.shape[:-1]
        event_dim = deterministic.shape[-1] + mean.shape[-1]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size([event_dim]),
            validate_args=False,
        )
        self.deterministic = deterministic
        self._normal = Normal(mean, std)
        self._det_dim = deterministic.shape[-1]
        self._sto_dim = mean.shape[-1]

    @property
    def mean(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self._normal.mean], dim=-1)

    @property
    def stddev(self) -> torch.Tensor:
        zeros = torch.zeros_like(self.deterministic)
        return torch.cat([zeros, self._normal.stddev], dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        stochastic = self._normal.sample(sample_shape)
        deterministic = self._expand_deterministic(sample_shape)
        return torch.cat([deterministic, stochastic], dim=-1)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        stochastic = self._normal.rsample(sample_shape)
        deterministic = self._expand_deterministic(sample_shape)
        return torch.cat([deterministic, stochastic], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        det_dim = self._det_dim
        _, stochastic = torch.split(value, [det_dim, self._sto_dim], dim=-1)
        log_prob = self._normal.log_prob(stochastic)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        return self._normal.entropy().sum(dim=-1)

    def _expand_deterministic(self, sample_shape: torch.Size) -> torch.Tensor:
        if sample_shape == torch.Size():
            return self.deterministic
        det = self.deterministic
        expand_shape = sample_shape + det.shape
        return det.unsqueeze(dim=0).expand(expand_shape)


def _build_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
    )


@register_dynamics_model("rssm")
class RSSMDynamicsModel(BaseDynamicsModel):
    """Dreamer-style RSSM prior dynamics with persistent deterministic state."""

    def _build_model(self) -> None:
        if self.state_dim is None:
            raise ValueError("RSSMDynamicsModel requires 'state_dim' in config")
        if self.action_dim is None:
            raise ValueError("RSSMDynamicsModel requires 'action_dim' in config")

        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.deterministic_dim = self.config.get("deterministic_dim")
        self.stochastic_dim = self.config.get("stochastic_dim")

        if self.deterministic_dim is None and self.stochastic_dim is None:
            raise ValueError("RSSMDynamicsModel requires 'deterministic_dim' or 'stochastic_dim'")

        if self.deterministic_dim is None:
            self.deterministic_dim = self.state_dim - self.stochastic_dim
        if self.stochastic_dim is None:
            self.stochastic_dim = self.state_dim - self.deterministic_dim

        if self.deterministic_dim + self.stochastic_dim != self.state_dim:
            raise ValueError(
                "state_dim must equal deterministic_dim + stochastic_dim ("
                f"got {self.state_dim} vs {self.deterministic_dim} + {self.stochastic_dim})"
            )

        self.min_std = self.config.get("min_std", 0.1)

        transition_input_dim = self.stochastic_dim + self.action_dim
        self.transition = nn.GRUCell(transition_input_dim, self.deterministic_dim)

        self.prior_net = _build_mlp(self.deterministic_dim, self.hidden_dim)
        self.prior_mean_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)
        self.prior_std_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)

        self._initialize()
        self._last_state: Optional[RSSMState] = None

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GRUCell):
                for name, param in module.named_parameters():
                    if param.ndim >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        device = device or self.device
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return RSSMState(deterministic=deterministic, stochastic=stochastic)

    def forward(
        self,
        state: Union[torch.Tensor, RSSMState],
        action: torch.Tensor,
    ) -> RSSMTransitionDistribution:
        rssm_state = self._ensure_state(state)

        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] != self.action_dim:
            raise ValueError("Action dimension mismatch for RSSMDynamicsModel")
        if action.shape[0] != rssm_state.batch_size:
            raise ValueError("Action batch size mismatch for RSSMDynamicsModel")

        transition_input = torch.cat([rssm_state.stochastic, action], dim=-1)
        deterministic = self.transition(transition_input, rssm_state.deterministic)

        hidden = self.prior_net(deterministic)
        mean = self.prior_mean_layer(hidden)
        std = F.softplus(self.prior_std_layer(hidden)) + self.min_std

        distribution = RSSMTransitionDistribution(deterministic=deterministic, mean=mean, std=std)
        self._last_state = RSSMState(
            deterministic=deterministic,
            stochastic=mean,
            prior_mean=mean,
            prior_std=std,
        )
        return distribution

    def dynamics_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        predicted = self.forward(states, actions)
        log_prob = predicted.log_prob(next_states)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        nll = -log_prob.mean()
        return {
            "dynamics_loss": nll,
            "dynamics_nll": nll,
        }

    def _ensure_state(self, state: Union[torch.Tensor, RSSMState]) -> RSSMState:
        if isinstance(state, RSSMState):
            return state
        if state.dim() != 2:
            raise ValueError("RSSMDynamicsModel expects 2D state tensors")
        if state.shape[-1] != self.state_dim:
            raise ValueError("State tensor has incorrect dimension for RSSM dynamics")
        det, sto = torch.split(state, [self.deterministic_dim, self.stochastic_dim], dim=-1)
        return RSSMState(deterministic=det, stochastic=sto)
