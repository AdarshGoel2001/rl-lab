"""Recurrent state-space model (RSSM) representation learner."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import register_representation_learner

from .base import BaseRepresentationLearner, LatentSequence, LatentStep, RSSMState


def _build_activation(name: str) -> nn.Module:
    name = (name or "elu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    if name in {"swish", "silu"}:
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    return nn.ELU()


def _split_stats(stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std_param = torch.chunk(stats, 2, dim=-1)
    return mean, std_param


@register_representation_learner("rssm")
class RSSMRepresentationLearner(BaseRepresentationLearner):
    """Dreamer-style RSSM with stochastic + deterministic latent components."""

    def __init__(self, config: Dict[str, int | float | bool]) -> None:
        super().__init__(config)

        self.stochastic_dim = int(self.config.get("stochastic_dim", self.config.get("latent_dim", 32)))
        self.deterministic_dim = int(self.config.get("deterministic_dim", self.config.get("hidden_dim", 200)))
        self.hidden_dim = int(self.config.get("hidden_dim", max(self.deterministic_dim, 200)))
        self.feature_dim = int(self.config.get("feature_dim", self.config.get("embedding_dim", self.stochastic_dim)))
        self.action_dim = int(self.config.get("action_dim", 0))
        self.discrete_actions = bool(self.config.get("discrete_actions", False))

        self.min_std = float(self.config.get("min_std", 0.1))
        self.max_std = float(self.config.get("max_std", 0.0)) or None
        self.std_transform = str(self.config.get("std_transform", "softplus"))
        self.sample_prior = bool(self.config.get("sample_prior", True))
        self.sample_posterior = bool(self.config.get("sample_posterior", True))

        activation = _build_activation(str(self.config.get("activation", "elu")))

        input_dim = self.stochastic_dim + self.action_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            activation,
        )

        self.gru = nn.GRUCell(self.hidden_dim, self.deterministic_dim)

        self.prior_mlp = nn.Sequential(
            nn.Linear(self.deterministic_dim, self.hidden_dim),
            _build_activation(str(self.config.get("prior_activation", "elu"))),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim),
        )

        self.posterior_mlp = nn.Sequential(
            nn.Linear(self.deterministic_dim + self.feature_dim, self.hidden_dim),
            _build_activation(str(self.config.get("posterior_activation", "elu"))),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim),
        )

        self._state: Optional[RSSMState] = None
        self._prior_state: Optional[RSSMState] = None

        self.to(self.device)

    # ------------------------------------------------------------------
    # BaseRepresentationLearner API
    # ------------------------------------------------------------------
    @property
    def representation_dim(self) -> int:
        return self.deterministic_dim + self.stochastic_dim

    def initial_state(self, batch_size: int) -> RSSMState:
        zeros_det = torch.zeros(batch_size, self.deterministic_dim, device=self.device)
        zeros_stoch = torch.zeros(batch_size, self.stochastic_dim, device=self.device)
        std = torch.ones_like(zeros_stoch) * self.min_std
        return RSSMState(
            deterministic=zeros_det,
            stochastic=zeros_stoch,
            mean=zeros_stoch.clone(),
            std=std,
        )

    def reset_state(
        self,
        batch_size: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        if batch_size is not None:
            self._state = self.initial_state(batch_size)
            self._prior_state = self.initial_state(batch_size)
            return
        if self._state is None:
            return
        if indices is None:
            self._state = None
            self._prior_state = None
            return
        idx = self._indices_from_mask(indices, self._state.deterministic.shape[0])
        if idx is None:
            return
        idx = idx.to(self._state.deterministic.device)
        self._state.deterministic[idx] = 0.0
        self._state.stochastic[idx] = 0.0
        self._state.mean[idx] = 0.0
        self._state.std[idx] = self.min_std
        if self._prior_state is not None:
            self._prior_state.deterministic[idx] = 0.0
            self._prior_state.stochastic[idx] = 0.0
            self._prior_state.mean[idx] = 0.0
            self._prior_state.std[idx] = self.min_std

    def get_state(self) -> RSSMState:
        if self._state is None:
            raise RuntimeError("RSSM state requested before observe() was called.")
        return self._state

    def set_state(self, state: RSSMState) -> None:
        self._state = state.to(self.device)

    def observe(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        *,
        detach_posteriors: bool = False,
    ) -> LatentStep:
        features = features.to(self.device)
        batch_size = features.shape[0]

        if self._state is None or self._state.deterministic.shape[0] != batch_size:
            self.reset_state(batch_size=batch_size)

        if reset_mask is not None:
            self.reset_state(indices=reset_mask)

        prev_state = self._state.clone()
        prior = self._compute_prior(prev_state, prev_action)
        posterior = self._compute_posterior(prior.deterministic, features)

        self._state = posterior.clone()
        self._prior_state = prior.clone()

        result = LatentStep(posterior=posterior, prior=prior)
        return result.detach() if detach_posteriors else result

    def observe_batch(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[LatentStep, RSSMState]:
        features = features.to(self.device)
        batch_size = features.shape[0]

        if state is None:
            state = self.initial_state(batch_size)
        else:
            state = state.to(self.device)

        if reset_mask is not None:
            idx = self._indices_from_mask(reset_mask, state.deterministic.shape[0])
            if idx is not None:
                idx = idx.to(state.deterministic.device)
                state.deterministic[idx] = 0.0
                state.stochastic[idx] = 0.0
                state.mean[idx] = 0.0
                state.std[idx] = self.min_std

        prior = self._compute_prior(state, prev_action)
        posterior = self._compute_posterior(prior.deterministic, features)
        return LatentStep(posterior=posterior, prior=prior), posterior.clone()

    def observe_sequence(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
    ) -> LatentSequence:
        features = features.to(self.device)
        batch_size, horizon, _ = features.shape

        if state is None:
            current_state = self.initial_state(batch_size)
        else:
            current_state = state.to(self.device)

        if actions is not None and self.action_dim > 0:
            actions = actions.to(self.device)
        else:
            actions = None

        if dones is not None:
            dones = dones.to(self.device)

        zero_action = (
            torch.zeros(batch_size, self.action_dim, device=self.device)
            if self.action_dim > 0
            else None
        )

        prior_det, prior_stoch, prior_mean, prior_std = [], [], [], []
        post_det, post_stoch, post_mean, post_std = [], [], [], []

        for t in range(horizon):
            if dones is not None and t > 0:
                reset_mask = dones[:, t - 1]
                idx = self._indices_from_mask(reset_mask, current_state.deterministic.shape[0])
                if idx is not None:
                    idx = idx.to(self.device)
                    current_state.deterministic[idx] = 0.0
                    current_state.stochastic[idx] = 0.0
                    current_state.mean[idx] = 0.0
                    current_state.std[idx] = self.min_std

            prior_state = current_state.clone()

            if self.action_dim > 0:
                if actions is None:
                    prev_action = zero_action
                elif t == 0:
                    prev_action = zero_action
                else:
                    prev_action = actions[:, t - 1]
            else:
                prev_action = None

            prior = self._compute_prior(prior_state, prev_action)
            posterior = self._compute_posterior(prior.deterministic, features[:, t])

            prior_det.append(prior.deterministic.clone())
            prior_stoch.append(prior.stochastic.clone())
            prior_mean.append(prior.mean.clone())
            prior_std.append(prior.std.clone())

            post_det.append(posterior.deterministic.clone())
            post_stoch.append(posterior.stochastic.clone())
            post_mean.append(posterior.mean.clone())
            post_std.append(posterior.std.clone())

            current_state = posterior.clone()

        posterior_stack = RSSMState(
            deterministic=torch.stack(post_det, dim=1),
            stochastic=torch.stack(post_stoch, dim=1),
            mean=torch.stack(post_mean, dim=1),
            std=torch.stack(post_std, dim=1),
        )
        prior_stack = RSSMState(
            deterministic=torch.stack(prior_det, dim=1),
            stochastic=torch.stack(prior_stoch, dim=1),
            mean=torch.stack(prior_mean, dim=1),
            std=torch.stack(prior_std, dim=1),
        )

        return LatentSequence(
            posterior=posterior_stack,
            prior=prior_stack,
            last_posterior=current_state.clone(),
        )

    def imagine_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> RSSMState:
        state = state.to(self.device)
        return self._compute_prior(state, action, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_prior(
        self,
        state: RSSMState,
        action: Optional[torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> RSSMState:
        action_tensor = self._prepare_action(action, state.deterministic.shape[0])
        if action_tensor is None:
            input_tensor = state.stochastic
        else:
            input_tensor = torch.cat([state.stochastic, action_tensor], dim=-1)
        hidden = self.input_layer(input_tensor)
        deter = self.gru(hidden, state.deterministic)

        stats = self.prior_mlp(deter)
        mean, std_param = _split_stats(stats)
        std = self._compute_std(std_param)
        stoch = self._sample(mean, std, deterministic=deterministic or not self.sample_prior)
        return RSSMState(deterministic=deter, stochastic=stoch, mean=mean, std=std)

    def _compute_posterior(self, deter: torch.Tensor, features: torch.Tensor) -> RSSMState:
        stats = self.posterior_mlp(torch.cat([deter, features], dim=-1))
        mean, std_param = _split_stats(stats)
        std = self._compute_std(std_param)
        stoch = self._sample(mean, std, deterministic=not self.sample_posterior)
        return RSSMState(deterministic=deter, stochastic=stoch, mean=mean, std=std)

    def _compute_std(self, std_param: torch.Tensor) -> torch.Tensor:
        if self.std_transform == "exp":
            std = torch.exp(std_param)
        else:
            std = F.softplus(std_param) + self.min_std
        if self.max_std is not None:
            std = torch.clamp(std, max=self.max_std)
        return std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return mean
        noise = torch.randn_like(std)
        return mean + noise * std

    def _prepare_action(self, action: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if self.action_dim <= 0:
            return None
        if action is None:
            return torch.zeros(batch_size, self.action_dim, device=self.device)
        action = action.to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if self.discrete_actions and action.shape[-1] == 1:
            one_hot = torch.zeros(batch_size, self.action_dim, device=self.device)
            indices = action.long().view(-1)
            one_hot.scatter_(1, indices.unsqueeze(-1), 1.0)
            action = one_hot
        return action

    @staticmethod
    def _indices_from_mask(mask: torch.Tensor, length: int) -> Optional[torch.Tensor]:
        mask = mask.to(mask.device)
        if mask.dtype == torch.bool:
            if mask.numel() != length:
                mask = mask.view(-1)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return None
            return idx.to(torch.long)
        if mask.dtype.is_floating_point:
            mask = mask >= 0.5
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return None
            return idx.to(torch.long)
        if mask.dtype in (torch.int32, torch.int64):
            return mask.view(-1).to(torch.long)
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")


__all__ = ["RSSMRepresentationLearner"]
