"""Recurrent State-Space Model (RSSM) representation learner."""

from __future__ import annotations

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .base import BaseRepresentationLearner
from ....utils.registry import register_representation_learner
from ..rssm import RSSMState


def _build_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
    )


@register_representation_learner("rssm")
class RSSMRepresentationLearner(BaseRepresentationLearner):
    """Dreamer-style RSSM posterior with deterministic and stochastic state."""

    def _build_learner(self) -> None:
        self.feature_dim = self.config.get("feature_dim")
        self.stochastic_dim = self.config.get("latent_dim")
        if self.feature_dim is None or self.stochastic_dim is None:
            raise ValueError("RSSMRepresentationLearner requires 'feature_dim' and 'latent_dim'")

        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.deterministic_dim = self.config.get("deterministic_dim", self.hidden_dim)
        self.action_dim = self.config.get("action_dim", 0)
        if self.action_dim < 0:
            raise ValueError("'action_dim' must be non-negative")

        self.min_std = self.config.get("min_std", 0.1)
        self.sample_posteriors = not self.config.get("deterministic", False)

        # Observation encoder that maps raw features to embedding used by posterior.
        self.obs_encoder = _build_mlp(self.feature_dim, self.hidden_dim)

        # GRU transition maintains deterministic state h_t given previous stochastic state and action.
        transition_input_dim = self.stochastic_dim + self.action_dim
        self.transition = nn.GRUCell(transition_input_dim, self.deterministic_dim)

        # Prior network p(z_t | h_t)
        self.prior_net = _build_mlp(self.deterministic_dim, self.hidden_dim)
        self.prior_mean_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)
        self.prior_std_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)

        # Posterior network q(z_t | h_t, o_t)
        self.posterior_net = _build_mlp(self.deterministic_dim + self.hidden_dim, self.hidden_dim)
        self.posterior_mean_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)
        self.posterior_std_layer = nn.Linear(self.hidden_dim, self.stochastic_dim)

        # Optional decoder that reconstructs encoder features from latent state.
        decoder_input_dim = self.deterministic_dim + self.stochastic_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

        self._last_state: Optional[RSSMState] = None
        self._prev_state: Optional[RSSMState] = None
        self._last_features: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        """Create zero-initialised RSSM state for a batch."""
        device = device or self.device
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return RSSMState(deterministic=deterministic, stochastic=stochastic)

    def reset_state(self) -> None:
        """Forget any cached recurrent state."""
        self._prev_state = None
        self._last_state = None
        self._last_features = None

    def _ensure_state_batch(self, state: Optional[RSSMState], batch_size: int, device: torch.device) -> RSSMState:
        if state is not None and state.batch_size == batch_size:
            return state
        return self.initial_state(batch_size, device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        features: torch.Tensor,
        prev_state: Optional[RSSMState] = None,
        prev_action: Optional[torch.Tensor] = None,
        *,
        sample: bool = True,
    ) -> torch.Tensor:
        """Encode features into RSSM latent by updating posterior state.

        Args:
            features: Encoder features for the current observation.
            prev_state: Optional RSSM state carrying (h_{t-1}, z_{t-1}). If not
                provided, the learner falls back to the cached recurrent state or
                initialises a fresh one for the batch size.
            prev_action: Optional previous action a_{t-1}. Required when the
                transition depends on control inputs. If omitted, zeros are used.
            sample: Whether to sample from the posterior (default) or return mean.
        """
        if features.dim() != 2:
            features = features.reshape(features.shape[0], -1)

        batch_size = features.shape[0]
        device = features.device
        obs_embed = self.obs_encoder(features)

        cached_state = prev_state or self._prev_state
        state = self._ensure_state_batch(cached_state, batch_size, device)

        if self.action_dim > 0:
            if prev_action is None:
                action = torch.zeros(batch_size, self.action_dim, device=device)
            else:
                action = prev_action
                if action.dim() == 1:
                    action = action.unsqueeze(-1)
                if action.shape[0] != batch_size:
                    raise ValueError("prev_action batch size does not match features")
        else:
            action = None

        transition_input = state.stochastic
        if action is not None:
            transition_input = torch.cat([transition_input, action], dim=-1)

        deterministic = self.transition(transition_input, state.deterministic)

        prior_hidden = self.prior_net(deterministic)
        prior_mean = self.prior_mean_layer(prior_hidden)
        prior_std = F.softplus(self.prior_std_layer(prior_hidden)) + self.min_std

        posterior_input = torch.cat([deterministic, obs_embed], dim=-1)
        posterior_hidden = self.posterior_net(posterior_input)
        posterior_mean = self.posterior_mean_layer(posterior_hidden)
        posterior_std = F.softplus(self.posterior_std_layer(posterior_hidden)) + self.min_std

        if sample and self.sample_posteriors:
            noise = torch.randn_like(posterior_std)
            stochastic = posterior_mean + posterior_std * noise
        else:
            stochastic = posterior_mean

        new_state = RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
            prior_mean=prior_mean,
            prior_std=prior_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
        )

        self._last_state = new_state
        self._prev_state = new_state.detach()
        self._last_features = features

        return new_state.latent()

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.dim() != 2:
            representation = representation.reshape(representation.shape[0], -1)
        return self.decoder(representation)

    def representation_loss(self, features: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """KL divergence between posterior and prior."""
        state = kwargs.get("state")
        if state is None:
            state = self._last_state
            if state is None:
                _ = self.encode(features, **{k: kwargs[k] for k in ("prev_state", "prev_action") if k in kwargs})
                state = self._last_state
        if state is None:
            raise RuntimeError("RSSMRepresentationLearner.representation_loss called without a state")

        if state.posterior_mean is None or state.prior_mean is None:
            raise RuntimeError("RSSMState missing posterior/prior statistics for KL computation")

        posterior = Normal(state.posterior_mean, state.posterior_std)
        prior = Normal(state.prior_mean, state.prior_std)
        kl = kl_divergence(posterior, prior)
        if kl.dim() > 1:
            kl = kl.sum(dim=-1)
        kl_loss = kl.mean()
        return {
            "representation_loss": kl_loss,
            "representation_kl": kl_loss,
            "posterior_mean_abs": state.posterior_mean.abs().mean(),
            "prior_std_mean": state.prior_std.mean(),
        }

    @property
    def representation_dim(self) -> int:
        return self.deterministic_dim + self.stochastic_dim

    @property
    def deterministic_state_dim(self) -> int:
        return self.deterministic_dim

    @property
    def stochastic_state_dim(self) -> int:
        return self.stochastic_dim

    def get_cached_state(self) -> Optional[RSSMState]:
        return self._last_state
