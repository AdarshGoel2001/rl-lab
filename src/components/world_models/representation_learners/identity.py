"""Identity representation learner that passes encoder features through."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


from .base import BaseRepresentationLearner, LatentSequence, LatentStep, RSSMState


def _infer_feature_dim(config: Dict[str, Any]) -> int:
    for key in ("feature_dim", "latent_dim", "representation_dim", "input_dim"):
        value = config.get(key)
        if value is not None:
            return int(value)
    raise ValueError(
        "IdentityRepresentationLearner requires one of "
        "'feature_dim', 'latent_dim', 'representation_dim', or 'input_dim' in config."
    )


class IdentityRepresentationLearner(BaseRepresentationLearner):
    """Fallback learner that treats encoder outputs as latent states."""

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.feature_dim = _infer_feature_dim(self.config)
        self._min_std = float(self.config.get("min_std", 1e-3))
        self._state: Optional[RSSMState] = None

    # ------------------------------------------------------------------
    # BaseRepresentationLearner API
    # ------------------------------------------------------------------
    @property
    def representation_dim(self) -> int:
        return self.feature_dim

    def initial_state(self, batch_size: int) -> RSSMState:
        zeros = torch.zeros(batch_size, self.feature_dim, device=self.device)
        std = torch.ones_like(zeros) * self._min_std
        return RSSMState(
            deterministic=zeros.clone(),
            stochastic=zeros.clone(),
            mean=zeros.clone(),
            std=std,
        )

    def reset_state(
        self,
        batch_size: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        if batch_size is not None:
            self._state = self.initial_state(batch_size)
            return
        if self._state is None:
            return
        if indices is None:
            self._state = None
            return
        idx = self._indices_from_mask(indices, self._state.deterministic.shape[0])
        if idx is None:
            return
        idx = idx.to(self.device)
        self._state.deterministic[idx] = 0.0
        self._state.stochastic[idx] = 0.0
        self._state.mean[idx] = 0.0
        self._state.std[idx] = self._min_std

    def get_state(self) -> RSSMState:
        if self._state is None:
            self._state = self.initial_state(1)
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
        del prev_action  # Identity learner ignores actions.
        features = features.to(self.device)

        batch_size = features.shape[0]
        if self._state is None or self._state.deterministic.shape[0] != batch_size:
            self._state = self.initial_state(batch_size)

        if reset_mask is not None:
            self.reset_state(indices=reset_mask)

        prior = self._state.clone()
        posterior = RSSMState(
            deterministic=features,
            stochastic=features,
            mean=features,
            std=torch.ones_like(features, device=self.device) * self._min_std,
        )
        self._state = posterior.clone()

        result = LatentStep(posterior=posterior, prior=prior)
        return result.detach() if detach_posteriors else result

    def observe_batch(
        self,
        features: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> tuple[LatentStep, RSSMState]:
        del prev_action
        features = features.to(self.device)

        if state is None:
            state = self.initial_state(features.shape[0])
        else:
            state = state.to(self.device)

        if reset_mask is not None:
            idx = self._indices_from_mask(reset_mask, state.deterministic.shape[0])
            if idx is not None:
                idx = idx.to(state.deterministic.device)
                state.deterministic[idx] = 0.0
                state.stochastic[idx] = 0.0
                state.mean[idx] = 0.0
                state.std[idx] = self._min_std

        posterior = RSSMState(
            deterministic=features,
            stochastic=features,
            mean=features,
            std=torch.ones_like(features) * self._min_std,
        )
        step = LatentStep(posterior=posterior, prior=state.clone())
        return step, posterior.clone()

    def observe_sequence(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        state: Optional[RSSMState] = None,
    ) -> LatentSequence:
        del actions  # Identity learner does not depend on actions.
        features = features.to(self.device)
        batch_size, horizon, _ = features.shape

        if state is None:
            current_state = self.initial_state(batch_size)
        else:
            current_state = state.to(self.device)

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
                    current_state.std[idx] = self._min_std

            prior_det.append(current_state.deterministic.clone())
            prior_stoch.append(current_state.stochastic.clone())
            prior_mean.append(current_state.mean.clone())
            prior_std.append(current_state.std.clone())

            feat_t = features[:, t]
            posterior = RSSMState(
                deterministic=feat_t,
                stochastic=feat_t,
                mean=feat_t,
                std=torch.ones_like(feat_t) * self._min_std,
            )

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
        del action, deterministic
        return state.clone()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _indices_from_mask(mask: torch.Tensor, length: int) -> Optional[torch.Tensor]:
        if mask.dtype == torch.bool:
            if mask.numel() != length:
                mask = mask.view(-1)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return None
            return idx.to(torch.long)
        if mask.dtype.is_floating_point or mask.dtype == torch.int32 or mask.dtype == torch.int64:
            mask = mask.to(torch.long).view(-1)
            return mask
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")
