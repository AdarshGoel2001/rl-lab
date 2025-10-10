"""Identity representation learner scoped to world models."""

from __future__ import annotations

from typing import Dict, Any

import torch

from .base import BaseRepresentationLearner
from ....utils.registry import register_representation_learner


@register_representation_learner("identity")
class IdentityRepresentationLearner(BaseRepresentationLearner):
    """Pass encoder features through unchanged."""

    def _build_learner(self) -> None:
        self.feature_dim = self.config.get("feature_dim")
        if self.feature_dim is None:
            raise ValueError("IdentityRepresentationLearner requires 'feature_dim'")

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def representation_loss(self, features: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        zero = torch.zeros(1, device=features.device)
        return {
            "representation_loss": zero,
        }

    @property
    def representation_dim(self) -> int:
        return self.feature_dim
