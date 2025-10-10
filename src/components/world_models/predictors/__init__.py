"""World-model predictor heads (e.g., reward, discount)."""

from .reward.base import BaseRewardPredictor
from .reward.mlp import MLPRewardPredictor

__all__ = [
    "BaseRewardPredictor",
    "MLPRewardPredictor",
]

