"""Reward prediction heads."""

from .base import BaseRewardPredictor
from .mlp import MLPRewardPredictor

__all__ = [
    "BaseRewardPredictor",
    "MLPRewardPredictor",
]

