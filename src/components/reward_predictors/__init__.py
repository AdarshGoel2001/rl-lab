"""
Reward Predictor Components

This module contains components for predicting rewards from state representations.
Reward predictors are essential for world models to enable imagination-based planning.
"""

from .base import BaseRewardPredictor
from .mlp_reward import MLPRewardPredictor

__all__ = [
    'BaseRewardPredictor',
    'MLPRewardPredictor',
]
