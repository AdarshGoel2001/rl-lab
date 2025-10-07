"""
Continue Predictor Components

This module contains components for predicting episode continuation from state representations.
Continue predictors are essential for world models to enable proper termination prediction.
"""

from .base import BaseContinuePredictor
from .mlp_continue import MLPContinuePredictor

__all__ = [
    'BaseContinuePredictor',
    'MLPContinuePredictor',
]
