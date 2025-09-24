"""
Representation Learners Module

Representation learners take encoder features and learn structured representations
beyond raw features. They can implement techniques like VAEs, contrastive learning,
masked autoencoders, etc.
"""

from .base import BaseRepresentationLearner

__all__ = ['BaseRepresentationLearner']