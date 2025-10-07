"""
Representation Learners Module

Representation learners take encoder features and learn structured representations
beyond raw features. They can implement techniques like VAEs, contrastive learning,
masked autoencoders, etc.
"""

from .base import BaseRepresentationLearner
from .identity import IdentityRepresentationLearner
from .autoencoder import MLPAutoencoderRepresentationLearner

__all__ = [
    'BaseRepresentationLearner',
    'IdentityRepresentationLearner',
    'MLPAutoencoderRepresentationLearner',
]
