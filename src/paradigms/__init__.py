"""
Paradigms Module

Paradigms are high-level agent architectures that compose different components
to create complete RL agents. Examples include model-free, world model, VLA,
and hybrid paradigms.
"""

from .base import BaseParadigm
from .model_free import ModelFreeParadigm
from .world_model import WorldModelParadigm

__all__ = ['BaseParadigm', 'ModelFreeParadigm', 'WorldModelParadigm']