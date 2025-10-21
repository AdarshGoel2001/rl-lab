"""Controllers decide how to act on latent states."""

from .base import BaseController
from .dreamer import DreamerActorController, DreamerCriticController

__all__ = [
    "BaseController",
    "DreamerActorController",
    "DreamerCriticController",
]
