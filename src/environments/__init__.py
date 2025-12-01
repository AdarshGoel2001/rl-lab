# Import all environment wrappers to ensure they are registered
from .base import BaseEnvironment, SpaceSpec
from .gym_wrapper import GymWrapper
from .vectorized_gym_wrapper import VectorizedGymWrapper
from .minigrid_wrapper import MiniGridWrapper
from .atari_wrapper import AtariEnvironment
from .carracing_wrapper import CarRacingWorldModelWrapper
from .vizdoom_wrapper import VizDoomWorldModelWrapper

__all__ = [
    'BaseEnvironment',
    'SpaceSpec', 
    'GymWrapper',
    'VectorizedGymWrapper',
    'MiniGridWrapper',
    'AtariEnvironment',
    'CarRacingWorldModelWrapper',
    'VizDoomWorldModelWrapper',
]
