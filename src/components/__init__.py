"""
Components Module

This module contains all the modular components that can be composed
to create different agent paradigms. Each component type has its own
subdirectory with a base abstract class and concrete implementations.

Component Types:
- encoders: Transform observations to feature representations
- representation_learners: Learn structured representations beyond raw features
- dynamics: Predict how the world evolves
- policy_heads: Convert representations to action distributions
- value_functions: Estimate state or state-action values
- planners: Plan actions using world models

Each component is registered using decorators and can be instantiated
via configuration files, enabling plug-and-play experimentation.
"""

# Ensure world-model specific components register with the global registry
from .world_models import (  # noqa: F401
    RSSMRepresentationLearner,
    RSSMDynamicsModel,
    MLPRewardPredictor,
)
