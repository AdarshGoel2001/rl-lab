"""
Components Module

This module contains all the modular components that can be composed
to create different agent paradigms. Each component type has its own
subdirectory with concrete implementations.

Component Types:
- encoders: Transform observations to feature representations
- representation_learners: Learn structured representations beyond raw features
- dynamics: Predict how the world evolves
- controllers: Action selection (actors, critics, planners)
- decoders: Reconstruct observations from latent states
- return_computers: Compute returns/advantages

Components are instantiated via Hydra configs using _target_ paths.
No base classes required - each component is independent.
"""
