"""Base interface for dynamics models.

TODO: This module needs to be implemented when adding planning-based controllers
(MPC, MCTS, etc.) that require explicit dynamics models for trajectory simulation.

For Dreamer, dynamics are handled by RSSM.imagine_step() in the representation
learner, so this is not used yet.
"""

from __future__ import annotations


class BaseDynamicsModel:
    """Placeholder for dynamics model interface.

    TODO: Implement abstract interface for dynamics models that predict
    next latent states given current state and action.

    Will be needed for:
    - TD-MPC (deterministic dynamics)
    - MuZero (learned dynamics)
    - Planning-based controllers (MPC/MCTS)
    """
    pass


__all__ = ["BaseDynamicsModel"]
