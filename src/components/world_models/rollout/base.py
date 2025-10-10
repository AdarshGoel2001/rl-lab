"""Shared rollout configuration structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration parameters for imagination rollouts."""

    horizon: int = 15
    deterministic_policy: bool = False
    deterministic_dynamics: bool = False
