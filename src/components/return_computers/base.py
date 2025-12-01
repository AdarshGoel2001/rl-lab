"""Base class for return computation strategies in world model paradigms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseReturnComputer(ABC):
    """Abstract base class for computing returns from trajectory sequences.

    Different world model algorithms require different return computation strategies:
    - Dreamer: No returns needed (trains only on imagined rollouts)
    - MuZero: N-step bootstrapped returns
    - TD-MPC: TD(λ) returns for temporal difference learning
    - Offline methods: Monte Carlo returns

    This base class provides the interface for pluggable return computation.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the return computer.

        Args:
            config: Configuration dictionary containing:
                - gamma: Discount factor (default: 0.99)
                - Additional strategy-specific parameters
        """
        self.config = config
        self.gamma = float(config.get("gamma", 0.99))

    @abstractmethod
    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute returns from trajectory data.

        Args:
            rewards: Reward array with shape (B, T) where B=batch, T=time
            dones: Done flags with shape (B, T)
            values: Optional value estimates with shape (B, T) for bootstrapping
            **kwargs: Additional strategy-specific arguments

        Returns:
            Returns array with shape (B, T)
        """
        raise NotImplementedError

    def __call__(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Convenience method to compute returns."""
        return self.compute_returns(rewards, dones, values, **kwargs)
