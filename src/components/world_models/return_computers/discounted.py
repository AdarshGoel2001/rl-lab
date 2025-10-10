"""Simple discounted (Monte Carlo) return computation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseReturnComputer
from ....utils.registry import register_return_computer


@register_return_computer("discounted")
class DiscountedReturnComputer(BaseReturnComputer):
    """Computes discounted Monte Carlo returns without bootstrapping.

    Returns are computed as:
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(T-t)*r_T

    This is suitable for episodic tasks or when you want pure Monte Carlo estimates.
    """

    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute discounted returns backward through time.

        Args:
            rewards: Rewards with shape (B, T) where B=batch, T=time
            dones: Done flags with shape (B, T)
            values: Unused (Monte Carlo doesn't bootstrap)
            **kwargs: Unused

        Returns:
            Discounted returns with shape (B, T)
        """
        if rewards.ndim != 2:
            raise ValueError(f"Expected 2D rewards array (B, T), got shape {rewards.shape}")

        batch_size, time_steps = rewards.shape
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = np.zeros(batch_size, dtype=np.float32)

        # Compute returns backward through time
        for t in reversed(range(time_steps)):
            # Reset return to 0 for environments that are done
            running_return[dones[:, t]] = 0.0

            # Accumulate discounted return
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return

        return returns
