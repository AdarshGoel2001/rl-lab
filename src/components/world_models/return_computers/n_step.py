"""N-step bootstrapped return computation for MuZero-style algorithms."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseReturnComputer


class NStepReturnComputer(BaseReturnComputer):
    """Computes n-step bootstrapped returns as used in MuZero.

    Returns are computed as:
        G_t = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})

    This provides a bias-variance tradeoff between Monte Carlo (n=∞) and TD(0) (n=1).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize n-step return computer.

        Args:
            config: Configuration containing:
                - gamma: Discount factor (default: 0.99)
                - n_step: Number of steps to bootstrap (default: 5)
        """
        super().__init__(config)
        self.n_step = int(config.get("n_step", 5))

    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute n-step bootstrapped returns.

        Args:
            rewards: Rewards with shape (B, T)
            dones: Done flags with shape (B, T)
            values: Value estimates with shape (B, T) for bootstrapping (required)
            **kwargs: Unused

        Returns:
            N-step returns with shape (B, T)

        Raises:
            ValueError: If values are not provided
        """
        if values is None:
            raise ValueError("N-step returns require value estimates for bootstrapping")

        if rewards.ndim != 2:
            raise ValueError(f"Expected 2D rewards array (B, T), got shape {rewards.shape}")

        batch_size, time_steps = rewards.shape
        returns = np.zeros_like(rewards, dtype=np.float32)

        # For each timestep, compute n-step return
        for t in range(time_steps):
            # Determine how many steps we can actually look ahead
            max_steps = min(self.n_step, time_steps - t)

            # Accumulate n-step reward
            discounted_reward = 0.0
            discount = 1.0

            for step in range(max_steps):
                idx = t + step
                discounted_reward += discount * rewards[:, idx]
                discount *= self.gamma

                # If episode ends, stop accumulating
                if idx < time_steps and np.any(dones[:, idx]):
                    # Set discount to 0 for done environments
                    mask = dones[:, idx].astype(np.float32)
                    discount = discount * (1.0 - mask)

            # Bootstrap with value if we didn't reach episode end
            bootstrap_idx = min(t + self.n_step, time_steps - 1)
            bootstrap_value = values[:, bootstrap_idx]

            # Zero out bootstrap for environments that terminated within n steps
            bootstrap_mask = np.ones(batch_size, dtype=np.float32)
            for step in range(max_steps):
                idx = t + step
                if idx < time_steps:
                    bootstrap_mask *= (1.0 - dones[:, idx].astype(np.float32))

            returns[:, t] = discounted_reward + discount * bootstrap_value * bootstrap_mask

        return returns
