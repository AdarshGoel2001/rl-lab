"""TD(λ) return computation for TD-MPC and similar algorithms."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseReturnComputer


class TDLambdaReturnComputer(BaseReturnComputer):
    """Computes TD(λ) returns (also known as λ-returns or eligibility traces).

    Returns are computed as a weighted average of all n-step returns:
        G_t^λ = (1-λ) * Σ_{n=1}^{∞} λ^{n-1} * G_t^{(n)}

    Where G_t^{(n)} is the n-step return. This provides a smooth interpolation
    between TD(0) (λ=0) and Monte Carlo (λ=1).

    This implementation uses the backward view (eligibility traces) for efficiency.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize TD(λ) return computer.

        Args:
            config: Configuration containing:
                - gamma: Discount factor (default: 0.99)
                - lambda_coef: Lambda coefficient (default: 0.95)
                - use_gae: Whether to use GAE formulation (default: False)
        """
        super().__init__(config)
        self.lambda_coef = float(config.get("lambda_coef", 0.95))
        self.use_gae = bool(config.get("use_gae", False))

    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        next_values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute TD(λ) returns using the backward view.

        Args:
            rewards: Rewards with shape (B, T)
            dones: Done flags with shape (B, T)
            values: Value estimates V(s_t) with shape (B, T) (required)
            next_values: Optional V(s_{t+1}) with shape (B, T). If not provided,
                        will shift values forward by one step.
            **kwargs: Unused

        Returns:
            TD(λ) returns with shape (B, T)

        Raises:
            ValueError: If values are not provided
        """
        if values is None:
            raise ValueError("TD(λ) returns require value estimates")

        if rewards.ndim != 2:
            raise ValueError(f"Expected 2D rewards array (B, T), got shape {rewards.shape}")

        batch_size, time_steps = rewards.shape

        # Compute next values if not provided
        if next_values is None:
            next_values = np.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]
            # Last timestep has next_value = 0 (terminal)

        # Compute TD errors (temporal difference)
        # δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        td_errors = rewards + self.gamma * next_values * (1.0 - dones.astype(np.float32)) - values

        # Compute λ-returns using backward view
        # This is equivalent to computing weighted average of all n-step returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = np.zeros(batch_size, dtype=np.float32)

        # Backward pass through time
        for t in reversed(range(time_steps)):
            # λ-return: r_t + γ*[(1-λ)*V(s_{t+1}) + λ*G_{t+1}]
            # Which can be rewritten as: V(s_t) + δ_t + γ*λ*(G_{t+1} - V(s_{t+1}))

            # Reset for environments that are done
            running_return[dones[:, t]] = 0.0

            if self.use_gae:
                # GAE formulation: advantage = δ_t + γ*λ*advantage_{t+1}
                # Then return = advantage + value
                running_return = td_errors[:, t] + self.gamma * self.lambda_coef * running_return * (1.0 - dones[:, t].astype(np.float32))
                returns[:, t] = running_return + values[:, t]
            else:
                # Direct λ-return formulation
                running_return = (
                    rewards[:, t]
                    + self.gamma * (
                        (1.0 - self.lambda_coef) * next_values[:, t]
                        + self.lambda_coef * running_return
                    ) * (1.0 - dones[:, t].astype(np.float32))
                )
                returns[:, t] = running_return

        return returns
