"""No-op return computer for algorithms that don't need real-data returns (e.g., Dreamer)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseReturnComputer


class NoReturnComputer(BaseReturnComputer):
    """Return computer that returns None - for algorithms that train only on imagined rollouts.

    Use this for Dreamer-style algorithms where the critic is trained exclusively
    on imagined trajectories and doesn't need real-data returns.
    """

    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """Return None - no computation needed.

        Args:
            rewards: Unused
            dones: Unused
            values: Unused
            **kwargs: Unused

        Returns:
            None
        """
        return None
