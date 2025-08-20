"""Twin Q-value critics for SAC."""

from typing import Any, Tuple


class SACCritic:
    """Wraps twin Q networks for SAC."""

    def __init__(self, q1_network: Any, q2_network: Any) -> None:
        self.q1 = q1_network
        self.q2 = q2_network

    def q_values(self, obs: Any, act: Any) -> Tuple[Any, Any]:
        """Return Q1 and Q2 estimates for given (obs, act)."""
        raise NotImplementedError


