"""Value network wrapper for PPO."""

from typing import Any


class PPOCritic:
    """Wraps a value function network for PPO."""

    def __init__(self, network: Any) -> None:
        self.network = network

    def value(self, obs: Any) -> Any:
        """Estimate the state value for given observations."""
        raise NotImplementedError


