"""Policy network wrapper for PPO."""

from typing import Any


class PPOActor:
    """Wraps a policy network for action selection under PPO."""

    def __init__(self, network: Any) -> None:
        self.network = network

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """Compute an action given an observation.

        Args:
            obs: Observation(s) from the environment.
            deterministic: If True, select the mean/greedy action.
        """
        raise NotImplementedError


