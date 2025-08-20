"""Experience buffers.

- RolloutBuffer for on-policy methods (e.g., PPO)
- ReplayBuffer for off-policy methods (e.g., SAC)
"""

from typing import Any


class RolloutBuffer:
    def add(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ReplayBuffer:
    def add(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Any:
        raise NotImplementedError


