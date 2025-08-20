"""Training runners.

- OnPolicyRunner for PPO-like algorithms
- OffPolicyRunner for SAC-like algorithms
- WorldModelRunner for Dreamer-style training
"""

from typing import Any


class OnPolicyRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self) -> None:
        raise NotImplementedError


class OffPolicyRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self) -> None:
        raise NotImplementedError


class WorldModelRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self) -> None:
        raise NotImplementedError


