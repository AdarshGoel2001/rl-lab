"""Stochastic Gaussian policy for SAC."""

from typing import Any


class SACActor:
    """Gaussian policy with reparameterization for SAC."""

    def __init__(self, network: Any) -> None:
        self.network = network

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """Sample or select action given observation."""
        raise NotImplementedError


