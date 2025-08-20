"""Actor for DreamerV3, operating in latent space."""

from typing import Any


class DreamerV3Actor:
    def __init__(self, network: Any) -> None:
        self.network = network

    def act_latent(self, latent: Any, deterministic: bool = False) -> Any:
        raise NotImplementedError


