"""Critic for DreamerV3, value in latent space."""

from typing import Any


class DreamerV3Critic:
    def __init__(self, network: Any) -> None:
        self.network = network

    def value_latent(self, latent: Any) -> Any:
        raise NotImplementedError


