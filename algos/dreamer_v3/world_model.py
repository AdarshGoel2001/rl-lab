"""World model components for DreamerV3.

This module would host RSSM or SSM/Mamba-based dynamics models for latent
rollouts and imagination.
"""

from typing import Any


class WorldModel:
    """Abstract world model interface for DreamerV3."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config or {}

    def imagine(self, start_states: Any, horizon: int) -> Any:
        """Perform latent rollouts for a given horizon."""
        raise NotImplementedError


