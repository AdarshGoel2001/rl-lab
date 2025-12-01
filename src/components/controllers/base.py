"""Base controller protocol for RL Lab."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class BaseController(Protocol):
    """Protocol for all controllers (actors, critics, planners)."""

    def act(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Select action given observation/latent state."""
        ...
