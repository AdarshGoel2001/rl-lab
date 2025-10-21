"""Base controller interface for world-model agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.distributions import Distribution

from ..dynamics.base import BaseDynamicsModel
from ...value_functions.base import BaseValueFunction


class BaseController(ABC, torch.nn.Module):
    """Abstract interface for modules that choose actions from latent states."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def act(
        self,
        latent_state: torch.Tensor,
        dynamics_model: Optional[BaseDynamicsModel] = None,
        value_function: Optional[BaseValueFunction] = None,
        *,
        deterministic: bool = False,
        horizon: Optional[int] = None,
        **kwargs: Any,
    ) -> Distribution:
        """Return an action distribution conditioned on the latent state."""
        raise NotImplementedError
