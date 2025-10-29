"""Base interface for world-model dynamics modules."""

from __future__ import annotations

import abc
from typing import Any, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


class BaseDynamicsModel(nn.Module, metaclass=abc.ABCMeta):
    """Common contract for latent dynamics models used by planners or controllers.

    Subclasses predict the next latent state (or distribution parameters) given
    the current latent representation and an action. The default ``rollout`` helper
    iteratively applies ``forward`` for a sequence of actions, which planning-based
    workflows can override for efficiency.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged: dict[str, Any] = dict(config or {})
        if kwargs:
            merged.update(kwargs)
        self.config = merged

    @abc.abstractmethod
    def forward(
        self,
        state: Any,
        action: torch.Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict the next latent state conditioned on the provided action."""

    def rollout(
        self,
        state: Any,
        actions: Sequence[torch.Tensor] | torch.Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """Iteratively apply the transition model for a sequence of actions."""
        if isinstance(actions, torch.Tensor) and actions.dim() == 3:
            iterator: Iterable[torch.Tensor] = actions.transpose(0, 1)
        elif isinstance(actions, torch.Tensor):
            iterator = actions
        else:
            iterator = actions

        trajectory: List[Any] = []
        current_state = state
        for action in iterator:
            current_state = self.forward(current_state, action, deterministic=deterministic, **kwargs)
            trajectory.append(current_state)
        return trajectory


__all__ = ["BaseDynamicsModel"]
