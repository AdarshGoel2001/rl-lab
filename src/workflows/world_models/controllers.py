"""Helper utilities for managing workflow controllers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch.distributions import Distribution


class ControllerManager:
    """Lightweight wrapper coordinating controller instances by role."""

    def __init__(self, controllers: Optional[Mapping[str, Any]] = None) -> None:
        self._controllers: Dict[str, Any] = dict(controllers or {})

    def __contains__(self, role: str) -> bool:
        return role in self._controllers

    def __len__(self) -> int:
        return len(self._controllers)

    def roles(self) -> Iterable[str]:
        return self._controllers.keys()

    def get(self, role: str) -> Any:
        if role not in self._controllers:
            raise KeyError(f"Controller role '{role}' is not registered.")
        return self._controllers[role]

    def add(self, role: str, controller: Any) -> None:
        self._controllers[role] = controller

    def act(
        self,
        role: str,
        *args: Any,
        deterministic: bool = False,
        sample_action: bool = False,
        return_distribution: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Request an action distribution (and optional sample) from a controller."""
        controller = self.get(role)
        hook = getattr(controller, "act", None)
        if not callable(hook):
            raise AttributeError(f"Controller '{role}' does not implement act().")

        distribution = hook(*args, deterministic=deterministic, **kwargs)
        if sample_action and isinstance(distribution, Distribution):
            action = self._sample_from_distribution(distribution, deterministic=deterministic)
            if return_distribution:
                return action, distribution
            return action
        return distribution

    def learn(
        self,
        role: str,
        batch: Mapping[str, Any],
        *,
        phase: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        controller = self.get(role)
        hook = getattr(controller, "learn", None)
        if hook is None:
            return {}

        metrics = hook(batch, phase=phase)  # type: ignore[misc]
        if metrics is None:
            return {}
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"Controller '{role}' learn() returned non-mapping metrics: {type(metrics)}"
            )
        return {f"{role}/{key}": float(value) for key, value in metrics.items()}

    def learn_all(
        self,
        batch: Mapping[str, Any],
        *,
        phase: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        for role in self.roles():
            metrics = self.learn(role, batch, phase=phase)
            if metrics:
                combined.update(metrics)
        return combined

    def state_dict(self, *, mode: str = "checkpoint") -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for role, controller in self._controllers.items():
            state_hook = getattr(controller, "state_dict", None)
            if not callable(state_hook):
                continue
            try:
                snapshot[role] = state_hook(mode=mode)  # type: ignore[call-arg]
            except TypeError:
                if mode == "checkpoint":
                    snapshot[role] = state_hook()
                else:
                    continue
        return snapshot

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        for role, payload in state.items():
            controller = self._controllers.get(role)
            if controller is None:
                continue
            load_hook = getattr(controller, "load_state_dict", None)
            if callable(load_hook):
                load_hook(payload)

    def items(self):
        return self._controllers.items()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._controllers)

    @staticmethod
    def _sample_from_distribution(
        distribution: Distribution,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        if deterministic:
            if hasattr(distribution, "mode"):
                mode = distribution.mode
                if callable(mode):
                    return mode()
                return mode
            if hasattr(distribution, "mean"):
                mean = distribution.mean
                if callable(mean):
                    return mean()
                return mean
            if hasattr(distribution, "probs"):
                return torch.argmax(distribution.probs, dim=-1)
        if getattr(distribution, "has_rsample", False):
            return distribution.rsample()
        return distribution.sample()
