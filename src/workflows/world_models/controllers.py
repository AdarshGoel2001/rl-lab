"""Helper utilities for managing workflow controllers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


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

    def state_dict(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for role, controller in self._controllers.items():
            state_hook = getattr(controller, "state_dict", None)
            if callable(state_hook):
                snapshot[role] = state_hook()
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
