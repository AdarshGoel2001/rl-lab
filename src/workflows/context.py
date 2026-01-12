"""Shared context passed to world-model workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config
from .controllers import ControllerManager


@dataclass(frozen=True)
class WorldModelComponents:
    """Dynamic container holding instantiated world-model modules."""

    components: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        components = super().__getattribute__("components")
        if name in components:
            return components[name]
        raise AttributeError(f"No component named '{name}'")

    def to(self, device: Any) -> "WorldModelComponents":
        """Move all components to the specified device."""
        for module in self.components.values():
            if module is None:
                continue
            mover = getattr(module, "to", None)
            if callable(mover):
                mover(device)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary view of the component bundle."""
        return {
            "components": dict(self.components),
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class WorkflowContext:
    """Container bundling resources shared between orchestrator and workflows."""

    config: Config
    device: str
    train_environment: Any
    eval_environment: Any
    components: WorldModelComponents
    checkpoint_manager: CheckpointManager
    experiment_logger: Any
    buffers: Dict[str, Any] = field(default_factory=dict)
    controllers: Optional[Dict[str, Any]] = None
    controller_manager: Optional[ControllerManager] = None
    simulator_service: Optional[Any] = None
    optimizers: Optional[Dict[str, Any]] = None
    initial_observation: Optional[Any] = None
    initial_dones: Optional[Any] = None
    global_step: int = 0

    def with_updates(self, **kwargs: Any) -> "WorkflowContext":
        """Return a shallow copy with updated fields."""

        return replace(self, **kwargs)
