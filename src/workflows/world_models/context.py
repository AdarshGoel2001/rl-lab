"""Shared context passed to world-model workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, TYPE_CHECKING

from ...utils.checkpoint import CheckpointManager
from ...utils.config import Config
from .controllers import ControllerManager

if TYPE_CHECKING:
    from ...data_sources.base import DataSource


@dataclass(frozen=True)
class WorldModelComponents:
    """Container holding instantiated world-model modules."""

    encoder: Any
    representation_learner: Any
    dynamics_model: Any
    reward_predictor: Optional[Any] = None
    observation_decoder: Optional[Any] = None
    planner: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    specs: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: Any) -> "WorldModelComponents":
        """Move all components to the specified device."""
        targets = [
            self.encoder,
            self.representation_learner,
            self.dynamics_model,
            self.reward_predictor,
            self.observation_decoder,
            self.planner,
        ]
        for module in targets:
            if module is None:
                continue
            mover = getattr(module, "to", None)
            if callable(mover):
                mover(device)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary view of the component bundle."""
        return {
            "encoder": self.encoder,
            "representation_learner": self.representation_learner,
            "dynamics_model": self.dynamics_model,
            "reward_predictor": self.reward_predictor,
            "observation_decoder": self.observation_decoder,
            "planner": self.planner,
            "config": self.config,
            "specs": self.specs,
        }


@dataclass(frozen=True)
class WorkflowContext:
    """Container bundling resources shared between orchestrator and workflows."""

    config: Config
    device: str
    train_environment: Any
    eval_environment: Any
    components: WorldModelComponents
    buffer: Any
    checkpoint_manager: CheckpointManager
    experiment_logger: Any
    controllers: Optional[Dict[str, Any]] = None
    controller_manager: Optional[ControllerManager] = None
    data_sources: Optional[Dict[str, "DataSource"]] = None
    simulator_service: Optional[Any] = None
    global_step: int = 0

    def with_updates(self, **kwargs: Any) -> "WorkflowContext":
        """Return a shallow copy with updated fields."""

        return replace(self, **kwargs)
