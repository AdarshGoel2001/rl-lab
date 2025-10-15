"""Base interfaces for world-model workflows."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

Batch = Any
PhaseConfig = Mapping[str, Any]


@dataclass
class CollectResult:
    """Summary of environment interaction produced by a workflow collect step."""

    episodes: int = 0
    steps: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class SupportsStateDict(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


class WorldModelWorkflow(abc.ABC):
    """Algorithm-specific contract implemented by Dreamer, MuZero, etc."""

    @abc.abstractmethod
    def initialize(self, context: "WorkflowContext") -> None:
        """Populate internal references and reset state."""

    def on_context_update(self, context: "WorkflowContext") -> None:
        """Called when the orchestrator refreshes the workflow context."""

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        """Interact with the environment or simulator to generate experience."""

    @abc.abstractmethod
    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Apply world-model parameter updates and return metrics."""

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Optional policy/controller learning step."""
        return {}

    def plan_phase(self, step: int) -> Optional[str]:
        """Allow the workflow to request a phase transition."""
        return None

    def imagine(
        self,
        *,
        observations: Any = None,
        latent: Any = None,
        horizon: Optional[int] = None,
        deterministic: bool = False,
        controller_role: Optional[str] = None,
        action_sequence: Any = None,
    ) -> Dict[str, Any]:
        """Optional imagination helper for planner-style controllers."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement imagination rollouts.")

    def log_metrics(self, step: int, writer: Any) -> None:
        """Emit workflow-specific metrics to the experiment logger."""

    def state_dict(self) -> Dict[str, Any]:
        """Persist workflow-managed state."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore workflow-managed state."""


from .context import WorkflowContext  # noqa: E402  (avoid circular import)
