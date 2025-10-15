"\"\"\"Base data source interface used by world-model orchestrator.\"\"\""

from __future__ import annotations

import abc
from typing import Any, Mapping, Optional


class DataSource(abc.ABC):
    """Abstract interface for sampling training data during orchestrated runs."""

    @abc.abstractmethod
    def initialize(self, context: "WorkflowContext") -> None:  # pragma: no cover - interface
        """Attach shared context resources."""

    @abc.abstractmethod
    def add(self, **kwargs: Any) -> None:  # pragma: no cover - interface
        """Insert new data into the backing store."""

    @abc.abstractmethod
    def sample(self, batch_size: Optional[int] = None) -> Any:  # pragma: no cover - interface
        """Return a batch for training updates."""

    def ready(self) -> bool:
        """Whether the data source can produce training batches."""
        return False

    def state_dict(self) -> Mapping[str, Any]:
        """Serializable snapshot of data source state."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore data source state."""


from ..workflows.world_models.context import WorkflowContext  # noqa: E402  (avoid circular import)
