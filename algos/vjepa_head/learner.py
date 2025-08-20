"""Learner for V-JEPA head."""

from typing import Any, Dict


class VJEPALearner:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def update(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError


