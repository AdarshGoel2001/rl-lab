"""Learner for DreamerV3."""

from typing import Any, Dict


class DreamerV3Learner:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def update(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError


