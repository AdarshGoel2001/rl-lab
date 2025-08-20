"""Logging abstraction for wandb/aim or local CSV."""

from typing import Dict, Any


class Logger:
    def __init__(self, project: str | None = None, config: Dict[str, Any] | None = None) -> None:
        self.project = project
        self.config = config or {}

    def log(self, metrics: Dict[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass


