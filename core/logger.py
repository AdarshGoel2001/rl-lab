"""Logging abstraction for wandb/aim or local CSV."""

from __future__ import annotations

from typing import Any, Dict


class Logger:
    """Minimal logger that prints metrics to stdout."""

    def __init__(
        self, project: str | None = None, config: Dict[str, Any] | None = None
    ) -> None:
        self.project = project
        self.config = config or {}

    def log(self, metrics: Dict[str, float], step: int | None = None) -> None:
        prefix = f"[step {step}] " if step is not None else ""
        items = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        print(prefix + items)

    def flush(self) -> None:
        pass


