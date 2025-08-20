"""Mamba SSM dynamics stub for world models."""

from typing import Any


class MambaSSM:
    def __init__(self, config: Any | None = None) -> None:
        self.config = config or {}

    def step(self, prev_state: Any, action: Any, embed: Any) -> Any:
        raise NotImplementedError


