"""Recurrent State-Space Model (RSSM) stub."""

from typing import Any


class RSSM:
    def __init__(self, config: Any | None = None) -> None:
        self.config = config or {}

    def step(self, prev_state: Any, action: Any, embed: Any) -> Any:
        raise NotImplementedError


