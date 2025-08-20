"""DINOv2 vision encoder wrapper stub."""

from typing import Any


class DINOv2Encoder:
    def __init__(self, config: Any | None = None) -> None:
        self.config = config or {}

    def __call__(self, obs: Any) -> Any:
        raise NotImplementedError


