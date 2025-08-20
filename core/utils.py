"""Utilities: seeding, device selection, mixed precision helpers."""

from typing import Any


def set_seed(seed: int) -> None:
    raise NotImplementedError


def get_device() -> str:
    raise NotImplementedError


class MixedPrecisionContext:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __enter__(self) -> "MixedPrecisionContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


