"""Utility helpers for reproducibility and device handling."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return ``"cuda"`` if available else ``"cpu"``."""

    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def mixed_precision(enabled: bool) -> Iterator[None]:
    """Context manager for optional ``torch.autocast``."""

    if enabled and torch.cuda.is_available():
        with torch.autocast("cuda"):
            yield
    else:
        yield


class MixedPrecisionContext:
    """Backward compatible wrapper around :func:`mixed_precision`."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __enter__(self) -> "MixedPrecisionContext":
        self._cm = mixed_precision(self.enabled)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._cm.__exit__(exc_type, exc, tb)
        return None


