"""Learning rate / coefficient schedulers."""

from __future__ import annotations


class LinearScheduler:
    """Linearly anneal a value from ``start`` to ``end`` over ``n_steps``."""

    def __init__(self, start: float, end: float, n_steps: int) -> None:
        self.start = start
        self.end = end
        self.n_steps = max(1, n_steps)
        self.step_count = 0

    def value(self) -> float:
        fraction = min(self.step_count / self.n_steps, 1.0)
        return self.start + fraction * (self.end - self.start)

    def step(self) -> None:
        self.step_count += 1

