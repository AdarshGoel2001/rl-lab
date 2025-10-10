"""Declarative schedule helpers for world-model training phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingPhase:
    """Represents a block of updates with a shared loss configuration."""

    name: str
    active_losses: List[str]
    update_modules: List[str]
    steps: int


@dataclass
class TrainingSchedule:
    """Sequence of training phases applied over iterations."""

    phases: List[TrainingPhase] = field(default_factory=list)

    def active_phase(self, step: int) -> TrainingPhase:
        """Return the phase covering the given global step."""
        accumulated = 0
        for phase in self.phases:
            accumulated += phase.steps
            if step < accumulated:
                return phase
        return self.phases[-1]
