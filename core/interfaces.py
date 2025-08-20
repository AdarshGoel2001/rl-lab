"""Core interfaces / protocols for algorithms and components."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol


class Encoder(Protocol):
    """Feature extractor used by actors/critics."""

    def __call__(self, obs: Any) -> Any:
        ...


class PolicyHead(Protocol):
    """Produces an action distribution from features."""

    def forward(self, feats: Any) -> Any:
        ...


class ValueHead(Protocol):
    """Maps features to a scalar state value."""

    def forward(self, feats: Any) -> Any:
        ...


class Algorithm(Protocol):
    def update(self, batch: Any) -> Dict[str, float]:
        ...


class Policy(Protocol):
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        ...


class Critic(Protocol):
    def value(self, obs: Any) -> Any:
        ...


class AdvantageEstimator(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class Buffer(Protocol):
    def add(self, *args: Any, **kwargs: Any) -> None:
        ...

    def get_minibatches(self, batch_size: int) -> Iterable[Any]:
        ...


class Scheduler(Protocol):
    def value(self) -> float:
        ...

    def step(self) -> None:
        ...


class WorldModel(Protocol):
    def imagine(self, start_states: Any, horizon: int) -> Any:
        ...


