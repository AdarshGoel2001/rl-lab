"""Core interfaces / protocols for algorithms and components."""

from typing import Protocol, Any, Dict


class Algorithm(Protocol):
    def update(self, batch: Any) -> Dict[str, float]:
        ...


class Policy(Protocol):
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        ...


class Critic(Protocol):
    def value(self, obs: Any) -> Any:
        ...


class WorldModel(Protocol):
    def imagine(self, start_states: Any, horizon: int) -> Any:
        ...


