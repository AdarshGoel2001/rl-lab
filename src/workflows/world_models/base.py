"""Base interfaces for world-model workflows."""

from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

import numpy as np

Batch = Any
PhaseConfig = Mapping[str, Any]


@dataclass
class CollectResult:
    """Summary of environment interaction produced by a workflow collect step."""

    episodes: int = 0
    steps: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class SupportsStateDict(Protocol):
    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


class WorldModelWorkflow(abc.ABC):
    """Algorithm-specific contract implemented by Dreamer, MuZero, etc."""

    def __init__(self, *, episode_history: int = 100) -> None:
        self._episode_history_len = max(int(episode_history), 1)
        self.num_envs = 1
        self.episode_returns: deque[float] = deque(maxlen=self._episode_history_len)
        self.episode_lengths: deque[int] = deque(maxlen=self._episode_history_len)
        self.total_episodes = 0
        self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    @abc.abstractmethod
    def initialize(self, context: "WorkflowContext") -> None:
        """Populate internal references and reset state."""

    def on_context_update(self, context: "WorkflowContext") -> None:
        """Called when the orchestrator refreshes the workflow context."""

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        """Interact with the environment or simulator to generate experience."""

    @abc.abstractmethod
    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Apply world-model parameter updates and return metrics."""

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Optional policy/controller learning step."""
        return {}

    def plan_phase(self, step: int) -> Optional[str]:
        """Allow the workflow to request a phase transition."""
        return None

    def imagine(
        self,
        *,
        observations: Any = None,
        latent: Any = None,
        horizon: Optional[int] = None,
        deterministic: bool = False,
        controller_role: Optional[str] = None,
        action_sequence: Any = None,
    ) -> Dict[str, Any]:
        """Optional imagination helper for planner-style controllers."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement imagination rollouts.")

    def log_metrics(self, step: int, writer: Any) -> None:
        """Emit workflow-specific metrics to the experiment logger."""

    def state_dict(self, *, mode: str = "checkpoint") -> Dict[str, Any]:
        """Persist workflow-managed state or provide derived projections.

        Args:
            mode: View to materialize. Defaults to "checkpoint" for full state
                restoration payloads. Other modes (e.g. "metrics") can return
                alternate projections while still using a dict contract.
        """
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore workflow-managed state."""

    # ------------------------------------------------------------------
    # Episode tracking utilities
    # ------------------------------------------------------------------
    def _reset_episode_tracking(
        self,
        num_envs: int,
        *,
        clear_history: bool = False,
        history: Optional[int] = None,
    ) -> None:
        """Reset accumulators used for per-environment episode accounting."""
        if history is not None:
            self._episode_history_len = max(int(history), 1)

        if not hasattr(self, "episode_returns") or clear_history:
            self.episode_returns = deque(maxlen=self._episode_history_len)
            self.episode_lengths = deque(maxlen=self._episode_history_len)
            if clear_history:
                self.total_episodes = 0
        else:
            if self.episode_returns.maxlen != self._episode_history_len:
                self.episode_returns = deque(self.episode_returns, maxlen=self._episode_history_len)
            if self.episode_lengths.maxlen != self._episode_history_len:
                self.episode_lengths = deque(self.episode_lengths, maxlen=self._episode_history_len)
            if clear_history:
                self.episode_returns.clear()
                self.episode_lengths.clear()
                self.total_episodes = 0

        self.num_envs = int(num_envs)
        self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def _update_episode_stats(
        self,
        rewards: Any,
        dones: Any,
        infos: Optional[Any] = None,
    ) -> None:
        """Accumulate per-environment episode statistics."""
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=bool)
        rewards_arr = np.atleast_1d(rewards_arr)
        dones_arr = np.atleast_1d(dones_arr)

        if rewards_arr.shape[0] != self.vector_episode_returns.shape[0]:
            raise ValueError(
                f"Reward batch dimension {rewards_arr.shape[0]} does not match tracking buffer "
                f"size {self.vector_episode_returns.shape[0]}."
            )

        self.vector_episode_returns += rewards_arr
        self.vector_episode_lengths += 1

        finished = np.nonzero(dones_arr)[0]
        for idx in finished:
            self.episode_returns.append(float(self.vector_episode_returns[idx]))
            self.episode_lengths.append(int(self.vector_episode_lengths[idx]))
            self.total_episodes += 1
            self.vector_episode_returns[idx] = 0.0
            self.vector_episode_lengths[idx] = 0

    def _snapshot_episode_tracking(self) -> Dict[str, Any]:
        """Return a serializable snapshot of episode tracking buffers."""
        return {
            "history_len": self._episode_history_len,
            "returns": deque(self.episode_returns, maxlen=self.episode_returns.maxlen),
            "lengths": deque(self.episode_lengths, maxlen=self.episode_lengths.maxlen),
            "total_episodes": self.total_episodes,
            "vector_returns": self.vector_episode_returns.copy(),
            "vector_lengths": self.vector_episode_lengths.copy(),
            "num_envs": self.num_envs,
        }

    def _restore_episode_tracking(self, state: Mapping[str, Any]) -> None:
        """Restore episode tracking buffers from a snapshot."""
        if not state:
            return

        history_len = int(state.get("history_len", self._episode_history_len))
        self._episode_history_len = max(history_len, 1)
        self.episode_returns = deque(
            state.get("returns", []), maxlen=self._episode_history_len
        )
        self.episode_lengths = deque(
            state.get("lengths", []), maxlen=self._episode_history_len
        )
        self.total_episodes = int(state.get("total_episodes", self.total_episodes))

        vector_returns = state.get("vector_returns")
        if vector_returns is not None:
            self.vector_episode_returns = np.asarray(vector_returns, dtype=np.float32).copy()
            self.num_envs = int(self.vector_episode_returns.shape[0])
        else:
            self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)

        vector_lengths = state.get("vector_lengths")
        if vector_lengths is not None:
            self.vector_episode_lengths = np.asarray(vector_lengths, dtype=np.int32).copy()
        else:
            self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)


from .context import WorkflowContext  # noqa: E402  (avoid circular import)
