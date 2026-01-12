"""Phase scheduler controlling workflow hook ordering for orchestrated runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import copy


DEFAULT_HOOKS: Dict[str, Tuple[str, ...]] = {
    "online": ("collect", "update_world_model", "update_controller"),
    "offline": ("update_world_model", "update_controller"),
    "eval_only": ("evaluate",),
}


@dataclass(frozen=True)
class PhaseDefinition:
    """Immutable phase configuration."""

    name: str
    type: str
    hooks: Tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)
    duration_steps: Optional[int] = None
    duration_updates: Optional[int] = None
    duration_cycles: Optional[int] = None

    def to_mapping(self, *, progress: dict[str, int] | None = None) -> dict[str, Any]:
          payload = dict(self.config)
          payload["name"] = self.name
          payload["type"] = self.type
          if self.duration_steps is not None:
              payload["duration_steps"] = self.duration_steps
          if self.duration_updates is not None:
              payload["duration_updates"] = self.duration_updates
          if self.duration_cycles is not None:
              payload["duration_cycles"] = self.duration_cycles
          if progress:
              payload.update(progress)  # e.g., steps_done/updates_done/cycles_done
          return payload


class PhaseScheduler:
    """Coordinates phase progression and hook scheduling."""

    def __init__(self, phases: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        if not phases:
            phases = [{"name": "online", "type": "online"}]

        self.phases: List[PhaseDefinition] = [self._build_phase(cfg) for cfg in phases]
        if not self.phases:
            raise ValueError("PhaseScheduler requires at least one phase definition.")

        self._current_index = 0
        self._pending_hooks: List[str] = []
        self._phase_steps = 0
        self._phase_updates = 0
        self._phase_cycles = 0
        self._finished = False

    def current_phase(self) -> PhaseDefinition:
        if self._finished:
            raise RuntimeError("PhaseScheduler has completed all phases.")
        return self.phases[self._current_index]

    def next_action(self) -> Optional[str]:
        if self._finished:
            return None
        if not self._pending_hooks:
            hooks = list(self.current_phase().hooks)
            self._pending_hooks = hooks
        if not self._pending_hooks:
            return None
        return self._pending_hooks[0]

    def advance(
        self,
        action: str,
        *,
        steps: int = 0,
        updates: int = 0,
    ) -> None:
        if self._finished:
            return

        if self._pending_hooks and self._pending_hooks[0] == action:
            self._pending_hooks.pop(0)

        if action == "collect":
            self._phase_steps += steps
        if action.startswith("update"):
            self._phase_updates += updates or 1
        if not self._pending_hooks:
            self._phase_cycles += 1

        phase = self.current_phase()
        if self._should_advance_phase(phase):
            self._next_phase()

    def is_finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing and progress tracking.

        Keys:
            current_index: Index of current phase
            pending_hooks: Remaining hooks in current cycle
            steps_done: Steps completed in current phase
            updates_done: Updates completed in current phase
            cycles_done: Cycles completed in current phase
            finished: Whether all phases are complete
        """
        return {
            "current_index": self._current_index,
            "pending_hooks": list(self._pending_hooks),
            "steps_done": self._phase_steps,
            "updates_done": self._phase_updates,
            "cycles_done": self._phase_cycles,
            "finished": self._finished,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        if not state:
            return
        self._current_index = state.get("current_index", 0)
        self._pending_hooks = list(state.get("pending_hooks", []))
        self._phase_steps = state.get("steps_done", 0)
        self._phase_updates = state.get("updates_done", 0)
        self._phase_cycles = state.get("cycles_done", 0)
        self._finished = state.get("finished", False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_phase(self, phase_cfg: Mapping[str, Any]) -> PhaseDefinition:
        if "type" not in phase_cfg:
            raise ValueError(f"Phase definition missing 'type': {phase_cfg}")
        name = str(phase_cfg.get("name", phase_cfg["type"]))
        phase_type = str(phase_cfg["type"])

        hooks: Tuple[str, ...]
        if "workflow_hooks" in phase_cfg and "hooks" not in phase_cfg:
            raw_hooks = phase_cfg["workflow_hooks"]
        else:
            raw_hooks = phase_cfg.get("hooks")

        if raw_hooks is not None:
            if isinstance(raw_hooks, str):
                hooks = (raw_hooks,)
            else:
                hooks = tuple(raw_hooks)
        else:
            hooks = DEFAULT_HOOKS.get(phase_type, DEFAULT_HOOKS["online"])

        duration_steps = self._coerce_int(phase_cfg, ("duration_steps", "steps"))
        duration_updates = self._coerce_int(phase_cfg, ("duration_updates", "updates"))
        duration_cycles = self._coerce_int(phase_cfg, ("duration_cycles", "cycles"))

        config_keys = {
            "name",
            "type",
            "hooks",
            "workflow_hooks",
            "duration_steps",
            "steps",
            "duration_updates",
            "updates",
            "duration_cycles",
            "cycles",
        }
        extra_config = {
            key: copy.deepcopy(value)
            for key, value in phase_cfg.items()
            if key not in config_keys
        }
        extra_config.setdefault("name", name)
        extra_config.setdefault("type", phase_type)

        return PhaseDefinition(
            name=name,
            type=phase_type,
            hooks=hooks,
            config=extra_config,
            duration_steps=duration_steps,
            duration_updates=duration_updates,
            duration_cycles=duration_cycles,
        )

    def _coerce_int(self, cfg: Mapping[str, Any], keys: Iterable[str]) -> Optional[int]:
        for key in keys:
            if key in cfg and cfg[key] is not None:
                return int(cfg[key])
        return None

    def _should_advance_phase(self, phase: PhaseDefinition) -> bool:
        if phase.duration_steps is not None and self._phase_steps >= phase.duration_steps:
            return True
        if phase.duration_updates is not None and self._phase_updates >= phase.duration_updates:
            return True
        if phase.duration_cycles is not None and self._phase_cycles >= phase.duration_cycles:
            return True
        return False

    def _next_phase(self) -> None:
        self._current_index += 1
        if self._current_index >= len(self.phases):
            self._finished = True
            return

        self._pending_hooks = []
        self._phase_steps = 0
        self._phase_updates = 0
        self._phase_cycles = 0
