# Opus: Final Checkpoint/Resume Spec

---

## 1. Checkpoint Schema

```python
{
    "version": 1,
    "global_step": int,                          # from: orchestrator.global_step

    "phase_state": {                             # from: scheduler internal state
        "current_index": int,
        "pending_hooks": List[str],
        "phase_steps": int,
        "phase_updates": int,
        "phase_cycles": int,
        "finished": bool,
    },

    "components": {                              # from: orchestrator loops context.components.components
        "vae": OrderedDict,                      #   component.state_dict()
        "dynamics_model": OrderedDict,
        # ... all nn.Module components
    },

    "controllers": {                             # from: controller_manager.state_dict()
        "actor": {
            "weights": OrderedDict,
            "optimizer": dict,
        },
        # ... all controllers
    },

    "optimizers": {                              # from: orchestrator loops context.optimizers
        "world_model": dict,                     #   optimizer.state_dict()
    },

    "workflow_custom": {                         # from: workflow.get_state()
        "total_episodes": int,
        "episode_returns": List[float],
        "episode_lengths": List[int],
        "vector_episode_returns": ndarray,
        "vector_episode_lengths": ndarray,
    },

    "buffers": {                                 # from: buffer.save_checkpoint() per buffer
        "replay": dict,
    },

    "rng_states": {                              # from: checkpoint_manager (kept)
        "numpy": tuple,
        "python": tuple,
        "torch_cpu": Tensor,
        "torch_cuda": List[Tensor],
    },
}
```

---

## 2. Save Sequence (Orchestrator)

```python
def _save_checkpoint(self, *, final: bool = False) -> None:
    context = self.ensure_context()

    # 1. Collect component states (loop-based)
    components_state = {}
    for name, component in context.components.components.items():
        if component is not None and isinstance(component, torch.nn.Module):
            components_state[name] = component.state_dict()

    # 2. Collect optimizer states (loop-based)
    optimizers_state = {}
    for name, optimizer in (context.optimizers or {}).items():
        if optimizer is not None:
            optimizers_state[name] = optimizer.state_dict()

    # 3. Collect controller states
    controllers_state = {}
    if self.controller_manager is not None:
        controllers_state = self.controller_manager.state_dict()

    # 4. Collect phase scheduler state
    phase_state = {
        "current_index": self.scheduler._current_index,
        "pending_hooks": list(self.scheduler._pending_hooks),
        "phase_steps": self.scheduler._phase_steps,
        "phase_updates": self.scheduler._phase_updates,
        "phase_cycles": self.scheduler._phase_cycles,
        "finished": self.scheduler._finished,
    }

    # 5. Collect workflow custom state
    workflow_custom = self.workflow.get_state()

    # 6. Collect buffer states
    buffers_state = {}
    for name, buffer in self.buffers.items():
        if hasattr(buffer, "save_checkpoint"):
            buffers_state[name] = buffer.save_checkpoint()

    # 7. Assemble checkpoint
    state = {
        "version": 1,
        "global_step": self.global_step,
        "phase_state": phase_state,
        "components": components_state,
        "controllers": controllers_state,
        "optimizers": optimizers_state,
        "workflow_custom": workflow_custom,
        "buffers": buffers_state,
    }

    # 8. Delegate I/O to checkpoint manager
    name = "final" if final else f"step_{self.global_step}"
    self.checkpoint_manager.save(state, self.global_step, name=name)
```

---

## 3. Load Sequence (Orchestrator)

```python
def _load_checkpoint(self, path: Path) -> None:
    # 1. Load raw checkpoint
    ckpt = self.checkpoint_manager.load(path)
    if ckpt is None:
        return

    context = self.ensure_context()

    # 2. Restore global step
    self.global_step = ckpt.get("global_step", 0)

    # 3. Restore phase scheduler state
    phase_state = ckpt.get("phase_state", {})
    self.scheduler._current_index = phase_state.get("current_index", 0)
    self.scheduler._pending_hooks = list(phase_state.get("pending_hooks", []))
    self.scheduler._phase_steps = phase_state.get("phase_steps", 0)
    self.scheduler._phase_updates = phase_state.get("phase_updates", 0)
    self.scheduler._phase_cycles = phase_state.get("phase_cycles", 0)
    self.scheduler._finished = phase_state.get("finished", False)

    # 4. Restore component weights (loop-based)
    components_state = ckpt.get("components", {})
    for name, state_dict in components_state.items():
        component = context.components.components.get(name)
        if component is not None and isinstance(component, torch.nn.Module):
            component.load_state_dict(state_dict)

    # 5. Restore optimizer states with LR override
    optimizers_state = ckpt.get("optimizers", {})
    for name, opt_state in optimizers_state.items():
        optimizer = (context.optimizers or {}).get(name)
        if optimizer is not None:
            optimizer.load_state_dict(opt_state)
            # Override LR from config
            lr_key = f"{name}_lr"
            config_lr = getattr(self.config.algorithm, lr_key, None)
            if config_lr is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = float(config_lr)

    # 6. Restore controller states
    controllers_state = ckpt.get("controllers", {})
    if self.controller_manager is not None and controllers_state:
        self.controller_manager.load_state_dict(controllers_state)

    # 7. Restore workflow custom state
    workflow_custom = ckpt.get("workflow_custom", {})
    self.workflow.set_state(workflow_custom)

    # 8. Restore buffer states
    buffers_state = ckpt.get("buffers", {})
    for name, buf_state in buffers_state.items():
        buffer = self.buffers.get(name)
        if buffer is not None and hasattr(buffer, "load_checkpoint"):
            buffer.load_checkpoint(buf_state)

    # 9. Update context global_step
    self._update_context(global_step=self.global_step)

    # 10. LR scheduler re-init (if applicable)
    # Scheduler is built fresh from config; set last_epoch to global_step
    # (handled externally in train.py if scheduler exists)
```

---

## 4. Per-File Edit List

| File | Changes |
|------|---------|
| **`src/orchestration/world_model_orchestrator.py`** | Rewrite `_save_checkpoint()` per save sequence above. Add `_load_checkpoint(path)` per load sequence above. Remove current `_buffer_checkpoint()` helper (inline the logic). |
| **`src/workflows/world_models/base.py`** | Add `def get_state(self) -> Dict[str, Any]: return {}`. Add `def set_state(self, state: Dict[str, Any]) -> None: pass`. Remove existing `state_dict(mode=)` and `load_state_dict()`. |
| **`src/workflows/world_models/og_wm.py`** | Implement `get_state()`: return `{"total_episodes": self.total_episodes, "episode_returns": list(self.episode_returns), "episode_lengths": list(self.episode_lengths), "vector_episode_returns": self.vector_episode_returns.tolist(), "vector_episode_lengths": self.vector_episode_lengths.tolist()}`. Implement `set_state()`: restore those fields using `_restore_episode_tracking()`. |
| **`src/utils/checkpoint.py`** | Keep: `save(state, step, name)` (torch.save + symlink + cleanup), `load(path)` (torch.load), `get_latest_path()`, RNG helpers. Remove: `save_checkpoint()`, `restore_training_state()`, `auto_save()`, all `.get_state()` calls. |
| **`scripts/train.py`** | Add `--resume` CLI arg (or `training.resume_path` config). After orchestrator init, before `run()`: `if resume_path: orchestrator._load_checkpoint(resume_path)`. |

---

## 5. Interface Summary

```python
# Workflow (base.py)
class WorldModelWorkflow:
    def get_state(self) -> Dict[str, Any]: ...      # researcher custom state only
    def set_state(self, state: Dict[str, Any]): ... # restore custom state

# CheckpointManager (checkpoint.py)
class CheckpointManager:
    def save(self, state: Dict, step: int, name: str = None): ...  # pure I/O
    def load(self, path: Path) -> Dict: ...                         # pure I/O
    def get_latest_path(self) -> Optional[Path]: ...                # resolve symlink

# ControllerManager (controllers.py) - unchanged
class ControllerManager:
    def state_dict(self) -> Dict[str, Any]: ...       # existing
    def load_state_dict(self, state: Dict): ...       # existing

# PhaseScheduler (phase_scheduler.py) - no new methods
# Orchestrator directly reads/writes scheduler internal state
```
