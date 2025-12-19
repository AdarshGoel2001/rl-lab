# Opus Analysis: Checkpoint/Resume System Design

## Summary of Current State

After reading the codebase, I've identified the exact failure points:

**Problem 1: Interface Mismatch**
- `CheckpointManager.save_checkpoint()` (checkpoint.py:99-100) calls `.get_state()` on passed objects
- `WorldModelOrchestrator._save_checkpoint()` (orchestrator.py:589) passes raw dicts, not objects
- Result: Silent failure - dicts don't have `.get_state()`, so empty data saved

**Problem 2: Missing Neural Network Weights**
- `DreamerWorkflow.state_dict()` (dreamer.py:611-624) only saves:
  - `world_model_optimizer` state
  - `world_model_updates` counter
  - Controller states via `controller_manager.state_dict()`
- **NOT saved**: encoder, rssm, dynamics_model, reward_predictor, observation_decoder

**Problem 3: Key Name Inconsistency**
- Orchestrator uses key `"workflow"`
- CheckpointManager expects `"algorithm"`

---

## Proposed Design

### 1. Interface Definition

I propose a clear Protocol-based interface in `src/workflows/world_models/base.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for components that support checkpointing."""

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Return all persistable state: PyTorch modules, optimizers, custom state."""
        ...

    def load_checkpoint_state(
        self,
        state: Dict[str, Any],
        config: Optional[Config] = None
    ) -> None:
        """Restore state, applying config overrides (e.g., LR) where applicable."""
        ...
```

**Why `get_checkpoint_state` instead of `get_state`?**
- Explicit naming avoids confusion with PyTorch's `state_dict()`
- Makes it clear this is the full checkpoint bundle

**Return Shape:**
```python
{
    "modules": {           # Loop-discovered PyTorch nn.Module state_dicts
        "encoder": {...},
        "representation_learner": {...},
        ...
    },
    "optimizers": {        # Optimizer state_dicts (momentum preserved)
        "world_model": {...},
        "actor": {...},
        ...
    },
    "custom": {            # Researcher-defined state
        "world_model_updates": 1234,
        "total_episodes": 567,
        ...
    }
}
```

### 2. Workflow State Implementation

In `WorldModelWorkflow` base class:

```python
def get_checkpoint_state(self) -> Dict[str, Any]:
    """Default implementation that uses loop-based discovery."""
    state = {
        "modules": {},
        "optimizers": {},
        "custom": self._get_custom_state(),  # Subclass overrides this
    }

    # Loop-based discovery of PyTorch modules
    for name, module in self._iter_modules():
        if module is not None and hasattr(module, 'state_dict'):
            state["modules"][name] = module.state_dict()

    # Loop-based discovery of optimizers
    for name, optimizer in self._iter_optimizers():
        if optimizer is not None and hasattr(optimizer, 'state_dict'):
            state["optimizers"][name] = optimizer.state_dict()

    return state

def _iter_modules(self) -> Iterator[Tuple[str, nn.Module]]:
    """Yield (name, module) pairs. Subclasses override for custom iteration."""
    # Default: use context.components
    if hasattr(self, 'context') and self.context.components:
        for name, comp in self.context.components.components.items():
            if isinstance(comp, nn.Module):
                yield name, comp

    # Also yield controllers
    if hasattr(self, 'controller_manager'):
        for role, ctrl in self.controller_manager.items():
            if isinstance(ctrl, nn.Module):
                yield f"controller_{role}", ctrl

def _iter_optimizers(self) -> Iterator[Tuple[str, torch.optim.Optimizer]]:
    """Yield (name, optimizer) pairs."""
    if hasattr(self, 'context') and self.context.optimizers:
        for name, opt in self.context.optimizers.items():
            yield name, opt
    # Controller optimizers
    if hasattr(self, 'controller_manager'):
        for role, ctrl in self.controller_manager.items():
            if hasattr(ctrl, 'optimizer'):
                yield f"{role}_optimizer", ctrl.optimizer

def _get_custom_state(self) -> Dict[str, Any]:
    """Override in subclass to persist researcher-defined state."""
    return {}

def _set_custom_state(self, state: Dict[str, Any]) -> None:
    """Override in subclass to restore researcher-defined state."""
    pass
```

**Key insight**: The workflow holds references to all components via `self.encoder`, `self.rssm`, etc. We iterate these rather than hardcoding.

### 3. Loading with LR Override

```python
def load_checkpoint_state(
    self,
    state: Dict[str, Any],
    config: Optional[Config] = None
) -> None:
    # Restore modules
    for name, module_state in state.get("modules", {}).items():
        module = self._get_module_by_name(name)
        if module is not None:
            module.load_state_dict(module_state)

    # Restore optimizers, then override LR from config
    for name, opt_state in state.get("optimizers", {}).items():
        optimizer = self._get_optimizer_by_name(name)
        if optimizer is not None:
            optimizer.load_state_dict(opt_state)
            if config is not None:
                self._apply_lr_override(optimizer, name, config)

    # Restore custom state
    self._set_custom_state(state.get("custom", {}))

def _apply_lr_override(
    self,
    optimizer: torch.optim.Optimizer,
    name: str,
    config: Config
) -> None:
    """Apply learning rate from current config to all param groups."""
    lr_key = f"{name.replace('_optimizer', '')}_lr"
    lr = getattr(config.algorithm, lr_key, None)
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = float(lr)
```

### 4. CheckpointManager Changes

Simplify `CheckpointManager` to be a pure save/load layer with no business logic:

```python
def save_checkpoint(
    self,
    state: Dict[str, Any],  # Already-serialized state dict
    step: int,
    name: Optional[str] = None,
    is_best: bool = False
) -> Path:
    """Save pre-serialized state to disk."""
    checkpoint_data = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'state': state,  # Just store it directly
        'rng_states': self._get_rng_states(),
    }
    # ... save to disk
```

Remove all `.get_state()` calls from CheckpointManager. The orchestrator is responsible for calling workflow methods.

### 5. Orchestrator Changes

```python
def _save_checkpoint(self, *, final: bool = False) -> None:
    state = {
        "workflow": self.workflow.get_checkpoint_state(),
        "global_step": self.global_step,
        "buffers": {},
    }

    # Buffers use existing save_checkpoint() interface
    for name, buffer in self.buffers.items():
        if hasattr(buffer, 'save_checkpoint'):
            state["buffers"][name] = buffer.save_checkpoint()

    self.checkpoint_manager.save_checkpoint(
        state, self.global_step, name=("final" if final else None)
    )

def _load_checkpoint(self, path: Path) -> None:
    checkpoint = self.checkpoint_manager.load_checkpoint(path)
    if checkpoint is None:
        return

    self.global_step = checkpoint.get("global_step", 0)

    # Restore workflow with config for LR override
    self.workflow.load_checkpoint_state(
        checkpoint.get("workflow", {}),
        config=self.config
    )

    # Restore buffers
    for name, buf_state in checkpoint.get("buffers", {}).items():
        buffer = self.buffers.get(name)
        if buffer and hasattr(buffer, 'load_checkpoint'):
            buffer.load_checkpoint(buf_state)
```

### 6. DreamerWorkflow Concrete Implementation

```python
class DreamerWorkflow(WorldModelWorkflow):

    def _iter_modules(self):
        """Yield all world model components."""
        modules = [
            ("encoder", self.encoder),
            ("representation_learner", self.rssm),
            ("dynamics_model", self.dynamics_model),
            ("reward_predictor", self.reward_predictor),
            ("observation_decoder", self.observation_decoder),
        ]
        for name, module in modules:
            if module is not None:
                yield name, module

        # Controllers are also modules
        if self.actor_controller is not None:
            yield "actor", self.actor_controller
        if self.critic_controller is not None:
            yield "critic", self.critic_controller

    def _iter_optimizers(self):
        if self.world_model_optimizer is not None:
            yield "world_model", self.world_model_optimizer
        if self.actor_controller and hasattr(self.actor_controller, 'optimizer'):
            yield "actor", self.actor_controller.optimizer
        if self.critic_controller and hasattr(self.critic_controller, 'optimizer'):
            yield "critic", self.critic_controller.optimizer

    def _get_custom_state(self) -> Dict[str, Any]:
        return {
            "world_model_updates": self.world_model_updates,
            "total_episodes": self.total_episodes,
            # Episode tracking state
            "episode_returns": list(self.episode_returns),
            "episode_lengths": list(self.episode_lengths),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        self.world_model_updates = state.get("world_model_updates", 0)
        self.total_episodes = state.get("total_episodes", 0)
        # ... restore episode tracking
```

---

## Risks and Considerations

### Risk 1: Optimizer LR Override with Multiple Param Groups

**Concern:** Some architectures use different LRs for different param groups (e.g., lower LR for pretrained backbone).

**My position:** The proposed `_apply_lr_override()` applies LR to ALL param groups uniformly. This is intentional simplicity - the most common case. If a researcher needs per-group LR, they can:
1. Override `_apply_lr_override()` in their workflow subclass
2. Structure their config with per-group LR settings

**Alternative considered:** Store LR ratios during save, restore proportionally. Rejected - adds complexity for rare use case.

### Risk 2: Module Discovery vs Explicit Registration

**Concern:** Loop-based discovery might miss components or include unwanted ones.

**My position:** The `_iter_modules()` pattern gives subclasses full control. Base class provides sensible defaults, but researchers can override completely if needed.

### Risk 3: Version Compatibility

**Concern:** If component structure changes between versions, checkpoint loading fails.

**My position:** Add version info to checkpoint:
```python
checkpoint_data = {
    "version": 1,
    "workflow_class": self.workflow.__class__.__name__,
    ...
}
```
On load, validate compatibility or provide migration hooks.

---

## Files to Modify

1. **src/workflows/world_models/base.py**
   - Add `Checkpointable` protocol
   - Add default `get_checkpoint_state()` / `load_checkpoint_state()` to `WorldModelWorkflow`
   - Add `_iter_modules()`, `_iter_optimizers()`, `_get_custom_state()`, `_set_custom_state()`

2. **src/workflows/world_models/dreamer.py**
   - Override `_iter_modules()`, `_iter_optimizers()`
   - Implement `_get_custom_state()` / `_set_custom_state()`
   - Remove old `state_dict()` / `load_state_dict()` (or keep for backward compat)

3. **src/orchestration/world_model_orchestrator.py**
   - Modify `_save_checkpoint()` to call `workflow.get_checkpoint_state()`
   - Add `_load_checkpoint()` method for resume
   - Pass config to workflow during load for LR override

4. **src/utils/checkpoint.py**
   - Simplify to pure save/load - remove `.get_state()` calls
   - Accept pre-serialized dicts

5. **scripts/train.py**
   - Add `--resume_path` argument
   - Call orchestrator resume before `run()`

---

## Questions for Other Models

1. **On interface naming:** I chose `get_checkpoint_state()` / `load_checkpoint_state()`. Is there value in keeping compatibility with PyTorch's `state_dict()` naming convention? The downside is potential confusion with nn.Module.state_dict().

2. **On buffer handling:** The constraint says "discard incomplete episodes." Should this be enforced at checkpoint time (buffers don't include incomplete trajectories) or at save/load boundary (orchestrator filters)?

3. **On controller state:** Controllers have both weights and optimizers. Currently controllers are nn.Modules that own their optimizers. Should controllers implement `get_checkpoint_state()` separately, or should the workflow iterate their internals?

4. **On atomicity:** What happens if checkpoint save is interrupted? Should we write to temp file then rename, or is torch.save's behavior sufficient?

---

## Confidence Assessment

**High confidence:**
- Three-level state structure (modules/optimizers/custom) is correct
- Loop-based discovery is the right pattern
- LR override on load is straightforward
- CheckpointManager should be simplified to pure I/O

**Medium confidence:**
- Exact protocol/interface naming
- Whether `WorldModelComponents` should have its own state_dict() or just be iterated by workflow

**Lower confidence:**
- Buffer handling details - need to understand buffer implementation better
- Whether there are edge cases in controller optimizer discovery

---

## Summary

The fix requires:
1. New protocol interface: `get_checkpoint_state()` / `load_checkpoint_state()`
2. Loop-based discovery via `_iter_modules()` and `_iter_optimizers()`
3. Custom state via `_get_custom_state()` / `_set_custom_state()`
4. LR override applied after optimizer load
5. CheckpointManager simplified to pure save/load
6. Orchestrator coordinates calling the right methods

Total estimated scope: ~200-300 lines of new/modified code across 5 files.
