Approach
- Define a strict checkpoint protocol in `src/workflows/world_models/base.py` (workflow-level `get_state()`/`set_state()` returning a stable schema) and have `DreamerWorkflow` implement loop-based collection over components/controllers/optimizers.
- Make `CheckpointManager` a dumb serializer: it should only accept already-materialized checkpoint dicts and validate required keys/types (no `hasattr` or `get_state` calls).
- Store counters where they belong: `global_step` in orchestrator/top-level checkpoint data; `world_model_updates`/`total_episodes` (and episode tracking) inside workflow custom state.
- Override optimizer LR by snapshotting current LRs before `load_state_dict`, then restoring those LRs after load to keep momentum buffers intact.

Reasoning
1) world_models/base.py protocol + dreamer loop-based state
- Add a runtime-checkable protocol:
  - `Checkpointable` with `get_state() -> Dict[str, Any]` and `set_state(state: Mapping[str, Any]) -> None`.
- In `WorldModelWorkflow`, define abstract `get_state()`/`set_state()` and document a required schema, e.g.:
  - `{"modules": Dict[str, StateDict], "controllers": Dict[str, StateDict], "optimizers": Dict[str, StateDict], "custom": Dict[str, Any]}`.
  - `custom` should include counters and episode tracking (`_snapshot_episode_tracking()`), not config.
- Keep `state_dict(mode="checkpoint")` as a compatibility wrapper that calls `get_state()` for `checkpoint` mode and `_metrics_state()` for `metrics` mode (if needed).

Dreamer changes (`src/workflows/world_models/dreamer.py`)
- Cache `self._components` (WorldModelComponents) and `self._optimizers` (dict) when binding context.
- Implement `get_state()`:
  - `modules`: call `self._components.state_dict()` (looped, automatic).
  - `controllers`: iterate `self.controller_manager.items()` and call `controller.state_dict()` for each controller (require `torch.nn.Module` or raise if not).
  - `optimizers`: iterate `self._optimizers.items()` and call `optimizer.state_dict()`.
  - `custom`: include `world_model_updates` and `episode_tracking` (from `_snapshot_episode_tracking()`), plus any other researcher-defined state.
- Implement `set_state(state)`:
  - `self._components.load_state_dict(state["modules"])`
  - Load controller state dicts.
  - Load optimizers with LR override (see #4).
  - Restore `world_model_updates` and episode tracking via `_restore_episode_tracking()`.

WorldModelComponents loop discovery (`src/workflows/world_models/context.py`)
- Add `state_dict()` and `load_state_dict()`:
  - `state_dict()` loops `components` and calls `module.state_dict()` for `torch.nn.Module` (skip `None`, raise on non-module to enforce strictness).
  - `load_state_dict()` does the inverse; raise if an expected component is missing or not a module.
- This avoids any hardcoded component names and ensures new modules are automatically persisted.

2) checkpoint.py strict input handling (no silent failures)
- Change `CheckpointManager.save_checkpoint()` to accept a fully materialized `checkpoint_state` dict and validate it. Suggested shape:
  - `{"workflow_state": {...}, "buffers": {...}, "controllers": {...}, "metrics": {...}, "global_step": int}` or simply rely on `step` argument and require `workflow_state` + `buffers`.
- Remove all `get_state()`/`set_state()`/`hasattr` logic from `CheckpointManager`. It should never infer state from objects.
- Add explicit validation with clear errors:
  - Missing required keys -> `KeyError`.
  - Wrong types (e.g., `workflow_state` not dict) -> `TypeError`.
- Optional: add `schema_version` to checkpoint data to make future migrations explicit.
- `restore_training_state()` can be removed or rewritten to accept the new schema; orchestrator should handle restoration directly.

3) Where counters live + restore flow
- `global_step`: top-level checkpoint (`checkpoint_data["step"]` or `global_step`) handled by orchestrator.
- `world_model_updates`, `total_episodes`, and episode buffers: inside `workflow_state["custom"]`.
  - `episode_tracking` from `_snapshot_episode_tracking()` already includes `total_episodes`, vector counters, etc.
- Restore flow in orchestrator (new method `restore_from_checkpoint()` or in `initialize()`):
  1) `ckpt = checkpoint_manager.load_checkpoint(path)`
  2) `workflow.set_state(ckpt["workflow_state"])`
  3) for each buffer in `buffers`: `buffer.load_checkpoint(ckpt["buffers"][name])`
  4) set `self.global_step = ckpt["step"]` and update context `global_step`.
- This keeps counters with the component that owns them and avoids duplicating `world_model_updates` at top-level.

4) Override optimizer LR while keeping momentum
- For each optimizer:
  1) Snapshot current lrs: `desired_lrs = [g["lr"] for g in optimizer.param_groups]`.
  2) `optimizer.load_state_dict(saved_state)` (restores momentum buffers).
  3) Reapply desired lrs: `for group, lr in zip(optimizer.param_groups, desired_lrs): group["lr"] = lr`.
- This avoids persisting config values in checkpoints while preserving optimizer warm-up/momentum.

Per-file change list (high level)
- `src/workflows/world_models/base.py`
  - Add `Checkpointable` protocol + abstract `get_state`/`set_state` in `WorldModelWorkflow`.
  - Optionally, make `state_dict(mode="checkpoint")` delegate to `get_state()` for backward compatibility.
- `src/workflows/world_models/context.py`
  - Add `WorldModelComponents.state_dict()` and `.load_state_dict()` with strict module validation.
- `src/workflows/world_models/dreamer.py`
  - Store `self._components`/`self._optimizers` in `_bind_context`.
  - Implement loop-based `get_state()`/`set_state()` as described; make `state_dict()` call `get_state()`.
- `src/orchestration/world_model_orchestrator.py`
  - Update `_save_checkpoint()` to pass `workflow_state=workflow.get_state()` and buffer/controller states.
  - Add `restore_from_checkpoint(path)` to load and set workflow/buffers/global_step.
  - Ensure context global_step is updated after restore.
- `src/utils/checkpoint.py`
  - Make `save_checkpoint()` accept pre-validated dict, remove `get_state()`/`hasattr` logic.
  - Validate required keys and types to fail loudly on schema mismatches.
  - (Optional) add `schema_version`.
- `scripts/train.py`
  - Add config option (e.g., `training.resume_path` or `experiment.resume_path`) and call orchestrator restore before training.

Tests / safety checks
- Update `tests/test_end_to_end_training.py` to expect `workflow_state` (or updated schema) and verify module weights present:
  - e.g., assert `"modules" in checkpoint_data["workflow_state"]` and `"encoder"` exists.
- Add a test for optimizer LR override:
  - Save checkpoint, change config lr, reload, assert `param_group["lr"]` matches new config but momentum buffers (`exp_avg`) remain.
- Add a test for counters restore:
  - Save with non-zero `global_step/world_model_updates/total_episodes`, reload, assert restored values.
- Add a validation test to ensure `CheckpointManager` raises on missing `workflow_state` or wrong types.

Questions for others
- Do we want controllers persisted inside `workflow_state["controllers"]` or keep them as a top-level checkpoint key (and why)?
- Should `WorldModelComponents.state_dict()` skip non-module components (strict raise) or allow explicit `Checkpointable` objects in `components`?
- Is there a preferred key naming (`workflow` vs `workflow_state`) to keep compatibility with existing checkpoints/tests?
- Should we version the checkpoint schema now (e.g., `schema_version: 2`) to formalize this change?

Confidence
- High: root cause and need to remove `CheckpointManager` introspection; loop-based module save/load is correct.
- Medium: controller persistence strategy (top-level vs inside workflow state) and strictness for non-module components.
- Medium-low: where to place resume handling (train script vs orchestrator) without breaking existing configs.
