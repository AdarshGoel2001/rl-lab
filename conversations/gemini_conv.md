# Final Architecture Spec: Checkpoint & Resume System

## 1. CheckpointManager: Minimal Responsibility
**Not redundant, but simplified.**
**Responsibility:** Pure persistence layer.
-   **Input:** `Dict[str, Any]` (the full state payload), `step`, `path`.
-   **Action:** Atomic write to disk (temp file -> rename), symlink management (`latest.pt`, `best.pt`), housekeeping (delete old ckpts).
-   **Output:** `Dict[str, Any]` (loaded payload).
-   **Removed:** All `hasattr` checks, state extraction logic, and object introspection.

## 2. Checkpoint Data Contract (`.pt` file structure)
```python
{
    "metadata": {
        "step": int,              # Global orchestrator step
        "timestamp": str,
        "run_id": str,
        "config_summary": Dict    # Optional: For validation, not restoration
    },
    "components": {               # Sourced from WorldModelComponents
        "encoder": OrderedDict,   # nn.Module state_dict
        "rssm": OrderedDict,
        "actor": OrderedDict,
        ...                       # Automatically discovered modules
    },
    "optimizers": {               # Sourced from WorldModelComponents
        "world_model": Dict,      # Optimizer state_dict
        "actor": Dict,
        ...
    },
    "workflow": {                 # Sourced from Workflow.get_state()
        "episodes": int,          # Algorithm counters
        "custom_metric": Any      # Researcher defined
    },
    "buffers": {                  # Sourced from Buffers
        "replay_buffer": Dict     # Pointers, metadata (data if Disk)
    }
}
```

## 3. Save/Load Sequence

### Save Sequence (Orchestrator-Driven)
1.  **Orchestrator** decides to save (e.g., `step % freq == 0`).
2.  **Orchestrator** calls `comp_state = self.context.components.state_dict()`.
    *   `WorldModelComponents` iterates its registry.
    *   Collects `state_dict()` for all `nn.Module` and `Optimizer`.
    *   Collects `get_state()` for any `StatefulComponent` (Protocol).
3.  **Orchestrator** calls `flow_state = self.workflow.get_state()`.
    *   `Workflow` returns dict of counters/custom vars.
4.  **Orchestrator** calls `buf_state = {k: v.save_checkpoint() for k,v in self.buffers.items()}`.
5.  **Orchestrator** assembles `payload = {metadata, components, workflow, buffers}`.
6.  **Orchestrator** delegates to `CheckpointManager.save_checkpoint(payload, path)`.

### Load Sequence (Orchestrator-Driven)
1.  **Orchestrator** delegates to `payload = CheckpointManager.load_checkpoint(path)`.
2.  **Orchestrator** extracts `step = payload['metadata']['step']`.
3.  **Orchestrator** calls `self.context.components.load_state_dict(payload['components'], payload['optimizers'])`.
    *   `WorldModelComponents` loads module weights (`strict=True`).
    *   `WorldModelComponents` loads optimizer states.
4.  **Orchestrator** (or Context) re-initializes **Schedulers** using `step` as `last_epoch`.
5.  **Orchestrator** calls `self.workflow.set_state(payload['workflow'])`.
6.  **Orchestrator** calls `self.buffers[k].load_checkpoint(payload['buffers'][k])`.

## 4. Refactor Plan

1.  **`src/utils/checkpoint.py`**:
    *   Strip `get_state`/`hasattr` logic.
    *   Make `save_checkpoint` accept a single `state_dict`.

2.  **`src/workflows/world_models/context.py`** (`WorldModelComponents`):
    *   Add `state_dict()`: Loop `self.components` + `self.optimizers` -> return Dict.
    *   Add `load_state_dict(state, optimizers_state)`: Restore modules and optimizers.
    *   Define `StatefulComponent` Protocol here.

3.  **`src/workflows/world_models/base.py`** (`WorldModelWorkflow`):
    *   Clean up `state_dict`: Remove optimizer handling (moved to Components).
    *   Focus `state_dict` purely on `total_episodes`, `episode_returns`.

4.  **`src/orchestration/world_model_orchestrator.py`**:
    *   Update `_save_checkpoint` to assemble the payload as described.
    *   Update `resume_from_checkpoint` (or `_load_checkpoint`) to follow the Load Sequence.
    *   Implement Scheduler re-init logic after load.