# Analysis: Checkpoint/Resume System

## Approach
I propose a strict **Extract-then-Save** architecture. The `Workflow` and `Buffer` objects are responsible for serializing their internal state into dictionaries (Extraction). The `Orchestrator` coordinates this extraction. The `CheckpointManager` is reduced to a "dumb" persistence layer that writes the aggregated dictionary to disk. This resolves the interface mismatch by ensuring `CheckpointManager` always receives serializable data, never objects.

## Reasoning
1.  **Truth**: The current `CheckpointManager` assumes it can call `.get_state()` on values passed to it. The `Orchestrator` passes dictionaries. This is a hard type error.
2.  **Simplicity**: Instead of making `CheckpointManager` handle both Objects and Dicts (using `hasattr`), we standardize on `Orchestrator` performing the extraction. This adheres to "Strict interface, not conditional".
3.  **Discovery**: By enforcing that `Workflow` iterates its components, we satisfy the "Loop-based discovery" constraint without hardcoding component names in the save logic.

## Data Contract & Sequence

### 1. Checkpoint Data Structure (The Contract)
The `.pt` file will contain a single top-level dictionary:

```python
{
    "metadata": {
        "step": int,
        "timestamp": str,
        "version": "1.0"
    },
    "workflow": {  # Derived from workflow.get_state()
        "components": {
            "encoder": OrderedDict(...),         # nn.Module state_dict
            "representation_learner": OrderedDict(...),
            "dynamics_model": OrderedDict(...),
            "actor": OrderedDict(...),           # Controllers are components
            "critic": OrderedDict(...)
        },
        "optimizers": {
            "world_model_optimizer": dict(...),  # Optimizer state_dict
            "actor_optimizer": dict(...),
            "critic_optimizer": dict(...)
        },
        "custom": {
            "world_model_updates": int,
            "total_episodes": int
            # ... researcher defined fields
        }
    },
    "buffers": {
        "replay_buffer": {
            "write_pointer": int,
            "size": int,
            # If DiskBuffer, path/metadata. If RAM, potentially data (but prompt warns against incomplete)
        }
    }
}
```

### 2. Interfaces

**Workflow (`src/workflows/world_models/base.py`):**
```python
class WorldModelWorkflow(abc.ABC):
    def get_state(self) -> Dict[str, Any]:
        """
        Aggregates state from 3 sources:
        1. Components (nn.Modules): Discovered via loop over self.components
        2. Optimizers: Discovered via loop over self.optimizers
        3. Custom: via self.get_custom_state()
        """
        # Implementation logic (pseudo-code)
        # return {
        #    'components': {k: v.state_dict() for k, v in self.components.items() if isinstance(v, nn.Module)},
        #    'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
        #    'custom': self.get_custom_state()
        # }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restores state:
        1. Components: load_state_dict(strict=True)
        2. Optimizers: load_state_dict(), then force LR from Config
        3. Custom: self.set_custom_state()
        """
        
    def get_custom_state(self) -> Dict[str, Any]:
        # Default implementation returns empty or basic stats
        return {'world_model_updates': self.world_model_updates}

    def set_custom_state(self, state: Dict[str, Any]) -> None:
        # Restore basic stats
        pass
```

### 3. Load/Save Sequence

**Save Sequence:**
1.  **Orchestrator** determines a checkpoint is needed (e.g. `step % freq == 0`).
2.  **Orchestrator** calls `state_payload = self.workflow.get_state()`.
3.  **Orchestrator** calls `buffer_payload = {k: b.save_checkpoint() for k, b in self.buffers.items()}`.
4.  **Orchestrator** combines them: `full_state = {'workflow': state_payload, 'buffers': buffer_payload}`.
5.  **Orchestrator** calls `self.checkpoint_manager.save_checkpoint(full_state, step)`.
6.  **CheckpointManager** writes `full_state` to disk using `torch.save`.

**Load Sequence:**
1.  **Orchestrator** calls `checkpoint = self.checkpoint_manager.load_checkpoint(path)`.
2.  **Orchestrator** extracts `workflow_state = checkpoint['workflow']`.
3.  **Orchestrator** calls `self.workflow.set_state(workflow_state)`.
    *   `Workflow` restores NN weights.
    *   `Workflow` restores Optimizers, **then overrides LR from current Config**.
    *   `Workflow` restores custom counters.
4.  **Orchestrator** calls `self.buffers[name].load_checkpoint(checkpoint['buffers'][name])`.

## Answers to Questions

1.  **Checkpoint Dict**: See "Data Contract" above. Top-level keys: `metadata`, `workflow`, `buffers`.
2.  **Loop Discovery**:
    *   In `DreamerWorkflow.initialize`, store `self.components = context.components` (the `WorldModelComponents` container).
    *   In `get_state`, iterate `self.components.as_dict().items()`.
    *   Filter: `if isinstance(obj, torch.nn.Module)`.
    *   This ensures configs are ignored (unless they accidentally masquerade as Modules) and new modules added to the container are automatically picked up.
3.  **Missing/Extra Keys**:
    *   **Workflow Logic**: Iterate the *checkpoint's* keys. If a key exists in `self.components`, load it.
    *   **Safety**: Use `strict=True` for the `load_state_dict` of individual modules to catch architecture mismatches (layer size changes).
    *   **Architecture Evolution**: If the checkpoint has a component `foo` that no longer exists in code, log a warning and ignore. If code has `bar` not in checkpoint, log a warning and leave initialized (random).
4.  **Buffer Pitfalls**:
    *   **Incomplete Episodes**: Buffers must store a `write_pointer` that strictly points to the end of the last *terminated* episode. When saving, only valid episodes up to that pointer are "valid".
    *   **Resumption**: On load, if the buffer finds data beyond the saved `write_pointer` (e.g. if using a persistent memory-mapped file that wasn't truncated), it must reset its internal pointer to the checkpointed value to ensure the partial episode is discarded/overwritten.

## Confidence
High. The proposed separation of concerns aligns with the "Ideal System Design" and solves the root cause (interface mismatch) by standardizing on Data Transfer Objects (Dictionaries) rather than active Objects at the CheckpointManager boundary.
