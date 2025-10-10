# Modular World-Model Training Framework Design

## Background

The current Dreamer-style implementation has grown into an ad-hoc trainer (`src/paradigms/world_models/trainer.py`) that simultaneously:

- owns experiment bootstrap (logging, checkpointing, resuming),
- drives environment interaction and buffer population,
- executes Dreamer-specific world-model / actor / critic updates, and
- hardcodes sequencing (collect → world model update → policy update).

Replay buffers (`src/buffers/world_model_sequence.py`) assume Dreamer rollouts, and controller components live inside the paradigm without a clear orchestration contract. Adding offline warm-up, planner-only phases, or alternative algorithms requires editing the single trainer file and reimplementing the entire loop.

## Goals

1. Decouple orchestration (scheduling, logging, checkpointing) from algorithm-specific training logic.
2. Introduce workflow interfaces so Dreamer, MuZero, TD-MPC, BC, or diffusion policies supply their own hooks.
3. Provide a unified data abstraction to mix online replay, demonstrations, or synthetic imagination batches.
4. Treat controllers (policies, planners, diffusion samplers) as configurable components with optional learning hooks.
5. Preserve current Dreamer behaviour while exposing extension points for future algorithms.

## Non-Goals

- Large-scale performance work (distributed training, accelerators).
- Shipping fully-featured planners or diffusion learners in this iteration.
- Replacing existing config/registry systems; extend them incrementally.
- Major CLI/UI changes beyond what new config blocks require.

## Current Pain Points

- **Trainer monolith** – WorldModelTrainer mixes environment control with Dreamer-specific losses and optimizers.
- **Rigid chronology** – Update order is embedded in the loop; offline or multi-phase schedules require code edits.
- **Coupled buffers** – Buffer schema assumes contiguous Dreamer sequences; MuZero-style n-step or BC batches need bespoke code.
- **Controller coupling** – Planners are attached directly to `WorldModelParadigm` with no orchestration contract.
- **Limited observability** – Metrics and checkpoints are tied to the trainer implementation, limiting reuse.

## Proposed Architecture

### System Overview

```
+--------------------+        +---------------------+
| WorldModelOrchestrator |--> | PhaseScheduler       |
+--------------------+        +----------+----------+
         |                                |
         v                                v
  WorkflowContext               Phase definitions (config)
         |
         v
+--------------------+        +---------------------+
| WorldModelWorkflow |<------>| DataSource adapters |
+--------------------+        +---------------------+
         |                                |
         v                                v
  Controllers / Simulators        ReplayBuffer / OfflineDataset
```

### Orchestrator

- New module `src/orchestration/world_model_orchestrator.py`.
- Responsibilities:
  - Experiment setup (logging, seeding, checkpointing).
  - Build `WorkflowContext` (environment(s), simulators, controllers, data sources, metrics/logging handles).
  - Drive the main loop by asking `PhaseScheduler` which hooks to call on the active workflow(s).
  - Persist workflow + component state via checkpoint manager.
  - Provide instrumentation (step timing, metrics aggregation) but never compute algorithm-specific losses.
- Orchestrator exposes `run()` returning final metrics while remaining algorithm-agnostic.

### Workflow Interface

- New package `src/workflows/world_models/`.
- `base.py` defines `WorldModelWorkflow` abstract class with required hooks:
  - `initialize(context: WorkflowContext) -> None`
  - `collect_step(step: int, *, phase: PhaseConfig) -> Optional[CollectResult]`
  - `update_world_model(batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]`
  - `update_controller(batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]`
  - `plan_phase(step: int) -> Optional[str]` (for dynamic phase switching)
  - `log_metrics(step: int, writer: MetricsLogger) -> None`
  - `state_dict()` / `load_state_dict()` for checkpoint integration.
- Dreamer-specific logic lives inside `DreamerWorkflow` (`dreamer.py`). Future workflows (MuZero, BC, diffusion) implement the same interface without touching orchestration.

### Workflow Context

- Located in `src/workflows/world_models/context.py`.
- Immutable dataclass bundling:
  - `config` (full Config object),
  - environment handles (`train_env`, `eval_env`),
  - component factory results (paradigm, controllers, optimizers),
  - registered data sources,
  - device, logger instances, checkpoint manager, simulator service,
  - utilities (grad clip helper, mixed precision manager).
- Context mediates access; workflows cannot instantiate new registries directly.

### Phase Scheduler

- New module `src/orchestration/phase_scheduler.py`.
- Reads `training.phases` configuration (ordered list) where each phase defines:
  - `name`
  - `type` (`online`, `offline`, `eval_only`, `planner_only`, `warmup`, etc.)
  - `duration` (timesteps or update count)
  - `workflow_hooks` (ordered list such as `collect`, `world_model`, `controller`)
  - optional `data_source` override or `controller_role`.
- Scheduler tracks progress per phase, emits the next hook invocation (e.g., `collect_step`, `update_world_model`). Supports looping, alternating patterns, and conditional transitions.
- Exposes `PhaseState` enabling workflows to inspect schedule metadata.

### Data Sources & Sampling Interface

- New package `src/data_sources/`.
- `base.py` defines `DataSource` with `prepare(context)`, `add(TransitionBatch | Trajectory)`, `sample(spec)`, `ready()`, and `state_dict` hooks.
- Adapters:
  - `ReplayDataSource` wrapping existing `WorldModelSequenceBuffer`.
  - `OfflineDatasetSource` reading saved tensor/npz/CSV demonstration sets.
  - Future `ImaginedBatchSource`, `VideoLatentSource`.
- `sample()` accepts a `BatchSpec` (batch size, sequence length, fields) enabling heterogenous retrieval.
- Orchestrator registers all configured sources and routes `collect_step` outputs to the appropriate source(s).

### Controller Integration

- Extend `ComponentFactory` to build controller instances keyed by roles (`"policy"`, `"planner"`, `"distill_target"`, etc.).
- New registry helper `get_controller(role, name)` optionally reusing existing `components.world_models.controllers`.
- Controllers implement `ControllerBase` with optional `learn()` hook invoked by workflows during `update_controller`.
- Config: `controllers: { policy: {...}, planner: {...} }`.
- `WorkflowContext` exposes a `ControllerManager` to fetch controllers by role and coordinate metrics/state dicts.

### Simulator Service

- `src/simulation/simulator_service.py`.
- Wrap existing world-model imagination utilities (`WorldModelParadigm.system.imagine`) plus planner-specific rollouts or tree search APIs.
- Workflows and controllers request imagined trajectories through this service, keeping orchestrator agnostic to Dreamer internals.

### Logging & Metrics

- Existing `experiment_logger` reused but moved into context.
- Workflows emit metrics via `MetricsSink` abstraction (writes to logger, stdout, progress bars).
- Orchestrator collects timing statistics (phase durations, throughput) and publishes generic metrics.

## Configuration Updates

Additive changes that retain backward compatibility:

```yaml
training:
  total_timesteps: 1_000_000
  phases:
    - name: offline_bc
      type: offline
      data_source: demos
      duration: 10000_updates
      workflow_hooks: [world_model, controller]
    - name: online
      type: online
      controller: policy
      duration: 1_000_000_steps
      workflow_hooks: [collect, world_model, controller, eval]

data_sources:
  replay:
    type: world_model_replay
    config:
      sequence_length: 16
  demos:
    type: offline_tensor
    config:
      path: data/cartpole_bc.pt

controllers:
  policy:
    type: dreamer_actor
    config: {...}
  planner:
    type: cross_entropy_mpc
    config: {...}
```

- Default config path continues to operate with implicit single `online` phase, `replay` data source, and Dreamer controller.
- `training.policy_warmup_updates` etc. migrate into phase definitions (via preset templates).

## Component Responsibility Matrix

| Component | Responsibilities | Key APIs |
|-----------|-----------------|---------|
| `WorldModelOrchestrator` | Experiment lifecycle, scheduling, logging, checkpointing | `run()`, `step_phase()` |
| `PhaseScheduler` | Manage phase chronology and emit workflow hook schedule | `next_action()`, `advance()` |
| `WorldModelWorkflow` | Algorithm-specific logic (Dreamer, MuZero, BC) | Hook methods |
| `WorkflowContext` | Shared resources for workflows | Accessors for envs, data, controllers |
| `DataSource` | Unified sampling and storage interface | `add()`, `sample()`, `ready()` |
| `ControllerManager` | Controller lookup and optional learning | `get(role)`, `learn()` |
| `SimulatorService` | Imagination/planning utilities | `rollout()`, `value_estimate()` |

## Implementation Roadmap

1. **Foundation (current sprint)**
   - Extract orchestration skeleton from `WorldModelTrainer` into `WorldModelOrchestrator`.
   - Implement `WorkflowContext` and `WorldModelWorkflow` base class.
   - Migrate Dreamer-specific logic into `DreamerWorkflow` while keeping behaviour identical.
   - Wire orchestrator to use existing replay buffer via a thin `ReplayDataSource`.
   - Provide compatibility adapter so legacy config path instantiates orchestrator + Dreamer workflow automatically.

2. **Data & Phases**
   - Implement `PhaseScheduler` with support for fixed ordered phases and warm-up counters.
   - Refactor replay buffer into `ReplayDataSource` (compose rather than inherit).
   - Add `OfflineDatasetSource` that loads tensors from disk at start-up.
   - Update configs + docs with new phase syntax (with backward-compatible defaults).

3. **Controllers & Simulation**
   - Extend `ComponentFactory` to register multiple controllers by role.
   - Implement `ControllerManager` + hooks for learning.
   - Introduce `SimulatorService` bridging workflows and imagination utilities.

4. **Testing & Migration**
   - Snapshot Dreamer regression tests (phase scheduling, checkpoint resume, metrics parity).
   - Add unit tests: `tests/orchestration/test_phase_scheduler.py`, `tests/workflows/test_dreamer_workflow.py`, `tests/data_sources/test_replay_source.py`.
   - Update documentation (this design doc, README, config examples).
   - Deprecate legacy `WorldModelTrainer` after tests green and configs updated.

5. **Extension (post-MVP)**
   - Implement `BehaviorCloningWorkflow` demonstrating offline training path.
   - Scaffold planner/diffusion workflow placeholders to validate interface.
   - Expand simulator with planner-specific utilities (CEM, tree search).

## Testing Strategy

- **Unit tests** for scheduler transitions, workflow hook invocation order, data source sampling semantics.
- **Integration tests** running short Dreamer training loops to verify metric parity with legacy trainer.
- **Checkpoint tests** ensuring orchestrator + workflows resume correctly.
- **Config validation tests** verifying schema defaults and backwards-compatible behaviour when `training.phases` is omitted.

## Migration Plan

1. Introduce orchestrator and new workflow path behind feature flag (`config.training.use_orchestrator`, default `False`).
2. Mirror existing Dreamer configs with new schema (`configs/experiments/dreamer_cartpole.yaml` -> add `training.phases`).
3. Once parity tests pass, flip default to orchestrator-driven path and mark legacy trainer deprecated.
4. Remove old trainer after one release cycle, leaving compatibility shim for archived runs.

## Risks & Open Questions

- **Config complexity** – Mitigated via defaults and helper templates in `ConfigManager`.
- **State ownership overlaps** – Need clear boundaries for which module saves optimizer/device state.
- **Performance regressions** – Additional abstractions may add overhead; profile on benchmark tasks.
- **Data source contracts** – Confirm how imagined batches or multi-modal data will specify shapes; schedule follow-up once base interface lands.

## Deliverables

- `src/orchestration/world_model_orchestrator.py`
- `src/orchestration/phase_scheduler.py`
- `src/workflows/world_models/{base.py,dreamer.py,context.py}`
- `src/data_sources/{base.py,replay.py,offline.py}`
- Controller manager + simulator service modules
- Updated configs, tests, and documentation (including this design doc)

