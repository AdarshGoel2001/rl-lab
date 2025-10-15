# Modular World-Model Framework Tasklist

The following task breakdown translates the design into concrete implementation work items. Each task links to intended files/modules and highlights dependencies. Checkboxes indicate completion status (all unchecked initially).

## Milestone 1 — Foundation

- [ ] Extract shared experiment bootstrap into `src/orchestration/world_model_orchestrator.py`
- [ ] Implement `WorkflowContext` (`src/workflows/world_models/context.py`) wiring environments, components, loggers, checkpoint manager
- [ ] Define `WorldModelWorkflow` ABC (`src/workflows/world_models/base.py`)
- [ ] Migrate current Dreamer training logic into `DreamerWorkflow` (`src/workflows/world_models/dreamer.py`)
- [x] Expose orchestrator entrypoint in `src/paradigms/world_models/__init__.py` (legacy trainer removed)
- [ ] Create thin replay data source adapter wrapping `WorldModelSequenceBuffer` (`src/data_sources/replay.py`)
- [ ] Update unit tests to point Dreamer path at workflow (`tests/workflows/test_dreamer_workflow.py`)

## Milestone 2 — Data & Phase Abstractions

- [ ] Implement `PhaseScheduler` (`src/orchestration/phase_scheduler.py`) with ordered phase support
- [ ] Extend orchestrator to consult scheduler for hook order
- [ ] Refactor buffer interactions through `DataSource` interface (`src/data_sources/base.py`)
- [ ] Add offline dataset adapter (`src/data_sources/offline.py`) with simple tensor/npz loader
- [ ] Update configs to accept `training.phases` / `data_sources` (e.g., `configs/experiments/world_model_cartpole_mvp.yaml`)
- [ ] Document configuration changes in `docs/worldModel_Universal_build_guidelines.md`

## Milestone 3 — Controllers & Simulation Hooks

- [ ] Extend `ComponentFactory` to instantiate controllers by role and register them in context
- [ ] Introduce `ControllerManager` helper (`src/workflows/world_models/controllers.py` or similar)
- [ ] Add optional `controller.learn()` hook invocation inside `update_controller`
- [x] Implement imagination helper inside Dreamer workflow (`DreamerWorkflow.imagine`) and share mixins if reused
- [x] Update Dreamer workflow to consume controller manager & its own imagination helper

## Milestone 4 — Testing & Migration

- [ ] Add scheduler coverage tests (`tests/orchestration/test_phase_scheduler.py`)
- [ ] Add data source tests (`tests/data_sources/test_replay_source.py`, `tests/data_sources/test_offline_source.py`)
- [ ] Create end-to-end smoke test (`tests/integration/test_world_model_orchestrator.py`)
- [ ] Mirror legacy metrics in new path and compare (guarded test or script)
- [ ] Update documentation + READMEs referencing orchestrator workflow
- [x] Remove legacy `WorldModelTrainer`; all configs use orchestrator path

## Milestone 5 — Post-MVP Extensions

- [ ] Implement `BehaviorCloningWorkflow` using offline data
- [ ] Scaffold planner/diffusion workflow stubs demonstrating interface usage
- [ ] Add reusable imagination helper mixins for planner rollouts (e.g., CEM loop)
- [ ] Remove deprecated trainer once parity confirmed

## Cross-Cutting Concerns

- [ ] Ensure checkpoint serialization keeps optimizer state per workflow hook
- [ ] Establish standard metric keys emitted by workflows (document in design doc)
- [ ] Verify logging remains backward compatible with `experiment_logger`

> **Note:** Track progress via repo issues or project board; keep this checklist in sync after each merged PR.
