# AGENTS.md

This is the single agent entrypoint for RL Lab. Read it before changing code or
starting experiments.

## Purpose

RL Lab is an executable chronology of world-model papers under a low-compute
constraint. The repo should let an agent move from paper idea to implementation,
training run, diagnostics, manifest row, and narrative update without relying on
hidden chat context.

Current completed chapter:

- PlaNet-style RSSM + CEM
- DMC `cartpole_swingup`
- state observations, not pixels
- run id: `planet_dmc_swingup_paper_authentic_20260602_123034`
- best/final eval mean: `637.53`
- Layer 2 target not solved yet

## Read First

For research direction:

1. `docs/roadmap/world_model_chronology.md`
2. `docs/roadmap/eval_ladder.md`
3. `docs/roadmap/run_manifest.md`
4. `reports/world_model_runs.csv`
5. `docs/research_lifecycle.md`

For system boundaries:

1. `README.md`
2. `docs/repo_map.md`
3. `docs/repo_inventory.md`
4. `docs/agentic_workflow.md`
5. `docs/agent_team_operating_model.md`
6. `docs/executable_doc_audit.md`
7. `docs/contracts/workflow_data_contract.md`
8. `docs/contracts/component_interfaces.md`
9. `docs/contracts/buffers.md`
10. `docs/contracts/run_artifacts.md`

For remote training:

1. `scripts/GPU/gpu_status.sh`
2. `scripts/GPU/gpu_sync_patch.sh`
3. `scripts/GPU/gpu_pull_latest.sh`

## Core Architecture

Keep the three-layer boundary intact:

- `scripts/train.py` wires Hydra configs, environments, buffers, components,
  controllers, optimizers, and workflows.
- `src/orchestration/orchestrator.py` owns infrastructure: loop execution,
  phase scheduling calls, buffer routing, logging, checkpointing, resume modes,
  and cleanup.
- `src/workflows/*.py` own algorithm logic: collection, losses, updates,
  imagination, planning, and workflow-specific state.

Short version: orchestrator owns infrastructure; workflows own algorithm logic.
Do not put losses, planners, or model-specific training rules in the
orchestrator. Do not put checkpoint I/O, TensorBoard setup, or experiment
directory policy inside workflows.

## Experiment Flow

Use configs, not new training scripts:

```bash
python scripts/validate_experiment.py <experiment> --budget planet_tiny
```

```bash
python scripts/train.py +experiment=<experiment> budget=planet_tiny
```

For serious GPU runs, keep the Mac repo as the source of truth. Send code by
patch, run on WSL, then pull lightweight evidence back.

```bash
scripts/GPU/gpu_status.sh
scripts/GPU/gpu_sync_patch.sh --paths "src scripts tests configs"
scripts/GPU/gpu_run.sh --session <name> --experiment <experiment> --budget <budget> -- <override...>
scripts/GPU/gpu_run_snapshot.sh --run experiments/<run_name>
scripts/GPU/gpu_pull_latest.sh --run experiments/<run_name> --analyze
```

Use `scripts/GPU/gpu_pull_patch.sh --paths "src scripts tests"` only when
remote WSL code edits are intentionally coming back to the Mac. It checks the
patch by default; add `--apply` only after reviewing it.

## Run Evidence

Every serious run must follow `docs/contracts/run_artifacts.md`.

Required habits:

- preserve `.hydra/config.yaml` and `.hydra/overrides.yaml`;
- keep TensorBoard event files under `runs/`;
- make `latest.pt` and `best.pt` point to immutable checkpoint targets;
- pull logs, configs, TensorBoard events, diagnostics, and checkpoint metadata
  by default, not heavyweight checkpoint `.pt` files;
- record dirty/WIP patches under `pre_run/`;
- update `reports/world_model_runs.csv` before writing narrative claims;
- run model-specific diagnostics when a result will support a chapter claim.

The manifest is the graphable chronology table. The run folder is raw evidence.
Narrative docs should point back to manifest rows.

## Adding Algorithms

New algorithm work should usually add:

1. one workflow in `src/workflows/`;
2. component configs under `configs/components/` or controller configs under
   `configs/controller/`;
3. one experiment config under `configs/experiment/`;
4. one small budget under `configs/budget/`;
5. focused tests for config resolution, tensor shapes, and the core update;
6. a smoke run before any long GPU run.

Do not implement many papers in parallel. Use parallel agents around one chapter
only: paper faithfulness, implementation, diagnostics, and artifact extraction.
When coordinating multiple agents, follow `docs/agent_team_operating_model.md`.

## Buffers

Buffers are not one universal abstraction. Use the existing docs as examples:

- RAM-only on-policy buffers for temporary rollout data;
- persistent replay buffers for resumable world-model learning;
- external dataset loaders for offline datasets.

Do not force a new algorithm into an existing buffer if its data semantics are
different. Build the smallest buffer shape needed and document the trajectory
contract.

## Checkpoints And Resume

Checkpoint files should be immutable except for pointer symlinks:

- regular: `step_<global_step>.pt`
- best: `best_step_<global_step>.pt`
- final: `final.pt`
- pointers: `latest.pt`, `best.pt`

Resume modes:

- `exact`: fault tolerance for the same interrupted run. It restores scheduler,
  step, optimizer, RNG, workflow state, and any required replay state.
- `warm_start`: restore learned weights/controllers, skip optimizer and old
  schedule.
- `warm_start_optimizer`: normal research continuation. It restores learned
  weights/controllers/optimizers, then uses the new schedule and fresh or
  explicitly loaded data.

## Diagnostics

Diagnostics live under `scripts/research/diagnostics/`. They may be
model-specific. Prefer useful, local scripts over a premature shared diagnostics
framework.

When a diagnostic supports a chapter claim, write outputs into the run artifact
tree or TensorBoard so another agent can inspect them later.

## Stop Conditions

Stop and explain before proceeding when:

- a run result contradicts the expected trend;
- checkpoint or resume behavior is ambiguous;
- a doc claim cannot be tied to a manifest row or run artifact;
- a change would cross orchestrator/workflow boundaries;
- a test failure is unrelated to your current task;
- user approval is needed for a new long GPU run.

When in doubt, make the smallest observable change, verify it, and write down
what happened.
