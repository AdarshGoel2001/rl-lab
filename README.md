# RL Lab

RL Lab is an executable chronology of world-model papers under a low-compute
constraint. The repo implements small, honest versions of important world-model
ideas, runs them on a shared evaluation ladder, and records what improved over
time.

## Current Chapter

The current completed chapter is PlaNet-style RSSM + CEM on state-observation
DMC cartpole_swingup.

Latest completed chapter result:

- run id: `planet_dmc_swingup_paper_authentic_20260602_123034`
- eval mean: `637.53`
- eval max: `820.11`
- modality: state, not pixels
- manifest: `reports/world_model_runs.csv`

This is not a full pixel PlaNet reproduction. It is the first serious Layer 2
state world-model anchor for this repo.

## How The System Works

- `scripts/train.py` wires Hydra configs, environments, buffers, components,
  controllers, optimizers, and the workflow.
- `src/orchestration/orchestrator.py` owns the loop, logging, checkpointing,
  resume modes, and buffer routing.
- `src/orchestration/phase_scheduler.py` chooses collect, update, and eval
  actions from the YAML phase schedule.
- `src/workflows/*.py` contain algorithm logic.
- `src/components/` contains duck-typed model pieces.

Infrastructure stays outside algorithms. Workflows should not own checkpointing
or logging setup. The orchestrator should not contain algorithm-specific loss or
planning logic.

## Agent Start Here

1. Read `docs/roadmap/world_model_chronology.md`.
2. Read `docs/roadmap/eval_ladder.md`.
3. Read `docs/roadmap/run_manifest.md`.
4. Read `docs/repo_map.md`.
5. Read `docs/repo_inventory.md`.
6. Read `docs/research_lifecycle.md`.
7. Read `docs/agent_team_operating_model.md`.
8. Read `docs/contracts/workflow_data_contract.md`.
9. Read `docs/contracts/run_artifacts.md`.
10. Inspect the closest existing workflow, config, and test before changing code.

## Running Locally

Use local runs for smoke tests and config checks.

```bash
python -m pytest -q
```

```bash
python scripts/validate_experiment.py planet_cartpole --budget planet_tiny
```

```bash
python scripts/train.py +experiment=planet_cartpole budget=planet_tiny
```

## Running On WSL GPU

Use the WSL GPU machine for serious training. The Mac repo remains the source of
truth; send patches to the GPU, run there, and pull artifacts back.

```bash
scripts/GPU/gpu_status.sh
```

```bash
scripts/GPU/gpu_sync_patch.sh
```

```bash
scripts/GPU/gpu_run.sh --session <name> --experiment <experiment> --budget <budget> -- <override...>
```

```bash
scripts/GPU/gpu_run_snapshot.sh --run experiments/<run_name>
```

```bash
scripts/GPU/gpu_pull_latest.sh --run experiments/<run_name> --analyze
```

After a serious remote run, update `reports/world_model_runs.csv` from the run
artifacts instead of from memory.

## GPU Communication Model

Control is SSH: check status, sync code, start tmux runs, and execute bounded
remote commands. Live monitoring should start with `gpu_run_snapshot.sh`, which
returns small JSON from the remote run without copying TensorBoard events,
replay data, or checkpoint payloads.

Heavy checkpoints stay on the GPU by default. Run checkpoint-heavy diagnostics
where the checkpoints live, then pull back configs, logs, summaries,
diagnostics, TensorBoard scalars, and checkpoint metadata. Use
`gpu_pull_latest.sh --checkpoint` only when a local agent truly needs checkpoint
payloads.

## Diagnostics

Model-specific diagnostics live under `scripts/research/diagnostics/`. The
current PlaNet diagnostic entrypoint is:

```bash
python scripts/research/diagnostics/diagnose_planet_checkpoint.py --help
```

Diagnostics should write artifacts or TensorBoard scalars when they support a
chapter claim.

## What Not To Do

- Do not add broad abstractions before one concrete run needs them.
- Do not implement many papers in parallel.
- Do not make narrative claims before the manifest row exists.
- Do not treat `latest.pt` or `best.pt` as immutable checkpoint files; they are
  pointers.
