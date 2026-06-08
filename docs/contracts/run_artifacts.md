# Run Artifact Contract

Every serious run must leave a reproducible evidence trail. This contract tells
agents where to look before making claims about a training run.

The goal is not to make a database. The goal is to make every run easy to
inspect, compare, resume, and summarize.

## Required Remote Artifacts

A remote GPU run should preserve:

- `.hydra/config.yaml`
- `.hydra/overrides.yaml`
- TensorBoard event file under `runs/`
- `run_status.json`
- `checkpoints/latest.pt`
- `checkpoints/best.pt` when eval is available
- immutable checkpoint target such as `best_step_<step>.pt`
- immutable regular checkpoint target such as `step_<step>.pt`
- `pre_run/working_tree.diff` for dirty/WIP runs
- `pre_run/resolved_config.yaml`

`latest.pt` and `best.pt` are pointers. They should resolve to immutable files.
Do not treat a mutable checkpoint filename as evidence for a best model.

## Recommended Run Folder Shape

```text
experiments/<run_name>-<timestamp>/
├── .hydra/
│   ├── config.yaml
│   └── overrides.yaml
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   ├── final.pt
│   ├── step_<step>.pt
│   └── best_step_<step>.pt
├── diagnostics/
├── pre_run/
│   ├── resolved_config.yaml
│   └── working_tree.diff
├── runs/
├── run_status.json
└── run_summary.json
```

Some runs will not have every optional directory. Missing optional artifacts are
fine when the run is a smoke test. Missing required artifacts on a chapter run
should be recorded as a failure note.

## Required Local Artifacts

After a serious run, local repo state should include:

- a row in `reports/world_model_runs.csv`;
- pulled config, logs, TensorBoard events, summaries, and diagnostics when available;
- checkpoint metadata such as remote path, size, hash, and pointer target;
- a rough note when training exposed infrastructure or research lessons;
- diagnostic output when the run informs a chapter claim;
- a narrative update only after the manifest row exists.

The run folder is raw evidence. `run_summary.json` is the compressed factual
view. `reports/world_model_runs.csv` is the chronology table across runs.
Local evidence does not need to include heavyweight checkpoint `.pt` files by
default. Keep checkpoint payloads on the GPU during active work, run
checkpoint-heavy diagnostics where the checkpoint lives, and pull back the small
diagnostic outputs.

## Commit Policy

Commit source, configs, tests, docs, and the manifest row that supports a
claim. Keep large raw evidence local unless the user explicitly asks to publish
it.

Default local evidence directories are ignored:

```text
experiments/
runs/
remote_artifacts/
research_notes/
```

`reports/world_model_runs.csv` is the durable source-controlled summary even
though other generated report outputs may stay ignored.

## Live Run Status

`run_status.json` is the live polling contract for agents and GPU helper
scripts. It answers "what is this run doing right now?" without requiring the
agent to infer phase/action state from TensorBoard or logs.

Expected fields include:

```json
{
  "schema_version": 1,
  "run_id": "planet_dmc_swingup_chapter-20260605_153643",
  "status": "running",
  "host": "ASUS-TUF",
  "pid": 275675,
  "command": "python scripts/train.py ...",
  "experiment_dir": "experiments/...",
  "workflow_name": "PlaNetWorkflow",
  "global_step": 1818,
  "phase": "train_world_model_1",
  "action": "update_world_model",
  "hook_state": "completed",
  "started_at": "2026-06-05T...",
  "updated_at": "2026-06-05T...",
  "last_metrics": {
    "train/world_model/total_loss": 0.54
  }
}
```

Use `run_status.json` for live polling. Use `run_summary.json` for final
post-run facts.

GPU helper scripts should stay cheap and bounded. They may report reachability,
tmux state, `run_status.json`, checkpoint files, TensorBoard event file paths,
and recent logs. They should not become model-specific TensorBoard
interpreters. Scalar summaries, plots, reconstructions, and checkpoint probes
belong in diagnostics scripts under `scripts/research/diagnostics/`.

During live runs, prefer a tiny snapshot before pulling artifacts:

```bash
scripts/GPU/gpu_run_snapshot.sh --run experiments/<run_name>
```

This returns bounded JSON from `run_status.json`, recent eval lines in
`train.log`, summaries, and checkpoint file metadata. It does not copy
TensorBoard events, replay data, or checkpoint payloads.

Workflow-internal progress is a later v2 contract. The current status file
records phase/action boundaries from the orchestrator; a long workflow hook may
still need its own nested progress fields later.

## Agent Inspection Order

When inspecting a completed or interrupted run:

1. run `scripts/GPU/gpu_run_snapshot.sh --run experiments/<run_name>` if the run
   is still remote;
2. read `run_summary.json` if it exists;
3. read `run_status.json` if the run is active or interrupted;
4. inspect the resolved config from `.hydra/config.yaml` or `pre_run/resolved_config.yaml`;
5. export TensorBoard scalars from `runs/`;
6. record final and best eval metrics;
7. verify checkpoint links resolve to immutable targets;
8. run model-specific diagnostics if available;
9. update `reports/world_model_runs.csv`;
10. update roadmap or chapter narrative only after the manifest row exists.

## GPU Code And Artifact Movement

Keep the Mac repo as source of truth. Treat WSL as a worker unless the user
explicitly asks to bring remote code back.

Use path-limited pushes for code:

```bash
scripts/GPU/gpu_sync_patch.sh --paths "src scripts tests configs"
```

Launch remote runs with explicit session names and Hydra overrides:

```bash
scripts/GPU/gpu_run.sh --session <name> --experiment <experiment> --budget <budget> -- <override...>
```

Pull named run evidence, then export TensorBoard scalars locally when needed:

```bash
scripts/GPU/gpu_pull_latest.sh --run experiments/<run_name> --analyze
```

By default this is logs-first: configs, summaries, diagnostics, TensorBoard
event files, and `checkpoint_manifest.json`. The manifest records checkpoint
remote paths, sizes, and hashes without copying heavyweight `.pt` payloads.

Use `--checkpoint` only when a local agent explicitly needs to inspect
`latest.pt`, `best.pt`, `step_*.pt`, or `best_step_*.pt` directly. `--analyze`
does not imply `--checkpoint`.

Only pull code from WSL when those remote edits are intentional:

```bash
scripts/GPU/gpu_pull_patch.sh --paths "src scripts tests"
```

`gpu_pull_patch.sh` checks the patch locally by default. Add `--apply` only
after reviewing the patch path it prints.

## Post-Run Agent Checklist

1. Pull or record the run directory.
2. Export TensorBoard scalars.
3. Record final and best eval metrics.
4. Verify best checkpoint target is immutable.
5. Run model-specific diagnostics if available.
6. Update `reports/world_model_runs.csv`.
7. Update the chapter narrative only after the manifest row exists.
