# Repo Map

This is the quick discovery map for agents. It separates live surfaces from
support code and future material.

## Live Research Surface

Use these first for current work:

```text
AGENTS.md
README.md
docs/agentic_workflow.md
docs/agent_team_operating_model.md
docs/repo_inventory.md
docs/research_lifecycle.md
docs/executable_doc_audit.md
docs/roadmap/world_model_chronology.md
docs/roadmap/eval_ladder.md
docs/roadmap/run_manifest.md
reports/world_model_runs.csv
```

For handoffs, use:

```text
docs/agent_handoff_template.md
```

Current live workflow:

```text
src/workflows/planet.py
configs/workflow/planet.yaml
configs/experiment/planet_cartpole.yaml
configs/experiment/planet_dmc_cartpole_swingup.yaml
configs/budget/planet_tiny.yaml
```

## Infrastructure Surface

Training entrypoint:

```text
scripts/train.py
scripts/validate_experiment.py
src/orchestration/orchestrator.py
src/orchestration/phase_scheduler.py
```

Contracts:

```text
docs/contracts/workflow_data_contract.md
docs/contracts/component_interfaces.md
docs/contracts/buffers.md
docs/contracts/run_artifacts.md
```

Remote GPU tools:

```text
scripts/GPU/gpu_status.sh
scripts/GPU/gpu_sync_patch.sh
scripts/GPU/gpu_run.sh
scripts/GPU/gpu_run_snapshot.sh
scripts/GPU/gpu_pull_latest.sh
scripts/GPU/gpu_pull_patch.sh
scripts/GPU/gpu_tail.sh
scripts/GPU/gpu_metrics.sh
```

Diagnostics:

```text
scripts/research/export_tensorboard_run.py
scripts/research/diagnostics/diagnose_planet_checkpoint.py
```

## Current PlaNet Chapter

Core code:

```text
src/workflows/planet.py
src/components/representation_learners/rssm.py
src/components/prediction_heads/mlp.py
src/components/controllers/mpc_planner.py
src/components/controllers/random_policy.py
src/buffers/world_model_sequence.py
```

Key tests:

```text
tests/test_planet_tiny.py
tests/test_planet_smoke.py
tests/test_dmc_cartpole_swingup_config.py
tests/test_orchestrator_evaluation_contract.py
tests/test_planet_diagnostics.py
```

## Support And Future Material

Some components exist for future chapters but are not active workflows today.
Do not assume their presence means the corresponding algorithm is implemented.

Examples:

```text
src/components/dynamics/
src/components/return_computers/
configs/environment/
configs/buffer/
```

Before using a support component in a chapter, inspect its tests and either
validate it against the current contract or rewrite it.

## Artifact Surface

Run evidence should live under the experiment directory and follow:

```text
docs/contracts/run_artifacts.md
```

Chronology evidence should be summarized in:

```text
reports/world_model_runs.csv
docs/roadmap/run_manifest.md
```

Rough thinking and temporary observations belong under:

```text
research_notes/rough_notes/
```

Clean future plans belong under:

```text
research_notes/clean_plans/
```

## Deletion Rule

If a file describes a workflow that is not live, it must be one of:

- a paper note;
- a future-plan note;
- a support component with tests;
- removed.

Do not leave dead Hydra experiment entrypoints around as examples.
