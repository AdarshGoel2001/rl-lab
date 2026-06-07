# Agentic Workflow

This repo should be easy for coding agents to extend without making it bigger
than it needs to be. The working rule is simple: agents change one boundary at a
time, prove the change with a tiny run or smoke test, and leave unrelated
abstractions alone.

## Operating Rules

1. Start from a live experiment config, not a new training script.
2. Change one component family at a time: representation learner, dynamics model,
   controller, buffer, environment, or workflow.
3. Add a Hydra config next to the closest existing config.
4. Add a smoke test that exercises forward/backward or config resolution.
5. Run `python scripts/validate_experiment.py <experiment> --budget planet_tiny` before
   running training.
6. Run `python scripts/train.py +experiment=<experiment> budget=planet_tiny` only after
   config validation passes.
7. Do not add registries, schemas, dashboards, databases, or new orchestration
   layers unless at least two working experiments need them.

## Extension Loop

```text
Read contract -> inspect closest live implementation -> add minimal component
-> add config -> add smoke test -> validate config -> run tiny budget -> compare artifacts
```

After any serious remote run, read `docs/contracts/run_artifacts.md` and update
the manifest before writing narrative claims.

For multi-agent chapter work, use `docs/agent_team_operating_model.md`. Start
from `docs/repo_map.md` so each agent knows which files are live, support
material, or future material.

For the full paper-to-run-to-narrative path, use
`docs/research_lifecycle.md`. For durable claims that should be backed by
tests or scripts, use `docs/executable_doc_audit.md`.

## Commands

Validate the current World Models baseline without training:

```bash
python scripts/validate_experiment.py planet_cartpole --budget planet_tiny
```

Run the tiny smoke experiment:

```bash
python scripts/train.py +experiment=planet_cartpole budget=planet_tiny
```

Run tests:

```bash
python -m pytest -q
```

## Good Agent Tasks

- Add a small dynamics model that matches the documented dynamics contract.
- Add a config that swaps one existing component.
- Add a smoke test for one component's forward/backward path.
- Add a run comparison script that reads existing `metrics.csv` files.
- Improve one contract doc using facts from live code.

## Bad Agent Tasks

- Rewrite the orchestrator.
- Replace Hydra.
- Add a plugin system.
- Add a database for experiments.
- Add a large model zoo before two small models are validated.
- Change multiple component families in one patch.
