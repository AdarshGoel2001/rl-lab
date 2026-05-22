# Agent Task: Add A Dynamics Model

Use this task when adding a small dynamics component. Do not modify the
orchestrator, phase scheduler, or unrelated workflows.

## Scope

Allowed files:

- `src/components/dynamics/<model_name>.py`
- `configs/components/dynamics_model/<model_name>.yaml`
- `tests/test_<model_name>.py`
- README or contract docs only if the public contract changes

## Steps

1. Read `docs/contracts/component_interfaces.md`.
2. Pick the target contract:
   - OG World Models sequence contract: implement `observe_sequence`.
   - Planning contract: implement `forward(state, action)`.
3. Copy the closest live implementation style.
4. Add a Hydra config with `_target_` and explicit dimensions.
5. Add a smoke test:
   - random tensors in
   - finite scalar loss out
   - `.backward()` populates at least one gradient
6. Run:

```bash
python -m pytest tests/test_<model_name>.py -q
python scripts/validate_experiment.py og_wm_carracing --budget tiny
```

7. If wiring into a full experiment, add one experiment config or override and
   run the tiny budget.

## Stop Conditions

Stop and ask the orchestrator if:

- The model needs a new workflow hook.
- The model needs a new buffer sample schema.
- You need to change `Orchestrator` or `PhaseScheduler`.
- You want to add a registry, schema system, dashboard, or experiment database.
