# Workflow And Data Contract

This contract describes how workflows communicate with the orchestrator and
buffers. Keep it small and explicit so coding agents can add experiments without
changing the orchestration layer.

## Workflow Hooks

The orchestrator calls workflow hooks based on the active `PhaseScheduler`
action:

```text
collect -> workflow.collect_step(step, phase=phase_config)
update_world_model -> workflow.update_world_model(batch, phase=phase_config)
update_controller -> workflow.update_controller(batch, phase=phase_config)
evaluate -> workflow.evaluate(...) when implemented
```

New workflows should inherit `WorldModelWorkflow` and implement:

```python
initialize(context: WorkflowContext) -> None
update_world_model(batch: Any, *, phase: Mapping[str, Any]) -> dict[str, float]
```

Only implement `collect_step`, `update_controller`, or `imagine` when the
workflow genuinely needs them.

## CollectResult

`collect_step` should return `CollectResult`:

```python
CollectResult(
    episodes=0,
    steps=number_of_environment_steps,
    metrics={"name": scalar_float},
    trajectory={...},
    extras={...},
)
```

Field meanings:

- `steps`: how much to advance the global step and phase step counter.
- `episodes`: completed episodes during this collection call.
- `metrics`: scalar logging values.
- `trajectory`: the canonical trajectory routed to the phase buffer.
- `extras`: optional workflow-specific metadata. It is accepted by the
  dataclass, but the orchestrator does not route it yet.

If a workflow needs nonstandard buffer routing through `extras`, add a focused
test and update the orchestrator deliberately. Do not hide routing behavior in a
component.

## Canonical Online Trajectory

The online world-model path uses time-major vectorized arrays:

```text
observations: (T, num_envs, ...)
actions: (T, num_envs, action_dim)
rewards: (T, num_envs)
dones: (T, num_envs)
```

Buffers may sample batch-major sequences:

```text
observations: (B, sequence_length, ...)
actions: (B, sequence_length, action_dim)
rewards: (B, sequence_length)
dones: (B, sequence_length)
```

When adding a new workflow, document whether it consumes online trajectories,
offline dataset sequences, or single transitions.

## Metrics

Metrics should be flat scalar floats:

```python
{
    "vae_recon_loss": 123.4,
    "mdn_latent_nll": 45.6,
}
```

Avoid nested metrics dictionaries in workflow returns. Put complex artifacts in
files under the run directory instead.
