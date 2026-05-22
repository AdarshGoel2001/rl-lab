# Component Interface Contracts

These contracts describe the live extension points in RL Lab. They are written
for humans and coding agents adding small, testable model components.

## General Contract

Every component should:

- Accept plain constructor kwargs from Hydra.
- Store important dimensions as attributes such as `latent_dim` or `action_dim`.
- Inherit `torch.nn.Module` if it has trainable parameters.
- Support `.to(device)` through normal PyTorch behavior when it owns tensors.
- Return dictionaries or dataclasses with stable keys, not positional tuples.
- Have one smoke test with random tensors and finite loss/backward behavior when
  the component is trainable.

## Representation Learner

Live examples:

- `src.components.representation_learners.conv_vae.ConvVAERepresentationLearner`
- `src.components.representation_learners.identity.IdentityRepresentationLearner`

Expected methods depend on the workflow. The OG World Models workflow expects a
VAE-style learner:

```python
observe(features: torch.Tensor) -> dict[str, torch.Tensor]
observe_sequence(features: torch.Tensor, actions=None, dones=None, **kwargs) -> dict[str, torch.Tensor]
decode(latent: torch.Tensor) -> torch.Tensor
```

For `ConvVAERepresentationLearner`, frame tensors use image-last layout:

```text
single batch: (B, C, H, W) for observe()
sequence batch: (B, T, H, W, C) for observe_sequence()
latent: (B, latent_dim) or (B, T, latent_dim)
decoded sequence: (B, T, H, W, C)
```

The VAE output dictionary must include:

```text
latent
mean
logvar
```

Dreamer-style learners may return `LatentStep` or `LatentSequence`, but those
objects are not currently the OG World Models path. Do not assume the two
families are interchangeable without a workflow adapter.

## Dynamics Model

Live examples:

- `src.components.dynamics.mdn_rnn.MDNRNNDynamics`
- `src.components.dynamics.gaussian_gru.GaussianGRUDynamics`
- `src.components.dynamics.deterministic_mlp.DeterministicMLPDynamics`

The OG World Models workflow expects an MDN-RNN-style sequence API:

```python
observe_sequence(
    latent: torch.Tensor,
    action: torch.Tensor,
    dones: torch.Tensor,
    *,
    deterministic: bool | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]
```

Expected shapes:

```text
latent: (B, T, latent_dim)
action: (B, T, action_dim)
dones: (B, T)
```

Expected output keys for training:

```text
pi_logits: (B, T, num_gaussians)
mu: (B, T, num_gaussians, latent_dim)
logvar: (B, T, num_gaussians, latent_dim)
reward_preds: (B, T, 1)
done_logits: (B, T, 1)
```

For collection-time recurrent state updates, OG World Models also calls:

```python
observe(
    latent: torch.Tensor,
    action: torch.Tensor,
    dones: torch.Tensor | numpy.ndarray | None = None,
    *,
    deterministic: bool = False,
    return_mixture: bool = False,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]
```

At minimum, `observe(..., return_mixture=False)` must return:

```text
next_latent: (B, latent_dim)
hidden: workflow-specific recurrent state
```

`DeterministicMLPDynamics` is a different contract used by planning-style
workflows. It has `forward(state, action)` and is not a drop-in replacement for
the OG MDN-RNN config.

## Controller

Live examples:

- `src.components.controllers.random_policy.RandomPolicyController`
- `src.components.controllers.cma_es.CMAESController`

Minimal controller protocol:

```python
act(*args, **kwargs) -> torch.Tensor
```

Controller shape expectations are workflow-specific. The current OG World
Models data-collection phase uses the random policy actor to produce continuous
CarRacing actions with:

```text
action_dim: 3
action_low: [-1.0, 0.0, 0.0]
action_high: [1.0, 1.0, 1.0]
```

The CMA-ES controller is scaffold only. Do not present it as a working optimizer.

## Workflow

Live contract:

```python
initialize(context: WorkflowContext) -> None
collect_step(step: int, *, phase: Mapping[str, Any]) -> CollectResult | None
update_world_model(batch: Any, *, phase: Mapping[str, Any]) -> dict[str, float]
update_controller(batch: Any, *, phase: Mapping[str, Any]) -> dict[str, float]
imagine(...) -> dict[str, Any]
```

`update_controller()` and `imagine()` are optional in the base class. In
`OriginalWorldModelsWorkflow`, they intentionally remain `NotImplementedError`
until dream rollouts and controller training are implemented.

### Recipe-Specific Notes

`PlaNetWorkflow` is the first RSSM planning recipe. It uses:

```text
representation_learner: RSSMRepresentationLearner
reward_predictor: MLPHead
continue_predictor: MLPHead
planner: MPCPlanner
buffer: WorldModelSequenceBuffer
```

Its `imagine()` accepts either an `RSSMState` or a flattened RSSM latent tensor.
The tensor path exists because `MPCPlanner` evaluates candidate action sequences
from flattened latent features. This is an adapter at the workflow boundary, not
a new global latent-state abstraction.

## Config Contract

A runnable experiment should resolve these groups:

```text
workflow
components
controllers
environment
buffer or buffers
training.phases
```

Use this command before training:

```bash
python scripts/validate_experiment.py og_wm_carracing --budget tiny
```
