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

Active example:

- `src.components.representation_learners.rssm.RSSMRepresentationLearner`

Support examples:

- `src.components.representation_learners.conv_vae.ConvVAERepresentationLearner`
- `src.components.representation_learners.identity.IdentityRepresentationLearner`

Expected methods depend on the workflow. PlaNet uses the RSSM learner through
`observe`, `observe_sequence`, `imagine_step`, and `initial_state` style calls.
The stable state container is `RSSMState`.

Pixel and identity learners are support material for future chapters. A
VAE-style learner may expose:

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

Do not assume RSSM states, VAE latents, and identity features are
interchangeable without a workflow adapter.

## Dynamics Model

Support examples:

- `src.components.dynamics.mdn_rnn.MDNRNNDynamics`
- `src.components.dynamics.gaussian_gru.GaussianGRUDynamics`
- `src.components.dynamics.deterministic_mlp.DeterministicMLPDynamics`

No current live workflow uses these dynamics components directly. They remain
tested support material for future chapters.

An MDN-RNN-style sequence API looks like:

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

For collection-time recurrent state updates, recurrent dynamics may expose:

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
MDN-RNN sequence configs.

## Controller

Active examples:

- `src.components.controllers.random_policy.RandomPolicyController`
- `src.components.controllers.mpc_planner.MPCPlanner`

Support examples:

Future actor-critic, diffusion, or evolutionary controllers should be
reimplemented against the active workflow and component contracts instead of
resurrecting old scaffolds.

Minimal controller protocol:

```python
act(*args, **kwargs) -> torch.Tensor
```

Controller shape expectations are workflow-specific. The current PlaNet chapter
uses random policy for seed collection and `MPCPlanner` for CEM planning in RSSM
latent space.

For DMC cartpole swingup, action bounds are:

```text
action_dim: 1
action_low: [-1.0]
action_high: [1.0]
```

## Workflow

Live contract:

```python
initialize(context: WorkflowContext) -> None
collect_step(step: int, *, phase: Mapping[str, Any]) -> CollectResult | None
update_world_model(batch: Any, *, phase: Mapping[str, Any]) -> dict[str, float]
update_controller(batch: Any, *, phase: Mapping[str, Any]) -> dict[str, float]
imagine(...) -> dict[str, Any]
```

`update_controller()` and `imagine()` are optional in the base class. The active
PlaNet workflow implements `imagine()` for CEM planning and leaves learned
controller updates out of scope.

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
python scripts/validate_experiment.py planet_cartpole --budget planet_tiny
```
