# World Model Plug-and-Play Guide

This guide describes how to use the new configuration surfaces added to `BaseWorldModelParadigm` so you can drop in alternative world-model components—RSSMs, transformer dynamics, pretrained encoders, etc.—without editing framework code.

## 1. Baseline Usage

Existing MVP configs run unchanged. If you leave the new keys (`representation_loss`, `sequence_processing`) absent, the paradigm behaves exactly as before: features are flattened, decoder reconstruction overwrites the legacy `representation_loss`, and imagination rollouts ignore context.

## 2. New Configuration Surfaces

Additions live under the `algorithm` section (the same place you specify `world_model_lr`, `imagination_horizon`, and friends).

### 2.1 `representation_loss`

Controls how decoder reconstruction blends with learner-provided objectives.

| Field | Default | Description |
| --- | --- | --- |
| `decoder_mode` | `accumulate` | How decoder loss interacts with the learner’s `representation_loss`. Options: `accumulate` (add to learner loss), `replace` (overwrite the key), `none` (keep decoder as a separate metric only). |
| `key` | `representation_loss` | Metric name used when reporting the combined representation loss. Set this if you want a different logging key (e.g., `kl_loss`). |

Example:

```yaml
algorithm:
  representation_loss:
    decoder_mode: accumulate
    key: dreamer_rep_loss
```

### 2.2 `sequence_processing`

Defines how batches are shaped, what context is made available to modules, and how imagination rollouts behave.

Top-level fields:

| Field | Default | Description |
| --- | --- | --- |
| `mode` | `auto` | Force or disable sequence mode. `auto` respects buffer layout; `always` reshapes as `[batch, time, ...]`; `never` keeps everything flattened. |
| `default_component_input` | `flatten` | Default tensor layout delivered to components. `flatten` → `(batch*time, dim)`, `sequence` → `(batch, time, dim)`. |
| `component_inputs` | `{}` | Per-component override (`representation_learner`, `dynamics_model`, `reward_predictor`, `continue_predictor`). |
| `provide_causal_mask` | `false` | Attach an upper-triangular attention mask for sequence-aware modules. |
| `causal_mask_includes_future` | `false` | When true, diagonal zeros are retained (mask allows attending to same timestep); otherwise strictly causal. |
| `provide_padding_mask` | `false` | Supplies padding mask for variable-length rollouts based on `dones`/`continues`. |
| `padding_mask_source` | `dones` | Choose which signal builds the mask (`dones` or `continues`). |
| `include_flat_in_context` | `false` | Keep flattened tensors in the context dictionary alongside sequences (useful for hybrids). |
| `context_attribute` | `sequence_context` | Attribute set on each component if `attach_context_attribute` is true. |
| `attach_context_attribute` | `true` | Controls whether context is stored as an attribute on components. |
| `context_method` | `null` | Optional method name (e.g., `set_sequence_context`) invoked on every component after context assembly. |
| `context_methods` | `{}` | Per-component overrides for method name (e.g., `dynamics_model: set_context`). |
| `imagination` | `{}` | Nested configuration controlling imagined rollout context (see below). |

The context dictionary always carries `sequence_length`, `batch_size`, plus whichever tensors you enabled (features, states, actions, rewards, dones, continues, masks, histories).

#### 2.2.1 `sequence_processing.imagination`

| Field | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | If true, imagination rollouts populate and forward a context dictionary to policy/dynamics/reward/continue components. |
| `carry_history` | `true` | Maintains a `state_history` tensor (B×t×dim) across rollout steps. |
| `context_key` | `sequence_context` | The dictionary key used when injecting imagination context into downstream calls. |
| `base_context` | `{}` | Optional static context merged into the imagination context before rollouts begin. |

## 3. Component Expectations

- **Encoders:** implement `BaseEncoder`. Pretrained weights can be loaded in `_build_encoder`; freeze parameters there if required.
- **Representation Learners:** `encode` should accept tensors shaped per your `component_inputs` setting. If you expect sequence context, add a signature like `encode(self, features, sequence_context=None)`.
- **Dynamics Models:** `forward` and `dynamics_loss` may receive `sequence_context`. Add optional kwargs or `**_` to your implementations.
- **Reward & Continue Predictors:** similarly can accept `sequence_context` and operate on sequence tensors when configured.
- **Policy/Value Heads:** continue to consume latent states. If you need context, include an optional `context` kwarg.

If a module does **not** accept the context argument, the helper will silently drop it, maintaining backward compatibility.

## 4. Example Configurations

### 4.1 MVP Baseline (no sequence features)

```yaml
algorithm:
  world_model_lr: 2e-4
  actor_lr: 3e-4
  critic_lr: 1e-4
  imagination_horizon: 15
  # No representation_loss / sequence_processing keys → legacy behaviour

components:
  encoder:
    type: mlp
    config: { input_dim: 4, hidden_dims: [64, 64], activation: tanh }
  representation_learner:
    type: mlp_autoencoder
    config: { hidden_dims: [64], activation: tanh, representation_dim: 16 }
  dynamics_model:
    type: deterministic_mlp
    config: { hidden_dims: [128, 128], activation: tanh }
  ...
```

### 4.2 Dreamer-style RSSM

```yaml
algorithm:
  world_model_lr: 6e-4
  actor_lr: 8e-5
  critic_lr: 4e-5
  imagination_horizon: 15
  representation_loss:
    decoder_mode: accumulate
    key: rssm_rep_loss
  sequence_processing:
    mode: auto
    default_component_input: sequence
    component_inputs:
      representation_learner: sequence
      dynamics_model: sequence
      reward_predictor: sequence
      continue_predictor: sequence
    provide_causal_mask: false
    provide_padding_mask: true
    padding_mask_source: continues
    imagination:
      enabled: true
      carry_history: true
      context_key: rssm_context

components:
  encoder:
    type: cnn
    config: { input_channels: 3, feature_dim: 1024 }
  representation_learner:
    type: rssm
    config:
      stochastic_size: 32
      deterministic_size: 200
      repr_steps: 15
  dynamics_model:
    type: rssm_dynamics
    config:
      stochastic_size: 32
      deterministic_size: 200
      prior_layers: [200]
      posterior_layers: [200]
  reward_predictor:
    type: rssm_reward
    config: { hidden_dims: [400, 400] }
  continue_predictor:
    type: rssm_continue
    config: { hidden_dims: [400, 400] }
  policy_head:
    type: gaussian_mlp
    config: { hidden_dims: [400, 400], std_mode: learned }
  value_function:
    type: rssm_value
    config: { hidden_dims: [400, 400] }
```

Key points:
- Sequence inputs are delivered directly (`batch × time × dim`).
- RSSM modules should accept `sequence_context` to access masks and history.
- Imagination context provides `state_history`, enabling Dreamer-like actor training.

### 4.3 Transformer Dynamics

```yaml
algorithm:
  sequence_processing:
    mode: always
    default_component_input: sequence
    component_inputs:
      representation_learner: sequence
      dynamics_model: sequence
    provide_causal_mask: true
    causal_mask_includes_future: false
    provide_padding_mask: true
    context_method: set_sequence_context
    imagination:
      enabled: true
      carry_history: true
      context_key: transformer_ctx

components:
  representation_learner:
    type: token_projector
    config: { in_dim: 128, token_dim: 128 }
  dynamics_model:
    type: transformer_dynamics
    config:
      model_dim: 256
      num_layers: 4
      num_heads: 8
      dropout: 0.1
  reward_predictor:
    type: transformer_reward
    config: { model_dim: 256 }
  continue_predictor:
    type: transformer_continue
    config: { model_dim: 256 }
```

Ensure your transformer modules implement a method `set_sequence_context(context)` (or adjust `context_method`) so padding/causal masks are cached internally before forward passes.

### 4.4 Pretrained Encoder + Feed-forward Dynamics

```yaml
components:
  encoder:
    type: resnet18_pretrained
    config:
      freeze: true
      output_dim: 512
  representation_learner:
    type: identity
    config: { representation_dim: 512 }
  dynamics_model:
    type: deterministic_mlp
    config:
      state_dim: 512
      action_dim: 4
      hidden_dims: [512, 512]

algorithm:
  representation_loss:
    decoder_mode: none
```

Freezing is handled inside the encoder class by calling `requires_grad_(False)` when `freeze: true`.

## 5. Recreating Published Paradigms

| Paradigm | Config Tips |
| --- | --- |
| **Dreamer (V2-style)** | Use the RSSM example above. Ensure KL balancing and reward/value targets are implemented inside custom components (e.g., `rssm_dynamics.dynamics_loss`). Set `imagination.enabled` and supply history so the actor sees latent rollouts. |
| **PlaNet** | Similar to Dreamer but you can disable the actor/critic portion: set `actor_updates_per_batch: 0`, `critic_updates_per_batch: 0`, and rely on planning in a custom planner component. Use sequence processing for RSSM losses. |
| **TD-MPC / Latent MPC** | Provide ensemble dynamics via multiple registered models and reference them from a custom planner. Use `sequence_processing.include_flat_in_context: true` if your planner needs both `[batch, time, dim]` and flattened states. |
| **Decision Transformer / IRIS** | Set `mode: always`, deliver sequences to dynamics/policy/value, enable causal masks, and ensure components ingest `sequence_context['causal_mask']` and `['padding_mask']`. |

## 6. Limitations & Caveats

- **Component Signatures:** context is only passed if your module accepts an explicit kwarg or `**_`. Update custom modules accordingly.
- **Observation Types:** current pipeline assumes tensor observations. For dict observations, introduce an adapter encoder or wrapper.
- **Masks & Variable Length:** padding masks are provided but dynamics/reward losses must honour them; otherwise truncated episodes may corrupt gradients.
- **Value Function Context:** the built-in value function still consumes flattened latents. If your critic requires sequence context, extend it accordingly.
- **Optimizer Groups:** encoders/representation/dynamics always join the world-model optimizer; to freeze parameters, toggle gradients inside the component.

## 7. Debugging Checklist

1. Start with defaults (`sequence_processing` absent) to verify baseline training.
2. Enable one feature at a time (e.g., set `component_inputs.dynamics_model: sequence`).
3. Log or assert inside custom modules to confirm `sequence_context` contains expected fields (`state_history`, `causal_mask`, etc.).
4. Use `python scripts/validate_config.py --config ...` to confirm YAML is parsed correctly.
5. Watch logs for warnings about invalid modes or missing fields—`BaseWorldModelParadigm` emits descriptive errors when configuration is malformed.

By combining these config switches with registered components you can re-create Dreamer-like RSSMs, transformer-based world models, or any hybrid architecture without touching the paradigm core.
