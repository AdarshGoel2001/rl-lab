# Atari PPO Researcher Workflow

This guide walks through training PPO on `ALE/Pong-v5` with the modular
infrastructure and highlights the pieces researchers typically modify.

## 1. Environment Configuration

Create a config (e.g. `configs/experiments/ppo_atari_pong.yaml`) that selects
the `atari` wrapper. The wrapper now handles both single and vectorized
execution. Set `num_environments` to the desired degree of parallelism for
training and leave evaluation at `1` for deterministic rollouts.

```yaml
environment:
  name: ALE/Pong-v5
  wrapper: atari
  num_environments: 8
  parallel_backend: sync
  frame_stack: 4
  frame_skip: 4
  sticky_actions: 0.25
  clip_rewards: true
```

Evaluation inherits the same preprocessing automatically. Provide an explicit
`evaluation` block if you need custom behaviour; otherwise the trainer will
instantiate a single-environment Atari wrapper with matching options.

## 2. Components

Use the registered components in `src/components` to assemble the agent:

- `nature_cnn` encoder for 84×84×4 stacked frames.
- `categorical_mlp` policy head for discrete logits.
- `critic_mlp` value head sharing the encoder outputs.

The default PPO paradigm wiring handles the shared encoder and separate policy
/ value heads. Adjust hidden sizes or activations in the YAML if needed.

## 3. Buffer & Training Schedule

For Atari PPO a common setup is 8 environments × 256 steps = 2048 samples per
update:

```yaml
buffer:
  type: trajectory
  capacity: 2048
  batch_size: 2048
  gamma: 0.99
  gae_lambda: 0.95

training:
  total_timesteps: 1000000
  eval_frequency: 50000
  checkpoint_frequency: 100000
```

The trainer clips gradients (`max_grad_norm`) and PPO ratios (`clip_ratio`)
for update “capping” and logs KL, entropy, and losses automatically.

## 4. Launching Training

Run the experiment with the CLI harness:

```bash
python scripts/train.py --config configs/experiments/ppo_atari_pong.yaml --device cuda
```

Artifacts (logs, checkpoints, TensorBoard runs) are written to
`experiments/<run_name>/`. Evaluation rollouts follow the same preprocessing as
training so you can trust deterministic metrics.

## 5. Sanity Checks

1. `pytest tests/test_atari_environment.py` – validates observation scaling and
   vectorized execution.
2. `python scripts/train.py ... --dry-run` – constructs all components without
   starting the loop to ensure configs resolve.
3. Inspect TensorBoard for reward curves and KL to confirm PPO stabilises.

With these pieces in place a new researcher can focus purely on experimenting
with encoder architectures, loss schedules, or additional components without
redoing the environment plumbing.
