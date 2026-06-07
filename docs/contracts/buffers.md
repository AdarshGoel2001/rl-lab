# Buffers

Buffers are the data boundary between workflows and training. They are not just
storage; they also define what a valid training sample means for an algorithm.

Do not force every algorithm into one universal replay buffer. When adding a
new learner, copy the closest pattern below and change only the sampling or
persistence behavior the learner actually needs.

## Shared Interface

Buffers should keep the small interface used by the orchestrator:

```python
initialize(context)
add(trajectory=...)
sample(batch_size=None)
ready()
clear()
finalize()
```

Not every buffer needs every method to do real work. Read-only loaders should
raise on `add`. Disk-backed buffers may refuse `clear`. Temporary on-policy
buffers should clear after their update cycle.

## Pattern 1: Temporary On-Policy Buffer

Use this when data is valid only for the policy that collected it.

Examples:

```text
PPO
A2C
REINFORCE
```

Lifecycle:

```text
collect rollout -> compute returns/advantages -> update policy -> clear buffer
```

Typical contents:

```text
observations
actions
rewards
dones
values
log_probs
returns
advantages
```

Persistence rule:

```text
Do not persist old rollouts as reusable training data.
For exact resume, only save the current unfinished rollout if needed.
```

Current reference:

```text
src/buffers/trajectory.py
```

## Pattern 2: Persistent Replay Buffer

Use this when collected environment experience remains useful after model or
policy updates.

Examples:

```text
PlaNet
Dreamer
TD-MPC
off-policy world models
```

Lifecycle:

```text
open replay storage -> append collected trajectories -> sample batches -> keep appending
```

Typical contents:

```text
observations
actions
rewards
dones
episode boundary metadata when needed
```

Persistence rule:

```text
For small runs, exact resume can store buffer state inline.
For larger runs, write replay data to disk and checkpoint only the dataset path,
sampling state, and metadata needed to reopen it.
```

Current references:

```text
src/buffers/world_model_sequence.py
src/buffers/disk_buffer.py
```

## Pattern 3: External Dataset Loader

Use this when the dataset already exists outside the run.

Examples:

```text
D4RL
downloaded robot datasets
expert demonstrations
offline RL datasets
```

Lifecycle:

```text
load dataset path -> build valid sample index -> serve batches
```

Typical contents depend on the dataset. Dataset-specific loaders are expected.
For example, an action-chunking diffusion policy loader may return:

```text
observations: (B, obs_dim)
actions: (B, horizon, action_dim)
```

Persistence rule:

```text
Do not copy dataset bytes into checkpoints.
Checkpoint only dataset identity, path, sampler RNG/cursor, and loader metadata.
```

Current reference:

```text
src/buffers/offline.py
```

## World-Model `.npz` Adapter Format

The lightweight reusable sequence format is compressed `.npz`.
`WorldModelSequenceBuffer` writes and reads arrays as:

```text
observations: [env, time, ...]
actions: [env, time, ...]
rewards: [env, time]
dones: [env, time]
```

It also stores metadata keys prefixed with `__`, such as sequence length and
stride.

Use this format for simple collect-once, train-later world-model experiments.
Use `DiskBuffer` when replay should be appended continuously or become a larger
research artifact.

## Resume Semantics

Do not assume every resume means the same thing.

```text
exact:
  fault tolerance for the same interrupted run; restore model, optimizer,
  scheduler, workflow state, RNG, and any buffer state needed to continue the
  same training process

warm_start:
  restore model/controller weights, but use a fresh optimizer, fresh scheduler,
  fresh global step, and fresh data unless the new config collects or loads data

warm_start_optimizer:
  research continuation; restore model/controller weights and optimizer state,
  but use a fresh scheduler, fresh global step, and fresh data unless the new
  config collects or loads data
```

If exact resume is requested and a buffer cannot restore the needed data, the
run should fail early with a clear message instead of silently becoming a warm
start.

## Questions Before Adding A Buffer

Answer these before creating or changing a buffer:

```text
Is the learner on-policy, off-policy, or offline?
Does old data remain valid after updates?
Does the run own the data, or is the data external?
Does exact resume require buffer contents?
Can data stay in RAM, or should it be written to disk?
What is one valid sample for this algorithm?
Where are episode boundaries allowed?
```

The last two questions are usually the hard part. Episode boundaries, burn-in,
action chunks, recurrent state, and terminal handling are valid reasons to make
a dataset-specific or algorithm-specific buffer.

## Current Direction

Use these defaults until a concrete experiment proves otherwise:

```text
On-policy reference:
  TrajectoryBuffer

Small world-model replay:
  WorldModelSequenceBuffer

Persistent replay:
  DiskBuffer

External dataset:
  Dataset-specific loader built for the dataset being used

Advanced recurrent or pixel world-model replay:
  Build a focused buffer when a concrete chapter needs it
```

Do not add a new buffer just to rename fields. Add one when the algorithm needs
different storage, sampling, episode-boundary, or persistence behavior.
