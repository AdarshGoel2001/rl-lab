# Buffer Adapters

Buffers are the data boundary between workflow phases. New model code should not
special-case datasets inside workflows; it should add or select a buffer that
implements the existing contract:

- `initialize(context)`: optional setup, including loading offline data.
- `add(trajectory=...)`: online collection path.
- `sample(batch_size=None)`: training batch path.
- `ready()`: whether training can proceed.
- `finalize()`: optional persistence hook.

## Sequence Dataset Format

The lightweight reusable format is compressed `.npz` with arrays stored as:

- `observations`: `[env, time, ...]`
- `actions`: `[env, time, ...]`
- `rewards`: `[env, time]`
- `dones`: `[env, time]`

`WorldModelSequenceBuffer` can now write this format when `dataset_path` is set,
and can read it with `read_only: true`.

## Recommended Patterns

Use three configs instead of one overloaded experiment:

- collect: online phases write a reusable dataset.
- offline train: read-only buffer, no collection phases.
- online finetune: future pattern, load old data and append new policy rollouts.

Keep adapters simple. Add a new buffer only when the batch shape or storage
format genuinely differs from the existing sequence contract.
