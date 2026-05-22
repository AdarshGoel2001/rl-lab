# Agent Task: Add A Representation Learner

Use this task when adding a vision or latent-state encoder component. Keep the
workflow target explicit; the VAE-style and RSSM-style contracts are different.

## Scope

Allowed files:

- `src/components/representation_learners/<model_name>.py`
- `configs/components/representation_learner/<model_name>.yaml`
- `tests/test_<model_name>.py`
- README or contract docs only if the public contract changes

## Steps

1. Read `docs/contracts/component_interfaces.md`.
2. Choose one contract:
   - VAE-style dict API for OG World Models.
   - RSSM-style `LatentStep` / `LatentSequence` API for Dreamer-style code.
3. Add the smallest model that satisfies that contract.
4. Add a Hydra config with `_target_` and required dimensions.
5. Add a smoke test:
   - random batch or sequence in
   - expected output keys or dataclass fields present
   - trainable models produce a finite scalar loss
   - `.backward()` populates at least one gradient
6. Run:

```bash
python -m pytest tests/test_<model_name>.py -q
python -m pytest tests/test_hydra_component_switching.py -q
```

## Stop Conditions

Stop and ask the orchestrator if:

- You need to make VAE-style and RSSM-style learners interchangeable.
- You need a new workflow adapter.
- You need a new image layout convention.
- You need to change `WorkflowContext` or `WorldModelComponents`.
