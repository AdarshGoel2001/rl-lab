# RL Lab

RL Lab is a modular world-models framework for experimenting with interchangeable vision, dynamics, and controller modules on top of an Orchestrator and phase scheduler. The current working baseline is an in-progress reproduction-style implementation inspired by Ha & Schmidhuber 2018: a convolutional VAE for visual representation learning, an MDN-RNN for latent dynamics, and scaffolding for a future CMA-ES controller.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| VAE vision module | ✅ Implemented | Convolutional encoder/decoder with reconstruction and KL losses. |
| MDN-RNN dynamics module | ✅ Implemented | LSTM dynamics with mixture latent prediction, reward loss, and done loss. |
| Gaussian GRU dynamics module | ✅ Implemented | Simpler single-Gaussian recurrent baseline swappable through Hydra. |
| PlaNet tiny recipe | ✅ Implemented | State-based CartPole RSSM + reward/continue heads + CEM-compatible imagination smoke. |
| Rollout collection | ✅ Implemented | Random-policy CarRacing rollouts feed the replay/disk buffer. |
| Training loops | ✅ Implemented | Hydra config drives phase-based VAE and MDN-RNN updates. |
| CMA-ES controller | 🟡 Partial / scaffold | Interface exists, optimizer is intentionally not implemented yet. |
| Dream rollouts | 🔴 Not implemented | `imagine()` remains a documented `NotImplementedError`. |
| Controller update step | 🔴 Not implemented | `update_controller()` remains a documented `NotImplementedError`. |
| Known failure modes | ⚠️ Known | VAE pretraining collapse on extended training is tracked in `NOTES.md`. |

## Architecture

1. `scripts/train.py` loads Hydra config and instantiates environments, buffers, components, controllers, and optimizers.
2. `Orchestrator` owns the training lifecycle: context creation, loop execution, checkpointing, logging, and cleanup.
3. `PhaseScheduler` decides which workflow hook runs next: collect, update world model, update controller, or evaluate.
4. `OriginalWorldModelsWorkflow` contains algorithm logic for collection, VAE updates, and MDN-RNN updates.
5. Components stay swappable through config: VAE, dynamics model, controller, buffer, and environment are all separate modules.

## Setup

Use the existing conda environment:

```bash
conda activate rl-lab
```

If CarRacing is missing Box2D in that environment, install it with:

```bash
conda install -n rl-lab -c conda-forge box2d-py swig
```

## Runnable Commands

```bash
pytest -q
```

```bash
python scripts/train.py +experiment=og_wm_carracing budget=tiny
```

The tiny budget keeps the CarRacing framing but runs only enough rollout and update steps to produce smoke-test artifacts: `metrics.csv`, `loss_curves.png`, and `reconstruction_grid.png` in the Hydra experiment output directory.

Validate an experiment config without running training:

```bash
python scripts/validate_experiment.py og_wm_carracing --budget tiny
```

Swap in the simpler Gaussian GRU dynamics baseline:

```bash
python scripts/train.py +experiment=og_wm_carracing budget=tiny components/dynamics_model=gaussian_gru
```

Run the tiny PlaNet-style recipe on CartPole state observations:

```bash
python scripts/train.py +experiment=planet_cartpole budget=planet_tiny
```

Run a small Mac-safe PlaNet CartPole solve attempt with TensorBoard metrics:

```bash
python scripts/train.py +experiment=planet_cartpole budget=planet_solve
```

View TensorBoard logs:

```bash
tensorboard --logdir experiments
```

## Agentic Extension

Agent-facing contracts and task templates live under `docs/`:

- `docs/agentic_workflow.md` defines the extension loop and anti-bloat rules.
- `docs/contracts/component_interfaces.md` documents component shape and method contracts.
- `docs/contracts/workflow_data_contract.md` documents workflow, phase, and trajectory boundaries.
- `docs/agent_tasks/` contains bounded task templates for adding model components.

## What This Is And Isn't

This is an honest, in-progress ML systems repo showing a modular world-models training stack and a runnable Original World Models baseline. It is not a finished agent, does not yet train a controller, and does not yet perform dream rollouts for policy optimization. The goal of the current state is reproducibility, inspectable artifacts, and a clear foundation for extending the framework.
