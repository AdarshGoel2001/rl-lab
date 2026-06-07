# Evaluation Ladder

The benchmark ladder exists so the repo can show chronological improvement
without pretending that every paper used the same official benchmark.

Each algorithm should be tested on the lowest layer it can reasonably support.
More capable algorithms climb higher. The output is not one universal score,
but a small table of capabilities across layers.

## Reporting Principles

Every run should report:

- environment and observation modality;
- data budget or environment steps;
- training updates;
- wall time and, when remote, GPU type;
- final evaluation return;
- best evaluation return if available;
- seed count;
- main model losses;
- artifact path;
- failure mode.

Use honest labels:

- `smoke`: proves wiring only;
- `tiny`: small repeatable check;
- `chapter`: serious low-compute result for the narrative;
- `stretch`: larger run, not required for iteration.

## Layer 0: Wiring Smoke

Purpose: catch broken configs, tensor shapes, and training-loop issues quickly.

| Environment | Modality | Why it exists | Pass condition |
| --- | --- | --- | --- |
| Synthetic/fake batches | state | component forward/backward tests | finite loss and gradients |
| Gym CartPole tiny | state | workflow and buffer smoke | train/eval completes |
| CarRacing tiny | pixels | image path smoke | reconstruction artifact produced |

This layer is not for claims about algorithm quality.

## Layer 1: Control Sanity

Purpose: prove a small model can actually learn/control something.

| Environment | Modality | Good for | Candidate pass condition |
| --- | --- | --- | --- |
| `CartPole-v1` | state | fast control sanity | mean return >= 475 over 5 episodes |
| `Pendulum-v1` | state | continuous action sanity | clear improvement over random |
| DMC `cartpole_balance` | state | DMC wrapper sanity | stable high return over 5 episodes |

All state-space algorithms should pass this layer before moving to harder DMC
tasks.

## Layer 2: State World-Model Benchmark

Purpose: compare the main chronology of latent dynamics and planning/control.

| Environment | Modality | Why it matters | Candidate chapter target |
| --- | --- | --- | --- |
| DMC `cartpole_swingup` | state | small but nontrivial planning/control | mean return >= 800 over 5 episodes |
| DMC `cheetah_run` | state | continuous locomotion and velocity reward | improvement curve over random/planner baseline |
| DMC `walker_walk` | state | harder dynamics and stability | stable improvement, not necessarily solved |

This is the main layer for:

- PlaNet;
- Dreamer;
- TD-MPC;
- small robust world-model recipes.

For this repo, DMC Swingup is the first serious target.

## Layer 3: Pixel And Partial-Observation Benchmark

Purpose: test representation learning and visual rollout quality.

| Environment | Modality | Good for | Candidate target |
| --- | --- | --- | --- |
| CarRacing 64x64 | pixels | VAE/RNN and visual control features | reconstruction + reward/control improvement |
| MiniGrid partial observation | symbolic/pixels | memory and partial observability | success-rate improvement |
| Atari Pong tiny | pixels | token/transformer/diffusion world models | rollout quality + policy improvement |

This layer should not block early state-space work. Pixel models are more
expensive and introduce representation issues that can hide control issues.

## Layer 4: Representation And Prediction Benchmark

Purpose: evaluate models whose value may not appear first as high RL score.

Metrics:

- one-step next-state or next-latent prediction error;
- 5/15/30-step rollout error;
- reward prediction error;
- done/continue calibration;
- latent linear probe for true state variables;
- action counterfactual consistency;
- planning improvement over random using frozen model;
- sampling or inference cost.

This layer is required for:

- JEPA-style representation models;
- diffusion world models;
- token/transformer world models;
- any model that looks visually good but does not yet solve control.

## Compute Tiers

The repo should make compute explicit.

| Tier | Intended use | Typical budget |
| --- | --- | --- |
| Local smoke | Mac sanity checks | seconds to a few minutes |
| WSL tiny | quick GPU validation | under 30 minutes |
| WSL chapter | serious low-compute result | 1-6 GPU hours |
| Stretch | optional larger run | only after chapter result is understood |

The RTX 3060 Laptop GPU is enough for state DMC experiments and small pixel
experiments. It is not enough to chase full-scale modern video world-model
results. The roadmap should embrace small versions and explain that constraint.

## First Completed Result

The first serious chapter result is:

```text
algorithm: PlaNet-style RSSM + CEM
environment: DMC cartpole_swingup
modality: state
target: mean return >= 800 over 5 eval episodes
result: mean return 637.53, max return 820.11
status: useful chapter result, target not solved
budget: WSL chapter tier
run id: planet_dmc_swingup_paper_authentic_20260602_123034
artifacts: run directory, TensorBoard logs, Hydra config, manifest row, checkpoint snapshot
```

Because the first run did not hit the target, the next work should be
diagnostic:

1. verify reward and observation dimensions;
2. verify planner action bounds and CEM scoring;
3. compare random collection versus planner collection;
4. inspect model losses and eval returns separately;
5. tune only the PlaNet loop before starting Dreamer.

## Run Manifest Reminder

Each serious run should be recorded in `reports/world_model_runs.csv`:

```text
paper -> year -> implementation -> config -> run directory -> score -> artifacts
```

The manifest is the source for chronology plots. Narrative docs should point
back to manifest rows instead of carrying unverified scores.
