# World Model Chronology

This repo is an executable history of world-model ideas under a deliberately
low-compute constraint. The goal is not to reproduce every paper's official
benchmark table. The goal is to implement small, honest versions of the main
architectural ideas and test them on a shared benchmark ladder so progress can
be explained chronologically.

The narrative should read:

```text
I implemented World Models, then PlaNet, then Dreamer-style imagination,
then TD-MPC-style planning/value learning, then token/transformer world
models, then JEPA-style latent prediction, then diffusion world models.

For each step, I kept the compute small, used a consistent set of easier
environments, and measured what the new idea actually improved.
```

## Why Not Use Paper Scores Directly?

World-model papers use different environments, data budgets, compute budgets,
observation modalities, and evaluation protocols. A direct graph of official
paper score versus year would be misleading.

Instead, this repo should create its own controlled graph:

```text
year / paper idea -> small implementation -> same eval ladder -> score,
rollout quality, representation quality, GPU time, and failure modes
```

This makes the project useful even when a tiny implementation does not match
the full paper. The question is not "did we beat the paper?" The question is
"what part of the idea survives in a small, runnable version?"

## The Spine: State-Space World Models

These are the core chapters. They should be implemented serially, because each
chapter depends on the evaluation discipline and infrastructure proven by the
previous one.

| Chapter | Main idea | Tiny repo version | Primary proof |
| --- | --- | --- | --- |
| World Models 2018 | VAE + recurrent latent dynamics + small controller | VAE/MDN-RNN on simple pixels or state features; controller first evaluated in real env | Learns compressed dynamics and can support control features |
| PlaNet 2019 | RSSM + latent CEM planning | State-observation RSSM with reward head and CEM planner | Solves or improves DMC Cartpole Swingup |
| Dreamer V1 | Actor-critic trained in imagined RSSM rollouts | Small RSSM + actor/critic on state tasks first | Better sample efficiency than pure CEM planning |
| Dreamer V2/V3 | Discrete/stabilized latent imagination at broader scale | Tiny robust recipe across multiple small tasks | Fewer per-task hacks, more stable across the ladder |
| TD-MPC / TD-MPC2 | Latent dynamics + value-guided MPC | State continuous-control planner/value variant | Stronger continuous-control scores per GPU hour |

## Second Lane: Token And Transformer World Models

These should start only after the state-space spine is measurable.

| Chapter | Main idea | Tiny repo version | Primary proof |
| --- | --- | --- | --- |
| IRIS-style | Discrete image tokens + autoregressive transformer world model | Tiny VQ tokenizer + transformer on MiniGrid/CarRacing/Pong snippets | Better long-horizon pixel rollout than RSSM decoder baseline |
| STORM-style | Stochastic transformer latent dynamics | Tiny transformer latent model with stochastic prediction | Improved sequence prediction and planning stability |

These models may not beat state-space models on small control tasks. Their
value is visible in long-horizon sequence prediction, discrete latent modeling,
and pixel rollout quality.

## Third Lane: JEPA-Style Latent Prediction

JEPA-style models should not begin as RL agents. Their first job is to learn
useful predictive representations without reconstructing every pixel.

Tiny repo version:

- encode observations into latent features;
- predict masked or future latent targets;
- evaluate with linear probes, rollout consistency, reward prediction, and
  planning usefulness after freezing or lightly finetuning the representation.

Primary proof:

- latent features preserve controllable state;
- future predictions stay useful over multiple steps;
- a simple planner or policy learns faster than from raw observations or a
  reconstruction-only encoder.

## Fourth Lane: Diffusion World Models

Full diffusion world models are compute-heavy. This repo should implement
diffusion-lite variants first.

Tiny repo version:

- action-conditioned next-observation or next-latent diffusion model;
- short-horizon rollout model on small pixels;
- compare against VAE/RSSM/transformer rollout quality under the same data
  budget.

Primary proof:

- sharper or more plausible pixel predictions than MSE decoders;
- useful short-horizon counterfactuals;
- explicit reporting of sampling cost, because diffusion may be better but too
  slow for low-compute control.

## What Each Chapter Must Produce

Each chapter should leave the same evidence trail:

1. Paper notes under `docs/papers/`.
2. Workflow/component/config references.
3. A smoke run on the lowest benchmark layer.
4. A serious run on at least one benchmark layer.
5. Logs, scores, and artifacts copied back from WSL when remote training is used.
6. A short "what changed from the previous chapter" explanation.
7. A short "what failed in the small version" explanation.

The failure section is important. For a low-compute repo, failure modes are part
of the research result.

## Parallelism Rule

Do not implement many papers in parallel.

Use parallel agents around one chapter:

- one agent checks paper faithfulness;
- one agent runs or monitors GPU experiments;
- one agent fixes the next concrete blocker;
- one agent extracts artifacts and updates docs.

Keep the algorithm frontier serial:

```text
finish PlaNet DMC evidence -> then Dreamer -> then TD-MPC -> then token models
```

This avoids five incomplete implementations fighting over buffers, metrics,
configs, and evaluation semantics.

## Current Completed Milestone

The first serious chronology point is complete:

- algorithm: PlaNet-style RSSM + CEM
- environment: DMC `cartpole_swingup`
- modality: state observations
- run id: `planet_dmc_swingup_paper_authentic_20260602_123034`
- best/final eval mean: `637.53`
- best max return: `820.11`

This is not a full pixel PlaNet reproduction. It is a state-observation,
paper-aligned low-compute run. It supports the claim that the RSSM+CEM chapter
is operational but has not solved the Layer 2 target of mean return `>= 800`.

## Current Next Milestone

Before starting Dreamer, finish the PlaNet evidence packet:

1. pull or record the paper-authentic run artifact directory;
2. run reward, open-loop, and planner diagnostics on the best available
   paper-authentic checkpoint evidence;
3. write a short PlaNet chapter note explaining what worked, what failed, and
   why the Layer 2 target is still unsolved.

Immutable checkpoint retention is now a code contract for future runs. The
paper-authentic run still needs careful artifact handling because it was created
before that fix.
