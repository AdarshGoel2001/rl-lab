# Hafner et al. 2020: Dreamer V1

Source files:

- PDF: `docs/papers/hafner_2020_dreamer_v1.pdf`
- Extracted text: `docs/papers/hafner_2020_dreamer_v1.txt`
- arXiv: https://arxiv.org/abs/1912.01603
- project page: https://dreamrl.github.io/
- official code: https://github.com/google-research/dreamer

## What Dreamer Is

Dreamer keeps the PlaNet-style learned latent dynamics model, but replaces
online CEM planning with a learned actor and value model trained inside
imagined RSSM rollouts.

Short version:

- learn a latent world model from replay;
- start imagined rollouts from posterior states inferred from replay;
- use an actor to choose imagined actions;
- predict rewards, continuation/discounts, and values along imagined rollouts;
- compute lambda returns;
- train the value model to regress those returns;
- train the actor to maximize those returns by backpropagating through imagined
  dynamics;
- execute the learned actor in the real environment with exploration noise.

The actor/value learning is the main chapter difference from PlaNet. PlaNet
plans online. Dreamer amortizes behavior into a policy.

## Model Pieces

World model:

- representation model: `p(s_t | s_{t-1}, a_{t-1}, o_t)`;
- transition model: `q(s_t | s_{t-1}, a_{t-1})`;
- reward model: `q(r_t | s_t)`;
- optional continuation/discount model for early termination.

Behavior models:

- action model: `q(a_t | s_t)`;
- value model: `v(s_t)`.

For continuous control, the paper uses a tanh-transformed Gaussian action
model. For this repo's first Dreamer chapter, use continuous DMC control first.
Discrete action support can wait.

## Training Loop

Dreamer alternates three operations:

1. train the latent dynamics model on replay sequences;
2. train actor and value model from imagined trajectories;
3. collect new real-environment episodes by executing the actor with
   exploration noise.

This matches the repo's existing phase-scheduler shape. Dreamer should not
require orchestrator-specific algorithm logic.

## Behavior Learning

Imagined trajectories begin from posterior states computed from replay
sequences. For each imagined step:

1. actor samples or predicts an action from the current latent state;
2. RSSM prior predicts the next latent state;
3. reward and continuation heads predict imagined rewards and discounts;
4. value model predicts state values.

Then:

- compute lambda-return targets with `gamma = 0.99` and `lambda = 0.95`;
- train critic/value with stopped-gradient lambda returns;
- train actor to maximize lambda returns;
- keep world-model parameters fixed during actor/critic updates, while still
  allowing gradients through the imagined computation graph to reach actor
  parameters.

## Paper Hyperparameters

Continuous-control values from the paper:

- random seed episodes: `S = 5`;
- collect interval: `C = 100` training steps per environment episode;
- batch size: `B = 50`;
- sequence length: `L = 50`;
- imagination horizon: `H = 15`;
- RSSM latent distributions: 30-dimensional diagonal Gaussians;
- dense functions: 3 layers of size 300 with ELU activations;
- world-model learning rate: `6e-4`;
- actor learning rate: `8e-5`;
- value learning rate: `8e-5`;
- gradient clip norm: `100`;
- free nats: `3`;
- discount: `0.99`;
- lambda: `0.95`;
- exploration noise: `Normal(0, 0.3)`;
- paper action repeat: fixed `2`.

For low-compute repo runs, keep the shape of the recipe but scale dimensions,
batch size, sequence length, and update counts down first. Record deviations in
the manifest and chapter note.

## Repo Implications

- Add a new `DreamerV1Workflow`; do not modify the orchestrator with
  Dreamer-specific losses.
- Reuse the PlaNet RSSM, replay buffer, reward/continue/observation heads, run
  artifact contract, GPU workflow, and DMC state environment.
- Add trainable actor and critic/value modules.
- Put actor and critic under controller roles for the first implementation so
  existing optimizer grouping can provide separate `actor` and `critic`
  optimizers without changing `scripts/train.py`.
- Implement lambda returns in Torch inside the Dreamer workflow first. The
  existing NumPy TD-lambda helper is for real-data returns and should not be
  forced into imagined-rollout training.
- The first benchmark should be state-observation DMC `cartpole_swingup`.
- The first claim is comparison against the PlaNet anchor run at `637.53`, not
  "solved", until the manifest proves otherwise.

## First Low-Compute Slice

Start with:

- `configs/experiment/dreamer_dmc_cartpole_swingup.yaml`;
- `configs/budget/dreamer_tiny.yaml`;
- state observations;
- small RSSM and MLP sizes;
- short replay sequences;
- horizon `5` for smoke, horizon `15` for the first serious run;
- one-episode evals;
- CPU smoke before WSL GPU.

Minimum verification before a serious GPU run:

```bash
python scripts/validate_experiment.py dreamer_dmc_cartpole_swingup --budget dreamer_tiny
python scripts/train.py +experiment=dreamer_dmc_cartpole_swingup budget=dreamer_tiny experiment.device=cpu
```

## Open Implementation Questions

- Exact actor distribution API: expose action, mode, distribution, and log-prob
  cleanly without overbuilding.
- Whether the first actor loss should include entropy regularization. The paper
  says no entropy bonus was necessary for their continuous-control setup.
- Whether target networks are useful later. The paper did not require target
  value networks for Dreamer V1.
- Whether state-observation Dreamer needs observation reconstruction loss, or
  whether state prediction can use the existing observation head directly. For
  chapter continuity, keep an observation/state reconstruction head unless tests
  prove a cleaner state-only objective.
