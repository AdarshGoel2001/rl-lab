# Hafner et al. 2019: PlaNet

Source files:

- PDF: `docs/papers/hafner_2019_planet.pdf`
- Extracted text: `docs/papers/hafner_2019_planet.txt`
- arXiv: https://arxiv.org/abs/1811.04551
- project page: https://planetrl.github.io/
- official code: https://github.com/google-research/planet

## What PlaNet Is

PlaNet is not the Ha and Schmidhuber `V/M/C + CMA-ES` recipe. It is a
model-based planner:

- learn a latent dynamics model from collected episodes;
- infer a belief over the current latent state from history;
- use CEM/MPC to choose actions online in latent space;
- do not train a policy or value network.

The observation model is used for representation learning, but planning uses
the transition and reward models directly in latent space.

## Model

PlaNet uses an RSSM with deterministic and stochastic state:

- deterministic path: `h_t = f(h_{t-1}, s_{t-1}, a_{t-1})`;
- stochastic path: `s_t ~ p(s_t | h_t)`;
- encoder/posterior: `q(s_t | h_t, o_t)`;
- observation model: `p(o_t | h_t, s_t)`;
- reward model: `p(r_t | h_t, s_t)`.

The paper found both deterministic and stochastic paths important for planning.

## Training Loop

The main loop is iterative:

1. collect `S` random seed episodes;
2. fit the world model on random sequence chunks from the replay dataset;
3. collect more episodes using the current planner plus Gaussian exploration
   noise;
4. repeat model fitting and data collection.

This means an offline-only run is useful for debugging, but a faithful PlaNet
agent needs planner-guided online collection as the dataset improves.

## Planning

PlaNet plans with CEM:

- initialize a factorized Gaussian over action sequences;
- sample candidate action sequences;
- imagine one latent trajectory per candidate;
- score each sequence by summed predicted reward means;
- refit the Gaussian to the top candidates;
- execute only the first mean action, then replan after the next observation.

Paper hyperparameters:

- horizon `H = 12`;
- optimization iterations `I = 10`;
- candidates `J = 1000`;
- elites `K = 100`;
- seed episodes `S = 5`;
- collect one episode every `C = 100` model updates;
- action noise `Normal(0, 0.3)`;
- batch size `B = 50`;
- sequence length `L = 50`;
- Adam learning rate `1e-3`;
- gradient clipping norm `1000`;
- free nats `3`.

For this MacBook repo, we should keep tiny versions of these settings, but the
shape of the recipe should remain the same.

## Continuation / Done

The PlaNet paper models reward but does not include a continuation/done model
in the main algorithm. Later Dreamer-style agents do. If we keep a
continuation head in this repo, it is a PlaNet-adjacent extension, not core
PlaNet.

For imagined planning, continuation-aware scoring is still principled when the
model predicts termination:

`value = r0 + gamma * continue0 * r1 + ...`

But for a paper-faithful PlaNet baseline, the planner should default to summed
predicted rewards over the planning horizon. Continuation-aware scoring should
be optional and measured, not silently assumed.

## Repo Implications

- Remove task-specific heuristic policies from the PlaNet demo path.
- Keep `.npz` offline replay because PlaNet depends on replayed sequence chunks.
- Keep online planner-guided collection as the real solve path.
- Keep RSSM + reward prediction as the minimal PlaNet world model.
- Treat continuation prediction as an extension useful for Dreamer-style agents.
- Do not touch Orchestrator for PlaNet-specific controller experiments unless
  the workflow contract truly cannot express the phase.
