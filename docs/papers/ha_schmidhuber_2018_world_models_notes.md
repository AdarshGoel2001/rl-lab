# Ha and Schmidhuber 2018: World Models

Source files:

- PDF: `docs/papers/ha_schmidhuber_2018_world_models.pdf`
- Extracted text: `docs/papers/ha_schmidhuber_2018_world_models.txt`
- arXiv: https://arxiv.org/abs/1803.10122
- interactive paper: https://worldmodels.github.io/

## Core Architecture

The paper decomposes the agent into three modules:

- `V`: a VAE that compresses image observations into latent vectors `z`.
- `M`: an MDN-RNN that predicts future latent vectors from `(z_t, a_t, h_t)`.
- `C`: a very small controller, usually a linear map from `[z_t, h_t]` to action.

The core design principle is to put most parameters and representation learning
inside `V` and `M`, then keep `C` small enough for black-box optimization.

## CarRacing Recipe

The CarRacing experiment is not dream training. It uses the world model as a
feature extractor while controller fitness is evaluated in the real environment.

Procedure:

1. Collect 10,000 random-policy rollouts.
2. Train VAE to encode frames into `z in R32`.
3. Train MDN-RNN to model `P(z_{t+1} | a_t, z_t, h_t)`.
4. Define controller as `a_t = W_c [z_t, h_t] + b_c`.
5. Use CMA-ES to optimize `W_c, b_c` against real-environment cumulative reward.

Important implication: for CarRacing, the paper's `M` predicts latent dynamics,
not reward or done. Reward is only used to score controller rollouts.

## Doom Recipe

Doom is the dream-training experiment. The world model must behave like a Gym
environment, so `M` predicts both the next latent vector and whether the agent
dies:

`P(z_{t+1}, d_{t+1} | a_t, z_t, h_t)`.

During hallucinated rollout, if predicted death probability exceeds a threshold,
the dream environment marks `done = true`. Survival time is the task reward.

Important implication: predicted done/continue is not cosmetic. It defines when
the dream episode terminates. A controller or planner that ignores predicted
done will score impossible post-death futures.

## Controller Details

The original controller is intentionally tiny:

- CarRacing controller input: `[z, h]`.
- Doom controller input: `[z, h, c]` because the LSTM cell state is also used.
- Action output is bounded with `tanh`.
- Controller parameters are optimized by CMA-ES, not gradient RL.

This repo should not claim that a hand-coded CartPole rule is a world-model
solution. It can only be a plumbing sanity check.

## Temperature and Model Exploitation

The paper emphasizes that learned dream environments are exploitable. In Doom,
too-low MDN-RNN temperature made the dream model collapse into unrealistic easy
worlds, causing policies that scored well in dreams to fail in reality.

Temperature is used as a robustness knob:

- too low: deterministic, easy to exploit;
- moderate: more stochastic, better transfer;
- too high: too noisy to learn useful behavior.

## Iterative Training

For harder environments, the paper proposes looping:

1. Roll out in the real environment and save observations/actions.
2. Train or refine the world model.
3. Train the controller inside or through the world model.
4. Repeat if the task is not solved.

The paper suggests richer future models may predict next observation, reward,
action, and done. This is closer to PlaNet/Dreamer-style latent RL than the
specific CarRacing setup.

## Repo Design Implications

For faithful World Models reproduction:

- keep offline replay/data adapters first-class;
- separate `V`, `M`, and `C` cleanly;
- implement CMA-ES over a tiny learned linear controller;
- for CarRacing, evaluate controller fitness in the real environment first;
- for dream rollouts, make predicted done/continue terminate or discount the
  imagined trajectory;
- do not substitute a task-specific heuristic controller for `C`.

For PlaNet/Dreamer-style recipes:

- reward and continuation heads are appropriate;
- planner/controller scoring must multiply future value by predicted
  continuation;
- bad continuation predictions can hurt, so this should be measured and
  visualized rather than assumed correct.
