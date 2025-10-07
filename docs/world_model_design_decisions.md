# World Model Design Decisions: Deep Dive

## Purpose of This Document

This document explains **every major design decision** in your world model implementation, including:
1. What was chosen
2. Why it was chosen
3. What alternatives exist
4. Trade-offs of each approach
5. When you might want to switch

---

## Decision 1: Separate Optimizers for WM, Actor, Critic

### What Was Implemented
```python
world_model_optimizer = Adam(
    [encoder, repr_learner, dynamics, reward, continue, decoder],
    lr=1e-4
)
actor_optimizer = Adam([policy_head], lr=3e-5)
critic_optimizer = Adam([value_function], lr=3e-5)
```

### Why This Design?

**1. Different Learning Rates**
- **World Model (1e-4):** Supervised learning (predict known next states)
  - Clear gradients, stable training
  - Can handle higher learning rate
  - Faster convergence without instability

- **Actor/Critic (3e-5):** Reinforcement learning (maximize future rewards)
  - Noisy gradients (bootstrapping, moving targets)
  - Needs slower, more careful updates
  - Typical ratio: WM LR is 3-5× actor LR

**2. Independent Update Schedules**
```python
for step in training:
    world_model_loss.backward()
    world_model_optimizer.step()  # Update WM

    if step > warmup:
        actor_loss.backward()
        actor_optimizer.step()  # Update actor

        critic_loss.backward()
        critic_optimizer.step()  # Update critic
```

Allows:
- Training WM before actor (warmup)
- Multiple WM updates per actor update (if needed)
- Different update frequencies for each component

**3. Clean Gradient Separation**

Without separate optimizers, all parameters would get gradients from all losses:
```python
# BAD: Single optimizer
total_loss = wm_loss + actor_loss + critic_loss
total_loss.backward()
optimizer.step()  # Updates everything with blended gradients
```

Problem: World model gets gradients from actor loss, which is:
- Noisy (RL gradients)
- Unrelated to prediction accuracy
- Could hurt WM quality

### Alternative Approaches

#### Alternative 1: Single Optimizer with Weighted Losses
```python
optimizer = Adam(all_parameters, lr=1e-4)
total_loss = wm_loss + α·actor_loss + β·critic_loss
total_loss.backward()
optimizer.step()
```

**Pros:**
- Simpler code
- One LR to tune
- Automatic gradient balancing

**Cons:**
- No independent LRs (suboptimal for each component)
- α, β become critical hyperparameters
- No warmup schedule possible
- Gradient pollution (WM sees RL gradients)

**When to use:**
- Small projects
- When simplicity > performance
- If you find good α, β values

#### Alternative 2: Single Optimizer with Gradient Masking
```python
optimizer = Adam(all_parameters, lr=1e-4)

# Update WM
wm_loss.backward()
zero_grad(actor_params + critic_params)  # Mask their gradients
optimizer.step()

# Update actor
actor_loss.backward()
zero_grad(wm_params + critic_params)  # Mask their gradients
optimizer.step()
```

**Pros:**
- Same effect as separate optimizers
- Single optimizer (simpler state for checkpointing)

**Cons:**
- More manual gradient management
- Still can't have different LRs
- More error-prone

**When to use:**
- If you need single optimizer for framework reasons
- Rare in practice

#### Alternative 3: Target Networks (Like DQN/DDPG)
```python
world_model = WorldModel()
world_model_target = copy.deepcopy(world_model)  # Frozen copy

# Use target for actor training
actor_loss = compute_loss(world_model_target, ...)
actor_loss.backward()

# Periodically update target
if step % target_update_freq == 0:
    world_model_target = copy.deepcopy(world_model)
```

**Pros:**
- Actor trains on stable WM (not changing under its feet)
- Reduces moving target problem
- Common in off-policy RL

**Cons:**
- Memory overhead (2× model size)
- Lag in WM improvements reaching actor
- More complex bookkeeping

**When to use:**
- Large world models (target provides stability)
- Off-policy training (like MBPO)
- If actor training unstable with current approach

### Recommended Choice

**Stay with separate optimizers** (current implementation) because:
1. ✅ Standard practice in MBRL (Dreamer, PlaNet, MBPO all use it)
2. ✅ Flexibility: Can tune each LR independently
3. ✅ Clean separation of concerns
4. ✅ Enables warmup and advanced schedules
5. ✅ Minimal code complexity for maximum benefit

---

## Decision 2: Distributional Predictions (Not Point Estimates)

### What Was Implemented

**Dynamics Model:**
```python
def forward(self, state, action) -> Normal:
    next_state_mean = self.network(torch.cat([state, action]))
    next_state_std = torch.full_like(next_state_mean, 1e-4)
    return Normal(next_state_mean, next_state_std)
```

**Reward Predictor:**
```python
def forward(self, state) -> Normal:
    reward_mean = self.mean_network(state)
    reward_std = self.std_network(state)  # or fixed
    return Normal(reward_mean, reward_std)
```

**Continue Predictor:**
```python
def forward(self, state) -> Bernoulli:
    logits = self.network(state)
    return Bernoulli(logits=logits)
```

### Why This Design?

**1. Principled Loss Functions**

With distributions:
```python
# For dynamics
predicted_dist = dynamics(state, action)
loss = -predicted_dist.log_prob(true_next_state).mean()  # NLL
```

This is **proper scoring rule**:
- Incentivizes honest uncertainty
- Penalizes overconfidence
- Handles stochastic environments naturally

Without distributions:
```python
# Point estimate
predicted_next_state = dynamics(state, action)
loss = MSE(predicted_next_state, true_next_state)
```

Problem with MSE:
- Assumes Gaussian noise
- No uncertainty quantification
- Struggles with multimodal dynamics (e.g., can go left OR right)

**2. Exploration in Imagination**

With distributions, during imagination:
```python
# Sample diverse rollouts
for i in range(num_samples):
    next_state = dynamics_dist.sample()  # Different each time
    # Explore multiple possible futures
```

Without distributions:
```python
next_state = dynamics(state, action)  # Always same
# Deterministic imagination = no diversity
```

**3. Uncertainty-Aware Planning**

```python
dynamics_dist = dynamics(state, action)
mean = dynamics_dist.mean  # Expected outcome
std = dynamics_dist.std    # Uncertainty

if std > threshold:
    # High uncertainty, explore carefully
    use_conservative_policy()
else:
    # Confident prediction, exploit
    use_greedy_policy()
```

### Alternative Approaches

#### Alternative 1: Pure Point Estimates
```python
class DeterministicDynamics:
    def forward(self, state, action):
        return self.network(torch.cat([state, action]))
        # Returns tensor, not distribution

    def dynamics_loss(self, state, action, next_state):
        predicted = self.forward(state, action)
        return F.mse_loss(predicted, next_state)
```

**Pros:**
- Simpler implementation
- Faster inference (no sampling)
- Lower memory (no std parameters)

**Cons:**
- No uncertainty information
- Can't handle stochastic dynamics well
- MSE assumes Gaussian errors
- No principled way to handle multi-modal outcomes

**When to use:**
- Deterministic environments (GridWorld, simple control)
- When speed > accuracy
- Quick prototyping

**Example environments:**
- CartPole: Nearly deterministic, point estimate OK
- Atari Pong: Deterministic, point estimate OK
- Robotics with noise: Need distributions
- Dice games: DEFINITELY need distributions

#### Alternative 2: Ensemble of Point Estimates
```python
class EnsembleDynamics:
    def __init__(self, n_models=5):
        self.models = [DynamicsModel() for _ in range(n_models)]

    def forward(self, state, action):
        predictions = [model(state, action) for model in self.models]
        mean = torch.stack(predictions).mean(dim=0)
        std = torch.stack(predictions).std(dim=0)  # Uncertainty!
        return Normal(mean, std)
```

**Pros:**
- Captures uncertainty via disagreement
- More robust (less likely to overfit)
- Can detect out-of-distribution states (high std)

**Cons:**
- 5× memory and compute
- Still uses point estimates internally
- Training complexity (need bootstrapping)

**When to use:**
- High-stakes applications (robotics, finance)
- When uncertainty crucial for safety
- MBPO paper uses ensembles successfully

#### Alternative 3: Mixture Models
```python
class MixtureDynamics:
    def forward(self, state, action):
        # Predict K possible outcomes
        means = [self.mean_net_i(state, action) for i in range(K)]
        stds = [self.std_net_i(state, action) for i in range(K)]
        weights = F.softmax(self.weight_net(state, action))

        # Return mixture of Gaussians
        return MixtureGaussian(means, stds, weights)
```

**Pros:**
- Can model multi-modal dynamics (multiple outcomes)
- Example: Ball bouncing (could go left OR right)
- More expressive than single Gaussian

**Cons:**
- Much more complex
- Harder to train (EM-style algorithms)
- Overkill for most tasks

**When to use:**
- Truly multi-modal dynamics
- Complex stochastic environments
- Research projects exploring expressiveness

### Recommended Choice

**Stay with single Gaussian distributions** (current implementation) because:
1. ✅ Good balance of expressiveness and simplicity
2. ✅ NLL loss is principled and works well
3. ✅ Uncertainty information available
4. ✅ Standard in MBRL (Dreamer, PlaNet use variants)

**Consider upgrades:**
- If training unstable → Try ensembles (5 models)
- If environment truly multi-modal → Try mixture models
- If deterministic and need speed → Try point estimates with MSE

---

## Decision 3: λ-Returns for Policy Targets

### What Was Implemented
```python
def compute_lambda_returns(rewards, values, continues, gamma, lambda_):
    returns = torch.zeros_like(rewards)
    last_value = values[:, -1]
    next_return = last_value

    for t in reversed(range(horizon)):
        discount = gamma * continues[:, t]
        reward = rewards[:, t]
        value = values[:, t]

        # TD-λ mixing
        next_return = reward + discount * (
            (1 - lambda_) * value + lambda_ * next_return
        )
        returns[:, t] = next_return

    return returns
```

### Why This Design?

**The Bias-Variance Tradeoff**

**Pure Monte Carlo (λ=1):**
```
return_t = r_t + r_{t+1} + r_{t+2} + ... + r_T
```
- ✅ Unbiased (uses actual rewards)
- ❌ High variance (long trajectory, lots of noise)
- ❌ Slow credit assignment (need full trajectory)

**Pure TD (λ=0):**
```
return_t = r_t + γ * V(s_{t+1})
```
- ✅ Low variance (bootstrap with value function)
- ❌ Biased (value function might be wrong)
- ✅ Fast credit assignment (single step)

**TD-λ (0 < λ < 1):**
```
return_t = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * return_{t+1}]
```
- ✅ Balanced bias-variance
- ✅ Flexible (tune λ for environment)
- ✅ Standard in RL (PPO, IMPALA use this)

**Typical λ values:**
- λ = 0.95 (common default): Mostly bootstrap, some MC
- λ = 0.99 (high): More MC, less bootstrap
- λ = 0.90 (low): More bootstrap, less MC

### Alternative Approaches

#### Alternative 1: Pure Monte Carlo
```python
def monte_carlo_returns(rewards, gamma):
    returns = torch.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns
```

**When to use:**
- Short episodes (< 100 steps)
- Simple environments (high signal-to-noise)
- When value function unreliable

**Pros:**
- Simple, no value function needed
- Unbiased

**Cons:**
- High variance → slow learning
- Needs full trajectories

#### Alternative 2: N-Step Returns
```python
def n_step_returns(rewards, values, gamma, n=5):
    # Use next n rewards, then bootstrap
    return_t = sum(gamma**i * rewards[t+i] for i in range(n))
    return_t += gamma**n * values[t+n]
    return return_t
```

**When to use:**
- Fixed-horizon problems
- When you want control over bias-variance (tune n)

**Pros:**
- Simple to understand
- Fixed computation (no full trajectory needed)

**Cons:**
- Less flexible than λ-returns
- Must choose n (another hyperparameter)

#### Alternative 3: Generalized Advantage Estimation (GAE)
```python
def compute_gae(rewards, values, gamma, lambda_):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        last_advantage = delta + gamma * lambda_ * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values  # Add baseline back
    return returns, advantages
```

**When to use:**
- Policy gradient methods (PPO, TRPO)
- When you want advantages (not just returns)
- Standard in model-free RL

**Pros:**
- Advantages are better for policy gradients
- Same bias-variance control as λ-returns
- Very popular (PPO default)

**Cons:**
- Slightly more complex
- Requires value function at each step

**Your friend's choice:**
Used λ-returns (not GAE). Why?

In world models:
- Return directly from imagination is intuitive
- Value estimates available throughout rollout
- λ-returns sufficient for policy gradient

GAE would also work, but λ-returns simpler for this use case.

### Recommended Choice

**Stay with λ-returns** (current implementation) because:
1. ✅ Good bias-variance tradeoff
2. ✅ Standard in both model-free and model-based RL
3. ✅ Easy to tune (λ=0.95 works for most tasks)
4. ✅ Simpler than GAE, similar performance

**Consider alternatives:**
- If episodes very short (< 50 steps) → Try pure MC
- If want advantages explicitly → Switch to GAE
- If want fixed horizon → Try n-step returns

---

## Decision 4: Separate Representation Learner

### What Was Implemented
```
Observations → Encoder → Features → Repr Learner → States
                         [256]                     [64]
```

Instead of:
```
Observations → Encoder → States
                         [64]
```

### Why This Design?

**1. Modularity**

Can swap representation learner without touching encoder:
```yaml
# Config 1: Autoencoder
representation_learner:
  type: "mlp_autoencoder"
  representation_dim: 64

# Config 2: VAE
representation_learner:
  type: "vae"
  latent_dim: 64
  beta: 1.0

# Config 3: Contrastive
representation_learner:
  type: "contrastive"
  projection_dim: 64
```

**2. Explicit Bottleneck**

Forces compression:
```
Features [256] → Must compress → States [64]
                  ↓
            Information bottleneck
                  ↓
          Forces learning of important features
```

Without bottleneck:
```
Observations → Encoder → States [256]
                         No compression, might learn identity
```

**3. Different Objectives**

- **Encoder:** Extract useful features from observations
  - Trained on: All downstream losses
  - Goal: Rich, expressive features

- **Repr Learner:** Compress to latent space
  - Trained on: Reconstruction, dynamics, reward
  - Goal: Compact, predictable representation

**4. Future Extensions**

Easy to add advanced repr learners:
- **RSSM** (Recurrent State Space Model, from Dreamer)
  - Stochastic latents
  - Temporal coherence
  - Requires separate component

- **Vector Quantization** (VQ-VAE style)
  - Discrete latent codes
  - Easier to predict with transformers
  - Clean interface via repr learner

### Alternative Approaches

#### Alternative 1: Single Encoder (No Repr Learner)
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim, state_dim):
        self.conv = nn.Sequential(...)  # Image processing
        self.fc = nn.Linear(conv_output, state_dim)

    def forward(self, obs):
        features = self.conv(obs)
        states = self.fc(features)
        return states  # Direct to state space
```

**Pros:**
- Simpler (one less component)
- Fewer parameters
- Slightly faster

**Cons:**
- Less modular (can't swap repr learner)
- Can't add VAE/RSSM without refactoring
- No explicit bottleneck control

**When to use:**
- Simple environments (state-based, not images)
- When you're certain about representation type
- Prototyping

#### Alternative 2: Implicit Repr Learner
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim, feature_dim, state_dim):
        self.feature_net = nn.Sequential(...)  # obs → features
        self.state_net = nn.Linear(feature_dim, state_dim)  # features → states

    def get_features(self, obs):
        return self.feature_net(obs)

    def get_states(self, obs):
        return self.state_net(self.get_features(obs))
```

**Pros:**
- Bottleneck controlled
- Still single component (simpler registration)

**Cons:**
- Less explicit than separate component
- Harder to swap (need to redefine encoder)
- Couples feature extraction with compression

**When to use:**
- If you value simplicity > extreme modularity
- Reasonable middle ground

#### Alternative 3: Hierarchical Repr Learners
```python
observation → Encoder → features [256]
                            ↓
                Repr Learner 1 → intermediate [128]
                            ↓
                Repr Learner 2 → states [64]
```

**Pros:**
- Multiple levels of abstraction
- Useful for complex observations (e.g., videos)

**Cons:**
- Overkill for most tasks
- Hard to train (multiple bottlenecks)

**When to use:**
- Very high-dim observations (videos)
- Hierarchical world models (rare)

### Recommended Choice

**Stay with separate repr learner** (current implementation) because:
1. ✅ Matches your modular architecture vision
2. ✅ Easy to experiment (swap VAE, RSSM, etc.)
3. ✅ Minimal complexity cost
4. ✅ Future-proof (can add advanced methods)

**Simplify if:**
- Prototype phase → Merge into encoder temporarily
- Very simple env → Single encoder might suffice

---

## Decision 5: Imagination-Based Policy Learning

### What Was Implemented
```python
# Don't use real environment for policy learning
# Instead: Imagine trajectories using world model

imagined = rollout_imagination(
    initial_states=states_from_real_data,  # Start from real
    length=15,  # Imagine 15 steps ahead
    with_grad=True  # Backprop through imagination
)

returns = compute_lambda_returns(imagined['rewards'], ...)
actor_loss = -returns.mean()  # Maximize imagined returns
actor_loss.backward()  # Gradients through world model
```

### Why This Design?

**Sample Efficiency**

With imagination:
```
Collect 100 real transitions
    ↓
Generate 1500 imagined transitions (100 × 15)
    ↓
Train policy on 1500 transitions
```

Without imagination (model-free):
```
Collect 100 real transitions
    ↓
Train policy on 100 transitions
    ↓
Need 15× more real data for same amount of learning
```

**Key insight:** Real environment expensive, world model cheap

**Differentiable Gradients**

Imagination allows end-to-end gradients:
```
action → dynamics → next_state → reward → return → loss
  ↑         ↑           ↑          ↑         ↑
  └─────────┴───────────┴──────────┴─────────┘
            All differentiable!
```

Real environment:
```
action → environment → next_state → reward
            ↑
      Black box (no gradients)
```

Must use policy gradient (higher variance).

### Alternative Approaches

#### Alternative 1: Model-Free Only (No Imagination)
```python
# Collect real transitions
real_trajectory = collect_from_environment()

# Train directly on real data
returns = compute_returns(real_trajectory)
actor_loss = -returns.mean()
```

**Pros:**
- Simple (no world model needed)
- Unbiased (real environment, no model error)
- Works for any environment

**Cons:**
- Sample inefficient (need lots of real data)
- Can't backprop through environment
- Slow in expensive environments (robotics)

**When to use:**
- Simple, fast environments (Atari in simulation)
- When world model quality poor
- Baseline comparison

**Example:**
- PPO on CartPole: 100K steps to solve
- Dreamer on CartPole: 10K steps to solve (10× more efficient)

#### Alternative 2: Dyna-Style (Mix Real and Imagined)
```python
# Collect real data
real_batch = env.collect(batch_size=256)

# Generate imagined data
imagined_batch = world_model.imagine(batch_size=1024)

# Train on both
combined = mix(real_batch, imagined_batch, ratio=0.2)
actor_loss = compute_loss(combined)
```

**Pros:**
- Sample efficient (uses imagination)
- Grounded in reality (real data prevents model exploitation)
- Flexible ratio (tune real vs imagined)

**Cons:**
- More complex training loop
- Need to tune mixing ratio
- Still need some real data

**When to use:**
- MBPO (Model-Based Policy Optimization) uses this
- When world model has systematic errors
- Hybrid approach

**Typical ratios:**
- 20% real, 80% imagined (aggressive)
- 50% real, 50% imagined (balanced)

#### Alternative 3: MPC (No Policy Learning)
```python
def act(state):
    # Don't learn policy, just plan online
    best_action = None
    best_return = -inf

    for _ in range(num_samples):
        actions = sample_random_sequence(horizon=15)
        imagined = rollout(state, actions, world_model)
        returns = compute_returns(imagined)

        if returns > best_return:
            best_action = actions[0]
            best_return = returns

    return best_action
```

**Pros:**
- No policy learning (simpler)
- Always uses latest world model
- Can handle changing environments

**Cons:**
- Slow at test time (plan every step)
- Need good world model
- Doesn't amortize planning cost

**When to use:**
- Small action spaces (easy to search)
- Accurate world models
- Offline planning acceptable (robotics)

#### Alternative 4: Dreamer V3 (Symlog Imagination)
```python
# Use symlog transform for stable imagination
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

imagined_rewards_symlog = symlog(imagined_rewards)
returns = compute_lambda_returns(imagined_rewards_symlog, ...)
returns_real = symexp(returns)  # Transform back
actor_loss = -returns_real.mean()
```

**Pros:**
- More stable for long-horizon imagination
- Handles large rewards better
- State-of-the-art (Dreamer V3 results)

**Cons:**
- More complex
- Another transform to tune

**When to use:**
- Imagination horizon > 15 steps
- If training unstable
- Pushing state-of-the-art

### Recommended Choice

**Stay with pure imagination** (current implementation) because:
1. ✅ Sample efficient (key benefit of world models)
2. ✅ Differentiable (better gradients than policy gradient)
3. ✅ Standard in MBRL (Dreamer, PlaNet)
4. ✅ Simpler than Dyna-style mixing

**Consider alternatives:**
- If model exploiting world model errors → Add Dyna mixing
- If test-time speed not important → Try MPC
- If long horizons unstable → Add symlog transforms

---

## Decision 6: Warmup Schedule (Train WM First)

### What Was Implemented
```python
warmup_steps = 5000  # Config param

if env_steps < warmup_steps:
    # Train world model only
    actor_updates = 0
    critic_updates = 0
    wm_updates = 1
else:
    # Train everything
    actor_updates = 1
    critic_updates = 1
    wm_updates = 1
```

### Why This Design?

**The Cold Start Problem**

At beginning of training:
```
World Model: Random → Predicts nonsense
Policy: Random → Explores randomly
```

If we train policy on bad WM predictions:
```
Imagined trajectory (wrong):
  Start → Go left → WM predicts +10 reward (WRONG!)
  Policy learns: "Go left is great!"

Reality:
  Start → Go left → Falls off cliff → -10 reward

Result: Policy learns wrong behavior, hard to unlearn
```

**Warmup solves this:**
```
Steps 0-5000:
  - Collect random exploration data
  - Train world model on real data
  - Don't train policy (let WM improve first)

Steps 5000+:
  - World model reasonably accurate
  - NOW train policy on imagination
  - Policy learns correct behaviors
```

### Alternative Approaches

#### Alternative 1: No Warmup
```python
# Train everything from step 0
for step in training:
    actor_loss.backward()
    critic_loss.backward()
    wm_loss.backward()
```

**Pros:**
- Simpler code
- Faster to "start" learning

**Cons:**
- Policy might learn wrong behaviors early
- Harder to recover from bad start
- More hyperparameter sensitive

**When to use:**
- Very simple environments
- Strong domain randomization (policy must be robust anyway)
- If warmup too slow for your patience

#### Alternative 2: Gradual Mixing
```python
# Slowly increase reliance on imagination
real_ratio = max(0, 1.0 - (step / warmup_steps))
imagined_ratio = 1 - real_ratio

critic_target = real_ratio * real_return + imagined_ratio * imagined_return
```

**Pros:**
- Smoother transition
- Always some real data grounding
- More robust

**Cons:**
- More complex
- Need to implement mixed targets

**When to use:**
- MBPO uses this (critic_real_return_mix parameter)
- When world model has persistent errors

Your friend included this!
```python
critic_real_mix = config.get('critic_real_return_mix', 0.0)
targets = (1 - critic_real_mix) * imagined_returns + critic_real_mix * real_returns
```

Currently set to 0.0 (pure imagination), but can tune!

#### Alternative 3: Conservative Warmup
```python
# Multiple phases
if step < 5000:
    wm_updates, actor_updates, critic_updates = 1, 0, 0
elif step < 10000:
    wm_updates, actor_updates, critic_updates = 2, 0, 1  # Critic first
elif step < 15000:
    wm_updates, actor_updates, critic_updates = 2, 1, 1  # Now actor
else:
    wm_updates, actor_updates, critic_updates = 1, 1, 1  # Normal
```

**Pros:**
- Even more careful
- Train value function before policy (helps actor)

**Cons:**
- Very conservative (slow)
- Many phases to tune

**When to use:**
- Research code
- Very sensitive environments
- Debugging training issues

### Recommended Choice

**Stay with simple warmup** (current implementation) because:
1. ✅ Prevents policy from learning on bad WM
2. ✅ Simple (one threshold)
3. ✅ Standard in MBRL
4. ✅ 5K steps reasonable (< 1 minute in most envs)

**Tune warmup_steps:**
- Simple env (CartPole): 1K sufficient
- Complex env (Humanoid): 10K+ needed
- Rule of thumb: Enough for dynamics_loss < 0.5

---

## Summary Table: When to Change Each Decision

| Decision | Current | Change If... | Change To |
|----------|---------|--------------|-----------|
| **Optimizers** | 3 separate | Need extreme simplicity | Single optimizer |
| **Distributions** | Gaussian | Env deterministic | Point estimates |
| | | Env multi-modal | Mixture models |
| | | Need robustness | Ensembles |
| **Returns** | λ-returns (0.95) | Episodes very short | Pure MC (λ=1.0) |
| | | Want advantages | GAE |
| | | High bias observed | Higher λ (0.99) |
| **Repr Learner** | Separate | Simplicity critical | Merge into encoder |
| | | Need stochastic | Add VAE/RSSM |
| **Imagination** | Pure (15 steps) | WM exploiting errors | Add Dyna mixing |
| | | Long horizon unstable | Add symlog |
| | | Need interpretability | Try MPC |
| **Warmup** | 5K steps | Simple environment | Reduce to 1K |
| | | Complex environment | Increase to 10K |
| | | WM has errors | Add gradual mixing |

This completes the deep dive into design decisions!
