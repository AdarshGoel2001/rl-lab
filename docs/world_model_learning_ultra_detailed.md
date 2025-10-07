# World Model Learning: Ultra-Detailed First Principles Guide

## Table of Contents
1. [Overview: What is a World Model?](#overview)
2. [All Losses Explained](#all-losses)
3. [Gradient Flow Diagram](#gradient-flow)
4. [Component Learning Interfaces](#component-interfaces)
5. [Design Decisions & Reasoning](#design-decisions)
6. [Signal Flow Analysis](#signal-analysis)

---

## Overview: What is a World Model?

### The Core Idea (Absolute First Principles)

Imagine you're learning to play chess. You could learn by:
1. **Playing many games** and remembering what worked (Model-Free Learning)
2. **Building a mental model** of how pieces move, then planning in your head (Model-Based Learning)

A **world model** is option #2 for AI. Instead of just memorizing "action A in situation B gives reward R", the AI learns:
- **How the world works** (dynamics model)
- **What rewards to expect** (reward predictor)
- **When episodes end** (continue predictor)

Then it uses this mental model to **imagine future scenarios** and plan better actions.

### Why This Matters

**Model-Free** (like PPO):
- Learns directly from real experience
- Needs LOTS of real environment interactions (sample inefficient)
- Simple but can be slow

**Model-Based** (like Dreamer):
- Learns a model of the environment
- Can **imagine** practicing without real interactions (sample efficient)
- More complex but potentially much faster

### Your Implementation Structure

```
Real Experience → World Model Learning → Imagination → Policy Learning
     ↓                    ↓                  ↓              ↓
  Replay Buffer    (understand world)  (practice in head) (improve actions)
```

---

## All Losses Explained

Your friend implemented **10 different loss terms**. Let's understand each one from scratch.

### Phase 1: World Model Learning (Learning to Predict)

These losses teach the AI to understand how the world works.

#### Loss 1: Dynamics Loss
**File:** `src/components/dynamics/deterministic_mlp.py:135-161`
**Line in paradigm:** `paradigm.py:743-750`

**What it learns:** "If I'm in state S and take action A, what will the next state S' be?"

**Mathematical form:**
```
dynamics_loss = MSE(predicted_next_state, actual_next_state)
             = mean((f(s, a) - s')²)
```

**First principles explanation:**
- The dynamics model `f` is a neural network
- Input: Current state `s` + action `a` (concatenated)
- Output: Predicted next state `f(s, a)`
- Loss: Mean squared error between prediction and reality
- **Why MSE?** We want the prediction to be as close as possible to the true next state

**What gradient does:**
```
∂dynamics_loss/∂θ_dynamics = ∂MSE/∂f(s,a) · ∂f(s,a)/∂θ_dynamics
                            = 2(f(s,a) - s') · [gradient through network]
```
This tells each neuron: "adjust your weights to predict s' more accurately"

**Example:**
- Current state: Cart at position 0.0, velocity 0.5
- Action: Push left (-1)
- True next state: Cart at position -0.05, velocity 0.3
- Prediction: Cart at position -0.03, velocity 0.4
- Error: (0.02, 0.1)
- Gradient says: "Predict more negative position and less velocity next time"

#### Loss 2: Reward Loss
**File:** `src/components/reward_predictors/mlp_reward.py:121-150`
**Line in paradigm:** `paradigm.py:758-764`

**What it learns:** "In state S, what reward will I receive?"

**Mathematical form:**
```
reward_loss = -log p(r | s)    [negative log likelihood]
            = NLL(predicted_dist, actual_reward)
```

**First principles explanation:**
- Reward predictor outputs a **probability distribution** (Gaussian)
- It predicts both mean μ(s) and std σ(s)
- Loss = negative log probability of the actual reward under this distribution
- **Why NLL instead of MSE?** NLL captures uncertainty. If rewards are noisy, the model learns higher σ

**Distribution details:**
```python
reward_dist = Normal(μ(s), σ(s))
p(r | s) = (1/√(2πσ²)) * exp(-(r - μ)² / (2σ²))
log p(r | s) = -0.5 * log(2πσ²) - (r - μ)² / (2σ²)
reward_loss = -log p(r | s) = 0.5 * log(2πσ²) + (r - μ)² / (2σ²)
```

**What gradient does:**
```
∂reward_loss/∂μ = (μ - r) / σ²     [push mean toward actual reward]
∂reward_loss/∂σ = (r - μ)²/σ³ - 1/σ [adjust uncertainty estimate]
```

**Example:**
- State: Cart at position 0.5, upright
- Actual reward: +1.0 (stayed upright)
- Prediction: μ = 0.7, σ = 0.3
- Error: Prediction too low
- Gradient says: "Increase μ, you're underestimating this state's value"

#### Loss 3: Continue Loss
**File:** `src/components/continue_predictors/mlp_continue.py:94-135`
**Line in paradigm:** `paradigm.py:773-779`

**What it learns:** "In state S, will the episode continue (1) or terminate (0)?"

**Mathematical form:**
```
continue_loss = Binary Cross Entropy
              = -[c * log(p) + (1-c) * log(1-p)]
```
where:
- c = actual continue flag (1 or 0)
- p = predicted probability of continuing

**First principles explanation:**
- This is binary classification
- Output: Sigmoid(logit) → probability in [0, 1]
- **Why BCE?** Standard loss for binary classification, penalizes confident wrong predictions heavily

**What gradient does:**
```
∂BCE/∂logit = p - c    [if predicted 0.8 but should be 1, gradient = -0.2]
```

**Example:**
- State: Cart at position 0.9, tilting far
- Actual: Episode terminated (c = 0)
- Prediction: p = 0.7 (70% chance of continuing)
- Error: Model thinks episode will continue but it ended
- Gradient says: "Lower your prediction, this state is dangerous"

#### Loss 4: Decoder Reconstruction Loss
**File:** `paradigm.py:787-793`

**What it learns:** "Can I reconstruct the original observation from my learned state representation?"

**Mathematical form:**
```
decoder_loss = MSE(decoder(state), original_observation)
```

**First principles explanation:**
- **Why this exists:** Ensures the learned state representation captures important information
- If the state can reconstruct the observation, it must contain useful info about the world
- This is the "autoencoder" part of the system

**Flow:**
```
observation → encoder → state → decoder → reconstructed_observation
              ↓                    ↓
         [features]           [should match observation]
```

**What gradient does:**
- Flows backward through decoder → state → encoder
- Forces encoder to preserve information needed for reconstruction
- Prevents state from becoming "meaningless" numbers

**Example:**
- Original observation: Image of cart [84x84 pixels]
- State: 64-dimensional vector
- Reconstructed: Image [84x84 pixels]
- Error: Pixel-wise differences
- Gradient says: "Adjust state to preserve more visual information"

#### Loss 5: Multi-step State Loss (Optional)
**File:** `paradigm.py:806-843`

**What it learns:** "Can I predict multiple steps ahead using my dynamics model?"

**Mathematical form:**
```
For each step t from 0 to T-1:
    predicted_state_t+1 = dynamics_model(predicted_state_t, action_t)
    multi_step_loss += MSE(predicted_state_t+1, actual_state_t+1) * mask_t
```

**First principles explanation:**
- **Why this exists:** Dynamics model should work for multi-step prediction, not just one-step
- Without this, model might work for one step but compound errors over time
- **Mask:** Stop gradient flow after episode termination (indicated by continues = 0)

**Key detail (line 813-836):**
```python
for t in range(sequence_length - 1):
    # Predict next state
    predicted_state = dynamics_model(current_state, action_t)

    # Compare to actual
    loss += MSE(predicted_state, actual_state_t+1) * mask

    # If episode continues, use prediction for next step
    # If episode ended, reset to actual state (don't compound errors)
    if continues_t:
        current_state = predicted_state  # Use our prediction
    else:
        current_state = actual_state_t+1  # Episode ended, reset to truth
        mask = 0  # Stop accumulating loss after termination
```

**Why the mask matters:**
- Episode ends at step 5 (continues_5 = 0)
- Steps 0-5: Count prediction errors
- Steps 6+: Don't penalize, episode is over in real data
- Prevents model from learning meaningless predictions after termination

### Phase 2: Policy Learning (Learning to Act)

These losses teach the AI to choose good actions using the learned world model.

#### Loss 6: Actor Loss (Policy Loss)
**File:** `paradigm.py:916-918, 942-950`

**What it learns:** "What actions should I take to maximize rewards?"

**Mathematical form:**
```
actor_loss = -mean(returns)  [negative because we want to maximize]
           = -mean(Σ γ^t * r_t)  [with entropy bonus]

With entropy:
actor_loss = -mean(returns) - α * entropy
```

**First principles explanation:**

This is the **core** of reinforcement learning. Let's break it down completely:

**Step 1: Imagination Rollout** (line 852-857)
```python
imagined = rollout_imagination(
    initial_states=states_from_real_data,
    length=15,  # Imagine 15 steps ahead
    with_grad=True  # Need gradients to learn
)
```

What happens in imagination:
1. Start from real state (e.g., cart at position 0.1)
2. Policy chooses action: a₀ (e.g., push left)
3. Dynamics predicts next state: s₁ = f(s₀, a₀)
4. Reward predictor estimates: r₀ = R(s₀)
5. Repeat for 15 steps, all in the AI's "mind"

**Step 2: Compute Returns** (line 1100-1118)
```python
returns = compute_lambda_returns(
    rewards=imagined_rewards,
    values=imagined_values,
    continues=imagined_continues,
    gamma=0.99,
    lambda_=0.95
)
```

Returns = "How good is this trajectory?" Uses **λ-returns** (TD-λ):
```
return_t = r_t + γ * continues_t * [(1-λ) * V(s_t+1) + λ * return_t+1]
```

This blends:
- **Bootstrapping** (1-λ): Trust value function V(s_t+1)
- **Monte Carlo** (λ): Trust actual reward trajectory

**Step 3: Compute Loss**
```
actor_loss = -mean(returns)
```

**Why negative?**
- Gradient descent minimizes loss
- We want to **maximize** returns
- Minimizing `-returns` = Maximizing `returns`

**What gradient does:**
```
∂actor_loss/∂θ_policy = ∂(-returns)/∂θ_policy
                      = -∂returns/∂θ_policy
                      = -∂(Σ rewards)/∂θ_policy
                      = -Σ (∂reward_t/∂action_t) * (∂action_t/∂θ_policy)
```

The gradient flows through:
1. Policy network → actions
2. Actions → dynamics model → next states
3. States → reward predictor → rewards
4. Sum rewards → returns

**Entropy bonus** (line 942-950):
```
entropy = -Σ p(a) log p(a)  [measure of randomness]
actor_loss = -returns - α * entropy
```

**Why entropy?**
- Prevents policy from becoming too deterministic too early
- Encourages exploration
- α = 0.01 typically (small bonus for diversity)

**Example:**
- Imagined trajectory: Cart moves right, stays balanced, +15 total reward
- Returns: High (good trajectory)
- Actor loss: -15 (negative of returns)
- Gradient: Increase probability of actions that led to this trajectory
- Entropy: If policy always chooses "push right", entropy = 0 (bad, no exploration)

#### Loss 7: Critic Loss (Value Function Loss)
**File:** `paradigm.py:920-930`

**What it learns:** "How much total reward will I get from state S if I follow my current policy?"

**Mathematical form:**
```
critic_loss = MSE(V(s), returns)
            = mean((V(s) - returns)²)
```

**First principles explanation:**

The critic is a **judge** that evaluates states. While the actor asks "what should I do?", the critic asks "how good is this state?"

**Why we need it:**
- Actor needs to know which states are good/bad
- Returns tell us "this trajectory got reward R"
- But was it because of good actions or lucky states?
- Value function V(s) learns "inherent goodness of state s"

**Training process:**
1. Use current policy to imagine trajectory
2. Compute actual returns from that trajectory
3. Train V(s) to predict those returns
4. Next iteration, V(s) gives better feedback to actor

**What gradient does:**
```
∂critic_loss/∂θ_critic = 2 * (V(s) - returns) * ∂V(s)/∂θ_critic
```

If V(s) = 10 but returns = 15:
- Error = -5
- Gradient says: "Increase your prediction, this state is better than you thought"

**TD Error monitoring** (line 933-940):
```python
td_error = V(s) - returns
```
This measures how wrong the critic is. Tracked but not used in loss directly.

---

## Gradient Flow Diagram

See `docs/world_model_gradient_flow.svg` for the visual diagram.

### Three Separate Optimization Phases

Your friend made a crucial design decision: **Three separate optimizers** with **three separate backward passes**.

```python
# Line 126-143
world_model_optimizer = Adam([encoder, repr_learner, dynamics, reward_pred, continue_pred, decoder])
actor_optimizer = Adam([policy_head])
critic_optimizer = Adam([value_function])
```

**Why separate?** (This is CRITICAL to understand)

1. **World Model** learns to **predict** the environment
2. **Actor** learns to **choose** good actions
3. **Critic** learns to **evaluate** states

These are **different objectives** and can benefit from:
- Different learning rates
- Different update frequencies
- Different gradient flow paths

### Gradient Flow Paths

#### Path 1: World Model Gradients
```
world_model_loss = dynamics_loss + reward_loss + continue_loss + decoder_loss
                   ↓
world_model_loss.backward()  [Line 1030]
                   ↓
Gradients flow to: encoder, repr_learner, dynamics, reward_pred, continue_pred, decoder
                   ↓
world_model_optimizer.step()  [Line 1063]
```

**Key detail** (line 1030):
```python
world_model_loss.backward(retain_graph=True)
```

`retain_graph=True` means: "Keep the computation graph after backward pass"

**Why?** Because we need to do MORE backward passes (actor and critic) through parts of this same graph.

**Gradient isolation** (line 1026-1052):
After world model backward, your friend **saves and restores** gradients:
```python
# Save world model gradients
saved_grads = [param.grad.clone() for param in world_model_params]

# Do actor backward (might contaminate world model grads)
actor_loss.backward(retain_graph=True)

# Restore pure world model gradients
for param, saved_grad in zip(world_model_params, saved_grads):
    param.grad = saved_grad
```

**Why this complexity?**
- Actor loss computation uses world model components (dynamics, reward predictor)
- Actor.backward() would add gradients to world model parameters
- But world model should only update based on world_model_loss
- Solution: Save WM grads, let actor compute its grads, restore WM grads, update separately

#### Path 2: Actor Gradients
```
imagined_rollout (uses dynamics + reward predictors)
        ↓
returns = f(imagined_rewards, imagined_values)
        ↓
actor_loss = -mean(returns) - entropy_bonus
        ↓
actor_loss.backward(retain_graph=True)  [Line 1038]
        ↓
Gradients flow THROUGH world model (read-only) TO policy_head (write)
        ↓
actor_optimizer.step()  [Line 1065]
```

**Critical insight:**
```
actor_loss depends on:
  → policy choices (∂loss/∂policy ✓ update this)
  → dynamics predictions (∂loss/∂dynamics ✗ don't update, world model handles this)
  → reward predictions (∂loss/∂reward ✗ don't update, world model handles this)
```

The gradient flows THROUGH dynamics and reward predictors, but they are **not updated** by actor_optimizer.

**Why `with_grad=True` in imagination?** (line 855)
```python
imagined = rollout_imagination(initial_states, length=15, with_grad=True)
```

Without gradients through imagination, actor has no signal to improve. The gradient must flow:
```
actor_loss → returns → imagined_rewards → reward_predictor(states) → dynamics_model(state, action) → policy(action)
```

#### Path 3: Critic Gradients
```
V(imagined_states) compared to returns
        ↓
critic_loss = MSE(V, returns.detach())  [Returns detached!]
        ↓
critic_loss.backward()  [Line 1042]
        ↓
Gradients flow ONLY to value_function
        ↓
critic_optimizer.step()  [Line 1067]
```

**Key detail** (line 927):
```python
value_targets = returns_flat.detach()
```

**Why detach?**
- Critic should predict returns, not change returns
- `detach()` stops gradients from flowing backward through returns
- Only value_function parameters get updated

### Update Schedule

**Multiple updates per batch** (line 972-975):
```python
world_model_updates = 1  # Can be > 1
actor_updates = 1
critic_updates = 1
```

**Warm-up schedule** (line 986-988):
```python
if env_steps < warmup_steps:
    actor_updates = 0
    critic_updates = 0
    # Only train world model initially
```

**Why warmup?**
- World model needs to learn environment first
- No point training actor on bad dynamics predictions
- Typical warmup: 1000-10000 steps of world-model-only training

### Gradient Clipping (line 1054-1060)
```python
torch.nn.utils.clip_grad_norm_(world_model_params, max_grad_norm)
torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
```

**Why?**
- Prevents "gradient explosion"
- If gradients get huge, weight updates are huge, training becomes unstable
- `max_grad_norm=1.0` typical: If total gradient norm > 1.0, scale all gradients down

---

## Component Learning Interfaces

### Interface 1: Encoder → Representation Learner
```python
# paradigm.py:562-581
features = encoder(observations)              # Shape: [batch, feature_dim]
states = representation_learner.encode(features)  # Shape: [batch, state_dim]
```

**Contract:**
- Encoder outputs raw features (e.g., CNN activations)
- Repr learner compresses to latent states
- **Gradient flows backward:** state → repr learner → features → encoder

**Signal quality check:**
- Encoder gets gradients from: decoder reconstruction, all downstream tasks
- If decoder_loss high, encoder must preserve more information
- If dynamics_loss high, states must be more predictable

### Interface 2: Representation Learner → Dynamics Model
```python
# paradigm.py:739-750
states_input = states  # Shape: [batch, state_dim]
actions_input = actions  # Shape: [batch, action_dim]
next_states_target = next_states  # Shape: [batch, state_dim]

dynamics_dist = dynamics_model(states_input, actions_input)
dynamics_loss = NLL(dynamics_dist, next_states_target)
```

**Contract:**
- States must be in a space where dynamics are predictable
- Dynamics model outputs distribution (not point estimate)
- **Gradient signal:** dynamics_loss → dynamics_model AND backwards to representation_learner

**Signal quality check:**
- If dynamics_loss high, either:
  1. Dynamics model is too simple (increase capacity)
  2. State representation is not predictive (train longer or adjust repr learner)

### Interface 3: States → Reward Predictor
```python
# paradigm.py:752-764
reward_dist = reward_predictor(states)  # Output: Normal distribution
reward_loss = -reward_dist.log_prob(true_rewards).mean()
```

**Contract:**
- States should contain information about expected rewards
- Reward predictor outputs distribution to model uncertainty
- **Gradient signal:** reward_loss → reward_predictor → states → repr learner

**Signal quality check:**
- If reward_loss high but dynamics_loss low:
  - States are predictable but not reward-informative
  - May need different state representation (e.g., add bottleneck)

### Interface 4: States → Continue Predictor
```python
# paradigm.py:767-779
continue_dist = continue_predictor(states)  # Output: Bernoulli (binary)
continue_loss = -continue_dist.log_prob(continues).mean()
```

**Contract:**
- States should indicate whether episode will terminate
- Binary classification (Bernoulli distribution)
- **Gradient signal:** continue_loss → continue_predictor → states

**Signal quality check:**
- Track accuracy (line 119 in continue_predictor)
- If accuracy < 70%, states may not capture terminal conditions
- Common issue: Continue predictor sees "death" states as similar to "alive" states

### Interface 5: States → Policy Head
```python
# paradigm.py:438-442 (in imagination rollout)
action_dist = policy_head(states)
actions = action_dist.sample()  # or rsample() for continuous
```

**Contract:**
- States should contain information needed for action selection
- Policy outputs action distribution (not single action)
- **Gradient signal (during imagination):** actor_loss → returns → rewards → states → actions → policy

**Signal quality check:**
- Entropy of action distribution (line 946)
- If entropy → 0 too early, policy converged prematurely
- If entropy stays high, policy not learning (may need higher actor learning rate)

### Interface 6: States → Value Function
```python
# paradigm.py:500-502 (in imagination rollout)
values = value_function(states)
```

**Contract:**
- Value function estimates expected return from state
- Single scalar output per state
- **Gradient signal:** critic_loss → value_function (no backprop to states)

**Signal quality check:**
- TD error magnitude (line 939)
- If |TD error| large, critic is poorly calibrated
- If TD error → 0, critic might be overfitting to current policy

---

## Design Decisions & Reasoning

### Decision 1: Why distributions instead of point estimates?

**For Dynamics:**
```python
# Instead of:
next_state = dynamics_model(state, action)  # Point estimate

# Your friend uses:
next_state_dist = dynamics_model(state, action)  # Distribution
next_state = next_state_dist.mean  # or .sample()
```

**Reasoning:**
1. **Stochastic environments:** Real world has noise
2. **Uncertainty quantification:** Model can express "I'm not sure"
3. **Better loss:** NLL loss is proper scoring rule (incentivizes honest uncertainty)
4. **Exploration:** Can sample from distribution during imagination

**Trade-off:**
- Pro: More principled, handles noise
- Con: Slightly more complex, requires variance estimation

### Decision 2: Why separate representation learner?

**Architecture:**
```
Encoder → Features → Representation Learner → States → [Dynamics, Reward, Continue, Policy, Value]
```

**Instead of:**
```
Encoder → States → [Everything else]
```

**Reasoning:**
1. **Modularity:** Can swap representation learner (e.g., VAE, contrastive, autoencoder)
2. **Explicit bottleneck:** Forces compression to learnable dimension
3. **Flexibility:** Representation learner can have its own loss (VAE KL, contrastive)

**Your friend's choice:** MLP Autoencoder
- Simple, deterministic
- No fancy regularization
- Just learns invertible compression

**Alternative:** Dreamer uses RSSM (Recurrent State Space Model)
- Stochastic latent states
- Temporal coherence
- More complex but potentially better

### Decision 3: Why three separate optimizers?

**Code** (line 135-143):
```python
world_model_optimizer = Adam([encoder, repr, dynamics, reward, continue, decoder], lr=1e-4)
actor_optimizer = Adam([policy], lr=3e-5)
critic_optimizer = Adam([value], lr=3e-5)
```

**Reasoning:**

**Different learning rates:**
- World model: 1e-4 (faster, less sensitive)
- Actor/Critic: 3e-5 (slower, more stable)

Why? World model is supervised (predict next state), while actor/critic is RL (noisy gradient estimates). RL typically needs lower learning rate.

**Different update frequencies:**
```python
world_model_updates_per_batch = 1  # Can be > 1
actor_updates_per_batch = 1
```

Could do multiple world model updates per actor update (like MBPO).

**Independent gradient flow:**
- World model updates don't affect actor (except through improved predictions)
- Actor updates don't affect world model
- Clean separation of concerns

**Trade-off:**
- Pro: Flexible, stable, interpretable
- Con: More complex code, more hyperparameters

### Decision 4: Why λ-returns instead of Monte Carlo?

**λ-returns** (line 1100-1118):
```python
return_t = r_t + γ * continues_t * ((1-λ) * V(s_t+1) + λ * return_t+1)
```

**Alternatives:**
1. **Monte Carlo:** return_t = r_t + r_t+1 + r_t+2 + ... (pure trajectory return)
2. **TD(0):** return_t = r_t + γ * V(s_t+1) (one-step bootstrap)
3. **TD(λ):** Blend of both

**Reasoning:**

**Monte Carlo problems:**
- High variance (trajectory rewards are noisy)
- Credit assignment unclear (which action caused the reward?)

**TD(0) problems:**
- Biased by value function (if V is wrong, returns are wrong)
- Short-sighted (only looks one step ahead)

**TD(λ) benefits:**
- λ=0: Pure bootstrapping (like TD(0))
- λ=1: Pure Monte Carlo
- λ=0.95 (typical): Mostly bootstrap, some trajectory

**Why λ=0.95 works:**
- First few steps: Use actual rewards (low bias)
- Later steps: Bootstrap with V (lower variance)
- Best of both worlds

### Decision 5: Why imagination rollout instead of real data?

**Your friend's approach** (line 852-857):
```python
imagined = rollout_imagination(states_from_real_data, length=15, with_grad=True)
actor_loss = -mean(imagined_returns)
```

**Alternative (model-free):**
```python
real_trajectory = collect_from_environment()
actor_loss = -mean(real_returns)
```

**Reasoning:**

**Sample efficiency:**
- Collect 100 real transitions
- Generate 1000 imagined transitions from those 100
- 10x more data for policy learning

**Gradient quality:**
- Real environment: Can't backprop through it (no gradient)
- World model: Differentiable, clean gradients
- Can do true end-to-end gradient from reward to action

**Trade-off:**
- Pro: Sample efficient, differentiable
- Con: Biased by world model errors (if dynamics wrong, policy learns wrong behavior)

This is why **world model warmup** (line 986-988) is crucial:
```python
if steps < warmup:
    actor_updates = 0  # Don't train policy on bad world model
```

### Decision 6: Why entropy bonus?

**Code** (line 942-950):
```python
entropy = policy_dist.entropy().mean()
actor_loss = -returns - entropy_coef * entropy
```

**Reasoning:**

**Without entropy bonus:**
- Policy can collapse to deterministic (entropy = 0)
- Stops exploring
- Gets stuck in local optima

**With entropy bonus:**
- Small penalty for being deterministic
- Encourages diversity in actions
- Maintains exploration throughout training

**Typical value:** entropy_coef = 0.01
- Small enough not to hurt performance
- Large enough to maintain some randomness

**Adaptive entropy:** Some algorithms (SAC) learn entropy_coef automatically.

---

## Signal Analysis: Is Each Component Learning Correctly?

### Encoder Signal Quality

**Gradient sources:**
1. Decoder reconstruction (direct)
2. Dynamics loss (through states)
3. Reward loss (through states)
4. Actor loss (through states, dynamics, rewards)

**Check if signal is good:**
```python
# Monitor these metrics:
metrics['decoder_reconstruction_loss']  # Should decrease
metrics['dynamics_loss']  # Should decrease
metrics['reward_loss']  # Should decrease
```

If all decrease, encoder is learning useful features.

**Problem signs:**
- decoder_loss stuck high: Encoder not preserving information
- dynamics_loss stuck high after many steps: Features not predictable
- gradient_norm very small (<1e-6): Vanishing gradients

### Representation Learner Signal Quality

**Gradient sources:**
1. All world model losses (dynamics, reward, continue, decoder)
2. Actor loss (indirectly through imagination)

**Check if signal is good:**
- representation_dim should be a bottleneck (e.g., 64)
- If too small: dynamics_loss will be high (not enough capacity)
- If too large: No compression, defeats the purpose

**Problem signs:**
- States become NaN: Gradient explosion, add gradient clipping or lower LR
- All states similar (low std): Representation collapsed, check initialization

### Dynamics Model Signal Quality

**Gradient source:**
1. dynamics_loss (direct MSE)
2. multi_step_loss (if enabled)
3. Actor loss (indirect, through imagination)

**Check if signal is good:**
```python
metrics['dynamics_mse']  # Should be small (< 0.1 for normalized states)
```

**Test quality:**
- Rollout predictions for 50 steps
- Compare to ground truth
- If diverges after 5 steps: Need better dynamics model

**Problem signs:**
- Predictions always return to mean state: Underfit, increase capacity
- Wild predictions: Overfit to noise, add regularization

### Reward Predictor Signal Quality

**Gradient source:**
1. reward_loss (NLL loss)
2. Actor loss (indirect, actor wants accurate reward predictions)

**Check if signal is good:**
```python
metrics['reward_mae']  # Mean absolute error should be small
metrics['reward_nll']  # Negative log likelihood should decrease
```

**Compare:**
- Predicted mean rewards vs actual rewards
- Should be highly correlated (correlation > 0.8)

**Problem signs:**
- High MAE but low NLL: Model predicting high uncertainty (large σ)
- Low MAE, high NLL: Model overconfident (σ too small)

### Continue Predictor Signal Quality

**Gradient source:**
1. continue_loss (BCE loss)

**Check if signal is good:**
```python
metrics['continue_accuracy']  # Should be > 80%
metrics['continue_precision']  # True positives / predicted positives
metrics['continue_recall']  # True positives / actual positives
```

**Class imbalance:**
- Most steps: Episode continues (class 1)
- Few steps: Episode ends (class 0)
- Predictor might just predict "always continue" for 99% accuracy

**Solution:** Monitor precision and recall for class 0 (termination)

**Problem signs:**
- Accuracy high but precision/recall for "terminate" low: Missing terminations
- Actor never expects episodes to end, doesn't prioritize short-term rewards

### Policy Head Signal Quality

**Gradient source:**
1. actor_loss (from imagined returns)
2. entropy bonus

**Check if signal is good:**
```python
metrics['mean_imagined_return']  # Should increase over training
metrics['entropy']  # Should slowly decrease (not immediately)
metrics['actor_grad_norm']  # Should be stable (0.1 - 10.0)
```

**Problem signs:**
- Returns not increasing: Actor not learning (check actor LR, world model quality)
- Entropy → 0 quickly: No exploration, increase entropy_coef
- Entropy stays max: Not learning, decrease entropy_coef or increase actor LR
- Gradient norm exploding (> 100): Add gradient clipping

### Value Function Signal Quality

**Gradient source:**
1. critic_loss (MSE against returns)

**Check if signal is good:**
```python
metrics['critic_loss']  # Should decrease
metrics['critic_td_error_mean']  # Should approach 0
metrics['critic_value_std']  # Should be > 0 (not collapsed)
```

**Test: Explained Variance**
```python
explained_var = 1 - var(V(s) - returns) / var(returns)
# Should be > 0.7 for good critic
```

**Problem signs:**
- TD error large and not decreasing: Critic not learning (check critic LR)
- Value predictions constant: Collapsed to mean, increase LR or capacity
- Value predictions exploding: Gradient explosion, lower LR or add clipping

---

## Common Issues and Debugging

### Issue 1: World Model Not Learning

**Symptoms:**
- dynamics_loss stuck high
- reward_loss stuck high
- imagined trajectories diverge immediately

**Debugging:**
1. Check gradient flow: `metrics['world_model_grad_norm']`
2. Verify data normalization: States should be roughly N(0, 1)
3. Check learning rate: 1e-4 usually works, try 1e-3 if too slow
4. Inspect predictions: Print predicted vs actual next states

### Issue 2: Policy Not Improving

**Symptoms:**
- mean_imagined_return not increasing
- evaluation performance flat

**Debugging:**
1. Check world model first: Is dynamics_loss < 0.1?
2. Check warmup: Are you skipping actor updates initially?
3. Verify gradient flow: `metrics['actor_grad_norm']` should be > 0
4. Check learning rate: 3e-5 typical, try 1e-4 if too slow
5. Monitor entropy: If 0, increase entropy_coef

### Issue 3: Training Unstable

**Symptoms:**
- Losses oscillating wildly
- NaN values appearing
- Performance deteriorating mid-training

**Debugging:**
1. Add gradient clipping if not present
2. Lower learning rates by 10x
3. Check for reward scaling: Rewards should be roughly [-10, 10]
4. Add warmup: Train world model alone first

---

## Summary: How Everything Learns Together

1. **World Model Phase:**
   - Real data → Encoder → States
   - States → Dynamics, Reward, Continue predictors
   - Train to predict next state, reward, termination
   - Train decoder to reconstruct observations

2. **Policy Phase:**
   - Take states from real data
   - Imagine rollout: Policy → Actions → Dynamics → Next States → Rewards
   - Compute returns from imagined trajectory
   - Update policy to maximize returns
   - Update value function to predict returns

3. **Gradient Flow:**
   - World model: Direct supervision (predict known quantities)
   - Actor: Indirect gradient through imagination (differentiable world model)
   - Critic: Supervised by computed returns

4. **Key Insight:**
   - World model learns from real data (supervised)
   - Policy learns from imagined data (RL in imagination)
   - Critic learns from both (bridges real and imagined)

This is a **sample-efficient** approach because:
- 1 real transition → N imagined transitions
- Can train policy much more than world model
- Amortizes cost of real environment interaction

Your friend built a solid implementation with good design choices!
