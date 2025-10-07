# World Model Implementation Gaps Analysis

## Executive Summary

Your architecture is **fundamentally sound** for implementing all major world models. The missing pieces are **specific components**, not architectural changes. Below is what needs to be added for each world model.

---

## Current System Strengths ✅

Your system already has:
1. ✅ Modular component system with registries
2. ✅ Separate optimizers (WM, actor, critic) with different learning rates
3. ✅ Distributional predictions (Gaussian dynamics/rewards)
4. ✅ Lambda returns for credit assignment
5. ✅ Imagination-based policy learning
6. ✅ Warmup schedules
7. ✅ Flexible sequence processing (flat/sequence modes)
8. ✅ Optional decoder for reconstruction
9. ✅ Context passing to components
10. ✅ Multiple updates per batch configuration
11. ✅ Optional planner interface

---

## What's Missing: Component-Level Gaps

### 1. PlaNet (Pure Planning)

#### Missing Components:
```python
# Need to implement:
1. src/components/representation_learners/rssm_gaussian.py
   - Continuous stochastic + deterministic states
   - Prior: p(s_t | s_{t-1}, a_{t-1})
   - Posterior: q(s_t | s_{t-1}, a_{t-1}, o_t)

2. src/components/planners/cem_planner.py
   - Cross-Entropy Method
   - Sample action sequences
   - Evaluate in imagination
   - Refit distribution to top-K
```

#### Configuration Example:
```yaml
algorithm:
  implementation: world_model_planet
  world_model_lr: 6e-4
  actor_lr: 0  # No actor!
  critic_lr: 0  # No critic!
  actor_updates_per_batch: 0
  critic_updates_per_batch: 0
  imagination_horizon: 12

components:
  representation_learner:
    type: rssm_gaussian
    config:
      stochastic_size: 30
      deterministic_size: 200
      min_std: 0.1

  planner:
    type: cem
    config:
      horizon: 12
      num_candidates: 1000
      num_elite: 100
      num_iterations: 10
```

**What Works Already:**
- ✅ Optional planner interface
- ✅ Can disable actor/critic updates
- ✅ Imagination rollout infrastructure

---

### 2. Dreamer v1 & v2

#### Missing Components:
```python
# Dreamer v1 (Gaussian RSSM):
1. src/components/representation_learners/rssm_gaussian.py
   - Same as PlaNet but with different hyperparams

# Dreamer v2 (Discrete RSSM):
2. src/components/representation_learners/rssm_discrete.py
   - 32 categorical distributions with 32 classes each
   - Straight-through gradients
   - KL balancing

3. src/paradigms/world_model/dreamer.py
   - KL balancing strategies
   - Free bits implementation
```

#### Key Difference from Your MVP:
Your system uses:
```python
# Current: Single representation from features
features = encoder(obs)
states = repr_learner.encode(features)
```

Dreamer needs:
```python
# Dreamer: RSSM with recurrence
features = encoder(obs)

# Posterior (with observation)
h_t = GRU(concat(h_{t-1}, s_{t-1}, a_{t-1}))
s_t ~ posterior(h_t, features_t)

# Prior (without observation)
h_t = GRU(concat(h_{t-1}, s_{t-1}, a_{t-1}))
s_t ~ prior(h_t)

# KL loss between posterior and prior
kl_loss = kl_divergence(posterior, prior)
```

#### Implementation Path:
```python
class RSSMRepresentationLearner(BaseRepresentationLearner):
    def __init__(self, config):
        self.gru = nn.GRUCell(stochastic_size + action_dim, deterministic_size)
        self.prior_net = MLP(deterministic_size, stochastic_params)
        self.posterior_net = MLP(deterministic_size + feature_dim, stochastic_params)

    def encode(self, features, sequence_context=None):
        """Use posterior when observations available"""
        # Extract prev states/actions from context
        prev_h = sequence_context.get('prev_deterministic')
        prev_s = sequence_context.get('prev_stochastic')
        actions = sequence_context.get('actions')

        # Deterministic path
        h = self.gru(concat(prev_s, actions), prev_h)

        # Stochastic posterior
        posterior_params = self.posterior_net(concat(h, features))
        stochastic = self._sample(posterior_params)

        return concat(h, stochastic)

    def representation_loss(self, features, sequence_context=None):
        # Compute both prior and posterior
        posterior_dist = self._posterior(...)
        prior_dist = self._prior(...)

        kl_loss = kl_divergence(posterior_dist, prior_dist)

        # Apply KL balancing (Dreamer v2)
        kl_loss = self._apply_kl_strategy(kl_loss)

        return {'representation_loss': kl_loss.mean()}
```

#### Configuration Example:
```yaml
# Dreamer v2
components:
  representation_learner:
    type: rssm_discrete
    config:
      stochastic_size: 32
      num_classes: 32
      deterministic_size: 512
      hidden_size: 512
      kl_strategy: free_bits
      free_bits: 1.0

algorithm:
  sequence_processing:
    mode: always
    component_inputs:
      representation_learner: sequence
    provide_padding_mask: true
```

**What Works Already:**
- ✅ Sequence processing infrastructure
- ✅ Context passing
- ✅ Lambda returns
- ✅ Imagination rollouts

---

### 3. Dreamer v3

#### Additional Missing Components:
```python
1. Symlog transforms in paradigm.py
2. EMA critic
3. Critic target regularization
```

#### Changes Needed in Paradigm:
```python
# In BaseWorldModelParadigm.compute_loss():

# Add symlog transform
if self.config.get('use_symlog', False):
    # Transform predictions
    pred_rewards_symlog = self.symlog(imagined_rewards)
    returns_symlog = self.compute_lambda_returns(pred_rewards_symlog, ...)
    returns = self.symexp(returns_symlog)
else:
    returns = self.compute_lambda_returns(imagined_rewards, ...)

# Add EMA critic
if self.config.get('use_ema_critic', False):
    # Use target network for value estimation
    with torch.no_grad():
        values = self.value_function_target(states)
    # Update target with exponential moving average
    self._update_target_network()
```

#### Configuration:
```yaml
algorithm:
  use_symlog: true
  use_ema_critic: true
  critic_ema_decay: 0.98
  world_model_pretrain_steps: 10000
```

**What Works Already:**
- ✅ Most of the infrastructure
- ✅ Just need to add transforms & EMA

---

### 4. MuZero

#### Major Architectural Difference:
MuZero does **not** reconstruct observations. It uses **value equivalence**:

```python
# Your MVP:
obs → encoder → features → repr_learner → states
                                           ↓
                                        decoder → reconstructed_obs

# MuZero:
obs → representation_net(h) → hidden_state
                                ↓
                          No decoder!
                          Only predict: value, policy, reward
```

#### Missing Components:
```python
1. src/components/representation_learners/muzero_representation.py
   - No posterior/prior distinction
   - No KL loss
   - Just: h = f(observation)

2. src/components/dynamics/muzero_dynamics.py
   - Returns BOTH next_state AND immediate reward
   - g(h, a) → (h', r)

3. src/components/planners/mcts.py
   - Monte Carlo Tree Search
   - UCB selection
   - Value + policy targets from search

4. src/buffers/muzero_buffer.py (optional)
   - Reanalyze support
   - Store MCTS statistics
```

#### Key Interface Change:
```python
# Current dynamics interface:
class BaseDynamicsModel:
    def forward(self, state, action) -> Distribution:
        next_state_dist = ...
        return next_state_dist  # Only state

# MuZero needs:
class MuZeroDynamics(BaseDynamicsModel):
    def forward(self, state, action) -> Tuple[Distribution, torch.Tensor]:
        next_state_dist, reward = self.g_function(state, action)
        return next_state_dist, reward  # State AND reward
```

#### Configuration:
```yaml
paradigm: world_model
implementation: world_model_muzero

algorithm:
  use_reconstruction: false  # Critical!
  use_mcts_targets: true
  mcts_simulations: 50
  td_steps: 5

components:
  representation_learner:
    type: muzero_representation
    config:
      hidden_dim: 256
      num_blocks: 16

  dynamics_model:
    type: muzero_dynamics
    config:
      hidden_dim: 256
      num_blocks: 16
      returns_reward: true  # Flag for interface

  planner:
    type: mcts
    config:
      num_simulations: 50
      c_puct: 1.25
      dirichlet_alpha: 0.25
```

#### Paradigm Changes Needed:
```python
# In BaseWorldModelParadigm:

def compute_loss(self, batch):
    # ...

    # Check if reconstruction is disabled
    use_reconstruction = self.config.get('use_reconstruction', True)
    if not use_reconstruction:
        decoder_loss = torch.tensor(0.0)
    else:
        decoder_loss = ...

    # Check if dynamics returns reward
    if hasattr(self.dynamics_model, 'returns_reward') and self.dynamics_model.returns_reward:
        # Dynamics already predicts reward, don't need separate reward predictor
        next_state_dist, predicted_rewards = self.dynamics_model(states, actions)
        reward_loss = F.mse_loss(predicted_rewards, true_rewards)
    else:
        # Standard: separate reward predictor
        reward_loss = self.reward_predictor.reward_loss(...)
```

**What Works Already:**
- ✅ Optional planner interface
- ✅ Can disable decoder (set `decoder_loss_weight: 0`)
- ✅ Flexible component system

---

## Priority Implementation Order

### Phase 1: Enable PlaNet & Dreamer v1 (Week 1-2)
1. **RSSMGaussian representation learner** (highest priority)
   - Posterior/prior distributions
   - KL loss
   - GRU recurrence

2. **Sequence processing integration**
   - Ensure RSSM can access `prev_states`, `actions`
   - Test with sequence buffers

3. **CEM planner** (for PlaNet)

**Expected Outcome:** Can run PlaNet (pure planning) and Dreamer v1

---

### Phase 2: Dreamer v2 & v3 (Week 2-3)
1. **RSSMDiscrete representation learner**
   - 32×32 categorical distributions
   - Straight-through gradients

2. **KL balancing in paradigm**
   - Free bits
   - Balanced KL

3. **Symlog transforms** (v3)
   - Add to paradigm

4. **EMA critic** (v3)

**Expected Outcome:** Can run Dreamer v2 & v3

---

### Phase 3: MuZero (Week 3-4)
1. **MuZero representation learner**
   - No posterior/prior
   - Just encoding

2. **MuZero dynamics**
   - Returns (state, reward) tuple
   - Update interface

3. **MCTS planner**
   - UCB
   - Expansion/simulation/backprop

4. **Paradigm modifications**
   - Disable reconstruction
   - MCTS targets

**Expected Outcome:** Can run MuZero

---

## Minimal Viable Extensions

If you want to test one model quickly:

### Quickest: Dreamer v1 (1 week)
```python
# Just need:
1. RSSMGaussian (200 lines)
2. Small paradigm tweaks (50 lines)
3. Config example (30 lines)
```

### Medium: Dreamer v3 (2 weeks)
```python
# Dreamer v1 +
1. RSSMDiscrete (250 lines)
2. Symlog functions (20 lines)
3. EMA critic (30 lines)
```

### Complex: MuZero (3-4 weeks)
```python
# From scratch:
1. MuZeroRepresentation (300 lines)
2. MuZeroDynamics (400 lines)
3. MCTS planner (500 lines)
4. Paradigm modifications (100 lines)
```

---

## Critical Design Question: Dynamics Interface

**Current Interface:**
```python
class BaseDynamicsModel:
    def forward(self, state, action) -> Distribution
```

**Should we extend to:**
```python
class BaseDynamicsModel:
    def forward(self, state, action) -> Union[Distribution, Tuple[Distribution, Tensor]]

    @property
    def returns_reward(self) -> bool:
        return False  # Override in MuZero
```

**Recommendation:** Yes, extend the interface. It's backward compatible (existing models return Distribution, MuZero returns tuple).

---

## Summary: What You Need

### Components to Implement:
1. ✅ **RSSMGaussian** - 200 lines (PlaNet, Dreamer v1)
2. ✅ **RSSMDiscrete** - 250 lines (Dreamer v2, v3)
3. ✅ **MuZeroRepresentation** - 300 lines (MuZero)
4. ✅ **MuZeroDynamics** - 400 lines (MuZero)
5. ✅ **CEMPlanner** - 300 lines (PlaNet)
6. ✅ **MCTSPlanner** - 500 lines (MuZero)

### Paradigm Extensions:
1. ✅ **KL balancing** - 50 lines
2. ✅ **Symlog transforms** - 20 lines
3. ✅ **EMA critic** - 30 lines
4. ✅ **Optional reconstruction** - 20 lines (mostly done)
5. ✅ **Dynamics reward return** - 30 lines

### Total New Code: ~2000 lines
- Your architecture: **Already built** ✅
- What's missing: **Specific components**

---

## Architectural Recommendations

### 1. Keep Your Current Design
Don't change the paradigm structure. It's excellent.

### 2. Extend Interfaces Minimally
```python
# BaseDynamicsModel: Add optional reward return
# BaseRepresentationLearner: Already flexible enough
# BasePlanner: Already optional
```

### 3. Use Config Flags
```yaml
# Enable/disable features per world model
algorithm:
  use_reconstruction: true/false
  use_mcts_targets: true/false
  use_symlog: true/false
  use_ema_critic: true/false
```

### 4. Component Registries
```python
# Register all variants
REPRESENTATION_LEARNER_REGISTRY = {
    'identity': IdentityRepresentationLearner,
    'autoencoder': AutoencoderRepresentationLearner,
    'rssm_gaussian': RSSMGaussianRepresentationLearner,  # New
    'rssm_discrete': RSSMDiscreteRepresentationLearner,  # New
    'muzero': MuZeroRepresentationLearner,               # New
}

PLANNER_REGISTRY = {
    'cem': CEMPlanner,     # New
    'mcts': MCTSPlanner,   # New
}
```

---

## Next Steps

1. **Decide priority:** Which world model first? (Recommend: Dreamer v1)
2. **Implement RSSM:** Start with Gaussian variant
3. **Test with sequence processing:** Your infrastructure is ready
4. **Add KL balancing:** Small paradigm extension
5. **Validate:** Compare to published results

Your architecture is **production-ready**. You just need components! 🚀
