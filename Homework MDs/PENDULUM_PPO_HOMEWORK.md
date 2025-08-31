# Pendulum-v1 PPO Implementation Homework

**Environment Goal**: Master PPO on continuous control with proper action scaling, value normalization, and reward processing. This is your first step into continuous action spaces!

**Why Pendulum?** This environment teaches you:
- Continuous action spaces (vs discrete like CartPole/Acrobot)
- Action scaling and clipping strategies
- Reward normalization for training stability  
- Value function scale management
- Exploration variance in continuous spaces

## Phase A.2 - Continuous Control Fundamentals

### 📋 Environment Analysis Tasks

#### 1. Environment Investigation
- [ ] Install and run Pendulum-v1 for 1000 steps with random policy
- [ ] Record observation space: `Box(-high, high, shape=(3,))` - [cos(θ), sin(θ), θ̇]
- [ ] Record action space: `Box(-2, 2, shape=(1,))` - continuous torque in [-2, 2]
- [ ] Measure random policy baseline: should get ~-1200 to -1600 average return
- [ ] Document reward structure: `-(θ² + 0.1*θ̇² + 0.001*action²)`

#### 2. Continuous Action Analysis
- [ ] Plot action distributions from random policy
- [ ] Test action clipping: verify actions stay in [-2, 2] range
- [ ] Measure how often random policy hits action bounds
- [ ] Analyze reward sensitivity to action magnitude

#### 3. Observation Space Understanding  
- [ ] Plot cos(θ), sin(θ), θ̇ trajectories over episodes
- [ ] Verify observations stay in expected ranges
- [ ] Check for any observation scaling issues

### Expected Results 📊
- **Random Policy**: -1400 ± 200 average return over 100 episodes
- **Action Range**: All actions should be in [-2, 2]
- **Episode Length**: 200 steps (fixed episode length)
- **Goal**: Get pendulum upright (θ=0) and keep it stable

## Phase A.3 - Continuous PPO Implementation

### 📋 PPO Modifications for Continuous Control

Your roadmap specifies these additions for Pendulum:

#### 1. Action Space Modifications
- [ ] Replace Categorical distribution with Normal distribution
- [ ] Implement action scaling wrapper to map network outputs to [-2, 2]
- [ ] Add action clipping to ensure bounds are respected
- [ ] Handle action std (either fixed or learned)

#### 2. Network Architecture Changes
- [ ] Actor network outputs mean (and optionally std) for continuous actions
- [ ] Output layer should NOT have activation (raw outputs)
- [ ] Consider using tanh activation then scale to action bounds
- [ ] Critic remains the same (single value output)

#### 3. Reward & Value Scaling
- [ ] Add reward normalization wrapper (running mean/std)
- [ ] Implement value loss clipping for stability
- [ ] Consider reward scaling (divide by 100 or normalize)

### Implementation Requirements 🔧

#### Modified Action Sampling
```python
def act(self, observation, deterministic=False):
    # Get action mean from actor network
    action_mean = self.networks['actor'](observation)
    
    # Create Normal distribution (fixed std for now)
    action_std = torch.ones_like(action_mean) * 0.5  # or learn this
    dist = Normal(action_mean, action_std)
    
    if deterministic:
        action = action_mean
    else:
        action = dist.sample()
    
    # Scale/clip to action bounds [-2, 2]
    action = torch.clamp(action, -2.0, 2.0)
    return action
```

#### Action Evaluation
```python
def evaluate_actions(self, observations, actions):
    action_mean = self.networks['actor'](observations)
    action_std = torch.ones_like(action_mean) * 0.5
    
    dist = Normal(action_mean, action_std)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    
    values = self.networks['critic'](observations)
    return log_probs, values.squeeze(-1), entropy
```

### Configuration Template
```yaml
# configs/experiments/ppo_pendulum.yaml
experiment:
  name: "ppo_pendulum_continuous"
  seed: 42
  device: "cuda"  # or "mps" for Apple Silicon
  
algorithm:
  name: "ppo"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01          # May need higher for exploration
  ppo_epochs: 10
  minibatch_size: 64
  normalize_advantages: true
  clip_value_loss: true       # Important for continuous control!
  
environment:
  name: "Pendulum-v1"
  wrapper: "gym"
  normalize_obs: false        # Pendulum obs already normalized
  normalize_reward: true      # Key addition for Pendulum!
  
network:
  actor:
    type: "mlp"
    hidden_dims: [64, 64]
    activation: "tanh"
    output_activation: null   # No activation on output for continuous
  critic:
    type: "mlp"  
    hidden_dims: [64, 64]
    activation: "tanh"
    
buffer:
  type: "trajectory"
  size: 2048
  
training:
  total_timesteps: 100000
  eval_frequency: 5000
  checkpoint_frequency: 10000
  num_eval_episodes: 10
```

## Phase A.4 - Environment Wrappers

### 📋 Wrapper Implementation Tasks

#### 1. Action Scaling Wrapper
- [ ] Create wrapper to handle action bound mapping
- [ ] Test different scaling strategies (tanh, clamp, etc.)
- [ ] Verify actions stay within [-2, 2] bounds

#### 2. Reward Normalization Wrapper  
- [ ] Implement running mean/std for reward normalization
- [ ] Track reward statistics across episodes
- [ ] Apply normalization: `(reward - mean) / (std + epsilon)`

#### 3. Optional Wrappers
- [ ] Action noise wrapper for additional exploration
- [ ] Observation normalization (if needed)
- [ ] Episode logging wrapper

### Wrapper Code Templates
```python
# Action scaling wrapper
class ActionScaleWrapper(gym.ActionWrapper):
    def __init__(self, env, low=-2.0, high=2.0):
        super().__init__(env)
        self.low = low
        self.high = high
        
    def action(self, action):
        # Map from [-1, 1] to [low, high]
        return self.low + (action + 1.0) * (self.high - self.low) / 2.0

# Reward normalization wrapper  
class RewardNormWrapper(gym.RewardWrapper):
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.count = 0
        
    def reward(self, reward):
        self.count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.count
        self.reward_std = np.sqrt(((self.count - 1) * self.reward_std**2 + delta**2) / self.count)
        
        return (reward - self.reward_mean) / (self.reward_std + self.epsilon)
```

## Phase A.5 - Training & Hyperparameter Tuning

### 📋 Training Tasks

#### 1. Baseline Training
- [ ] Run initial training with standard hyperparameters
- [ ] Verify continuous action sampling works correctly
- [ ] Check that reward normalization is functioning
- [ ] Monitor exploration variance (action entropy)

#### 2. Exploration Tuning
- [ ] Start with fixed action_std = 0.5
- [ ] Try different std values: [0.1, 0.3, 0.5, 1.0]
- [ ] Test learned vs fixed standard deviation
- [ ] Monitor entropy decay over training

#### 3. Value Function Tuning
- [ ] Enable value loss clipping 
- [ ] Try different value_coef: [0.25, 0.5, 1.0]
- [ ] Monitor value function learning progress

### Expected Learning Progression 📈
- **0-10k steps**: Random exploration, high variance
- **10k-30k steps**: Learning action preferences, reducing std
- **30k-60k steps**: Consistent improvement toward goal
- **60k+ steps**: Stable policy, pendulum stays near upright

## Phase A.6 - Success Criteria & Validation

### 🎯 Done-When Gates (From Your Roadmap)

**Primary Goal**: Avg episode reward ≥ -200 with 3 seeds; stable entropy decay; exploration variance ≥0.8

#### 1. Performance Thresholds
- [ ] **Target Performance**: Average episode reward ≥ -200 (vs -1400 random)
- [ ] **Multi-seed Consistency**: 3 different seeds all achieve ≥ -200
- [ ] **Sample Efficiency**: Achieve this within 100k timesteps

#### 2. Exploration Requirements  
- [ ] **Stable Entropy Decay**: Action entropy decreases but doesn't collapse to zero
- [ ] **Exploration Variance**: Maintain ≥0.8 action variance during training
- [ ] **Action Distribution**: Actions cover reasonable range, not stuck at bounds

#### 3. Additional Validation
- [ ] **Policy Stability**: Low variance in final policy performance
- [ ] **Value Function**: Critic loss decreases consistently
- [ ] **Action Bounds**: All actions respect [-2, 2] constraints

### Performance Benchmarks 📊
| Metric | Random Policy | Target Performance |
|--------|---------------|-------------------|
| Average Return | -1400 ± 200 | ≥-200 |
| Best Return | ~-800 | ≥-150 |
| Action Entropy | High (~1.0) | Stable decay to ~0.5 |
| Success Rate | 0% | Pendulum stable >80% time |

## Phase A.7 - Advanced Continuous Control Features

### 📋 Enhancement Tasks

#### 1. Learned Action Standard Deviation
- [ ] Modify actor to output both mean and log_std
- [ ] Implement std parameter learning
- [ ] Add std constraints (min/max bounds)
- [ ] Compare with fixed std performance

#### 2. Action Regularization
- [ ] Add L2 penalty on actions to reduce magnitude
- [ ] Implement action smoothness penalty (consecutive action diff)
- [ ] Test different regularization coefficients

#### 3. Improved Value Function
- [ ] Try different value loss functions (Huber loss)
- [ ] Implement value function regularization
- [ ] Test shared vs separate network trunks

### Code Templates for Enhancements
```python
# Learned std actor network
class ContinuousActorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        # ... build network layers ...
        self.mean_head = nn.Linear(hidden_dims[-1], output_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        # ... forward through layers ...
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent too small/large std
        return mean, log_std.exp()

# Action regularization in loss
def compute_ppo_loss_with_regularization(self, batch):
    # ... standard PPO loss computation ...
    
    # Add action regularization
    action_penalty = torch.mean(batch['actions']**2) * 0.01
    total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + action_penalty
    
    return policy_loss, value_loss, entropy_loss, action_penalty
```

## Phase A.8 - Debugging Guide

### Common Continuous Control Issues 🔧

#### Problem: Actions Saturate at Bounds
**Symptoms**: All actions become +2 or -2
**Solutions**:
- [ ] Reduce action std initialization
- [ ] Add action regularization penalty  
- [ ] Check reward normalization is working
- [ ] Verify network output activation (should be None for continuous)

#### Problem: No Learning / High Variance
**Symptoms**: Return stays around random level with high variance
**Solutions**:
- [ ] Enable reward normalization
- [ ] Try larger batch sizes (4096)
- [ ] Increase value_coef to 1.0
- [ ] Check advantage computation for continuous rewards

#### Problem: Learning Then Forgetting
**Symptoms**: Improves then performance degrades
**Solutions**:
- [ ] Reduce learning rate to 1e-4
- [ ] Enable value loss clipping
- [ ] Reduce clip_ratio to 0.1
- [ ] Add KL divergence penalty

#### Problem: Action Std Collapses
**Symptoms**: Entropy goes to zero, no exploration
**Solutions**:
- [ ] Increase entropy_coef to 0.05 or 0.1
- [ ] Use fixed std instead of learned
- [ ] Add minimum std constraint in learned case

### Mac-Specific Optimizations 🍎
- [ ] Use MPS backend for Apple Silicon: `device="mps"`
- [ ] Reduce batch size if memory limited: `size=1024`
- [ ] Use float32 for faster computation: `torch.set_default_dtype(torch.float32)`

## Phase A.9 - Analysis & Next Steps

### 📋 Analysis Tasks

#### 1. Performance Analysis
- [ ] Plot learning curves comparing seeds
- [ ] Analyze action distributions over training
- [ ] Visualize pendulum trajectories (angle over time)
- [ ] Create policy rollout videos

#### 2. Hyperparameter Sensitivity
- [ ] Document which hyperparameters were most important
- [ ] Record best configuration for continuous control
- [ ] Compare fixed vs learned action std results

#### 3. Comparison with Baselines
- [ ] Compare your PPO with published Pendulum results
- [ ] Benchmark sample efficiency vs other algorithms
- [ ] Document unique insights from continuous control

### Expected Insights 🧠
- **Reward normalization** is crucial for continuous control stability
- **Value loss clipping** prevents destabilizing value function updates
- **Action bounds** must be handled carefully to maintain exploration
- **Entropy scheduling** balances exploration vs exploitation over training

## Quick Start Commands 🚀

### 1. Environment Test
```bash
# Verify Pendulum works with continuous actions
python -c "import gymnasium as gym; env = gym.make('Pendulum-v1'); obs = env.reset(); print(f'Obs: {obs}'); action = env.action_space.sample(); print(f'Action: {action}'); env.step(action)"
```

### 2. Quick Training Test  
```bash
# Test continuous PPO doesn't crash
python scripts/train.py --config configs/experiments/ppo_pendulum.yaml --debug --total-timesteps 1000
```

### 3. Full Training
```bash
# Full continuous control training
python scripts/train.py --config configs/experiments/ppo_pendulum.yaml
```

### 4. Multi-seed Validation
```bash
# Test consistency across seeds
for seed in 1 2 3; do
  python scripts/train.py --config configs/experiments/ppo_pendulum.yaml --seed $seed --name pendulum_seed_$seed
done
```

## Success Checklist ✅

Your Pendulum PPO is complete when:
- [ ] Consistently achieves ≥-200 average return across multiple seeds
- [ ] Actions properly respect continuous bounds [-2, 2]
- [ ] Entropy decays stably without collapsing to zero
- [ ] Value function learns to predict returns accurately
- [ ] Ready for partial observability challenges (MiniGrid next!)

## What You've Learned 🎓

By completing this homework, you've mastered:
- **Continuous Action Spaces**: Normal distributions vs Categorical
- **Action Scaling**: Mapping network outputs to environment bounds
- **Reward Normalization**: Stabilizing training with proper scaling
- **Value Function Clipping**: Preventing destabilizing value updates
- **Exploration Management**: Balancing entropy in continuous spaces

**Next Up**: MiniGrid DoorKey - Partial observability and exploration with visual observations!