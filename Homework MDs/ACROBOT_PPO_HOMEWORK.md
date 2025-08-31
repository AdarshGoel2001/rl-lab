# Acrobot-v1 PPO Implementation Homework

**Environment Goal**: Master PPO on sparse, unstable rewards around goal regions. Unlike CartPole's dense rewards, Acrobot teaches you to handle delayed gratification and advantage estimation challenges.

**Why Acrobot?** This environment stresses your PPO implementation with:
- Sparse rewards (-1 until solved, then 0)  
- Unstable dynamics around the goal
- Need for proper GAE λ tuning
- Entropy management for exploration

## Phase A.1 - Environment Setup & Understanding

### 📋 Environment Analysis Tasks

#### 1. Environment Investigation
- [ ] Install and run basic Acrobot-v1 for 1000 steps with random policy
- [ ] Record observation space: `Box(-1, 1, shape=(6,))` - 6D state space
- [ ] Record action space: `Discrete(3)` - 3 discrete actions (left, none, right)
- [ ] Measure random policy baseline: should get ~-500 average return
- [ ] Document reward structure: -1 per step until goal (-cos(θ₁) - cos(θ₁+θ₂) > 1)

#### 2. Reward Analysis  
- [ ] Plot 10 episodes of random policy rewards to see sparsity
- [ ] Time average episode length: should be ~500 steps (max episode length)
- [ ] Identify when agent gets lucky and reaches goal early
- [ ] Document goal condition: both pendulum segments pointing upward

### Expected Results 📊
- **Random Policy**: -500 ± 50 average return over 100 episodes
- **Episode Length**: ~500 steps (hits time limit)
- **Success Rate**: <1% for random policy

## Phase A.2 - PPO Configuration & Training

### 📋 Hyperparameter Tuning Tasks

Your roadmap suggests these specific config nudges for Acrobot:

#### 1. Core PPO Settings
- [ ] Set GAE λ=0.95 (crucial for credit assignment)
- [ ] Set γ=0.99 (standard discount)
- [ ] Set clip_ratio=0.2 (standard PPO clipping)
- [ ] Set entropy coefficient ≈0.01 (encourage exploration)
- [ ] Set PPO epochs=10 (more updates per batch)
- [ ] Set minibatch_size=64
- [ ] Set batch_size=2048-8192 (test both ranges)

#### 2. Network Architecture
- [ ] Use the same MLP as CartPole: ActorMLP and CriticMLP
- [ ] Input dim: 6 (Acrobot observation space)
- [ ] Actor output: 3 (discrete actions)
- [ ] Critic output: 1 (value estimate)

#### 3. Learning Rate Tuning
- [ ] Start with lr=3e-4 (PPO default)
- [ ] Try lr=1e-4 if learning is unstable
- [ ] Try lr=5e-4 if learning is too slow

### Configuration Template
```yaml
# configs/experiments/ppo_acrobot.yaml
experiment:
  name: "ppo_acrobot_sparse_rewards"
  seed: 42
  device: "cuda"  # or "mps" for Apple Silicon
  
algorithm:
  name: "ppo"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95          # Key for sparse rewards!
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01        # Exploration crucial
  ppo_epochs: 10            # More updates per batch
  minibatch_size: 64
  normalize_advantages: true
  
environment:
  name: "Acrobot-v1"
  wrapper: "gym"
  
network:
  actor:
    type: "mlp"
    hidden_dims: [64, 64]
    activation: "tanh"
  critic:
    type: "mlp"  
    hidden_dims: [64, 64]
    activation: "tanh"
    
buffer:
  type: "trajectory"
  size: 2048               # Try 4096, 8192 too
  
training:
  total_timesteps: 100000  # More steps needed than CartPole
  eval_frequency: 5000
  checkpoint_frequency: 10000
  num_eval_episodes: 10
```

## Phase A.3 - Implementation & Training

### 📋 Training Tasks

#### 1. Initial Training Run
- [ ] Run 10k timesteps - should show some learning signals
- [ ] Check that losses are finite (no NaN/Inf)
- [ ] Verify advantage estimates are reasonable (not all zero)
- [ ] Confirm entropy decreases over time (but not to zero)

#### 2. Debugging Common Issues
- [ ] **No Learning**: Check if advantages are all zero → GAE λ too high/low
- [ ] **Unstable Learning**: Reduce lr to 1e-4, check clip_ratio
- [ ] **Too Slow**: Increase batch_size to 4096/8192
- [ ] **Poor Exploration**: Increase entropy_coef to 0.02-0.05

#### 3. Hyperparameter Experiments
- [ ] Test batch_size: [2048, 4096, 8192] - larger often better for sparse rewards
- [ ] Test GAE λ: [0.90, 0.95, 0.97] - find best credit assignment
- [ ] Test entropy_coef: [0.005, 0.01, 0.02] - balance exploration/exploitation

### Expected Learning Curve 📈
- **0-20k steps**: Random policy performance (-500)
- **20k-40k steps**: Slow improvement, occasional good episodes  
- **40k-60k steps**: Consistent improvement, reaching goal more often
- **60k+ steps**: Stable policy, consistent success

## Phase A.4 - Success Criteria & Validation

### 🎯 Done-When Gates (From Your Roadmap)

**Primary Goal**: Median return consistently beats baseline (≥-100) across 3 seeds; variance <20%

#### 1. Performance Thresholds
- [ ] **Baseline Beat**: Median return ≥ -100 (much better than -500 random)
- [ ] **Consistency**: 3 different seeds all achieve ≥-100 median
- [ ] **Stability**: Variance in final performance <20% across seeds
- [ ] **Sample Efficiency**: Achieve this within 100k timesteps

#### 2. Additional Validation
- [ ] **Success Rate**: >50% episodes reach the goal state
- [ ] **Episode Length**: Average episode length <300 steps (faster than random)
- [ ] **Value Function**: Critic accurately predicts returns (low value loss)

#### 3. Robustness Tests
- [ ] Test with 5 different random seeds
- [ ] Confirm learning curve is smooth (not just lucky runs)
- [ ] Verify policy actually learned the task (not just random luck)

### Performance Benchmarks 📊
| Metric | Random Policy | Target Performance |
|--------|---------------|-------------------|
| Average Return | -500 ± 50 | ≥-100 ± 20 |
| Success Rate | <1% | >50% |
| Episode Length | ~500 steps | <300 steps |
| Convergence | Never | <100k timesteps |

## Phase A.5 - Analysis & Understanding

### 📋 Analysis Tasks

#### 1. Learning Analysis
- [ ] Plot learning curves for all 3 seeds
- [ ] Analyze value function learning (value_loss decreasing)
- [ ] Check policy entropy decay (not too fast, not too slow)
- [ ] Measure advantage estimation quality

#### 2. Policy Visualization
- [ ] Record videos of trained policy
- [ ] Compare early vs late training behavior
- [ ] Analyze action distribution changes over training
- [ ] Verify goal-reaching strategy is consistent

#### 3. Hyperparameter Sensitivity
- [ ] Document which hyperparameters mattered most
- [ ] Record best configuration for future reference
- [ ] Note Mac-specific optimizations used

### Expected Insights 🧠
- **GAE λ** is crucial for credit assignment in sparse reward tasks
- **Batch size** often needs to be larger than dense reward environments  
- **Entropy scheduling** helps balance exploration vs exploitation
- **Value function** learning is harder with sparse rewards

## Phase A.6 - Debugging Guide

### Common Issues & Solutions 🔧

#### Problem: No Learning at All
**Symptoms**: Loss stays constant, return doesn't improve
**Solutions**:
- [ ] Check GAE λ - try 0.90 if 0.95 doesn't work
- [ ] Increase batch_size to 8192
- [ ] Verify advantage computation is working
- [ ] Check that rewards are being recorded correctly

#### Problem: Learning but Plateaus Early  
**Symptoms**: Improves to -300 then stops
**Solutions**:
- [ ] Increase entropy_coef to encourage more exploration
- [ ] Try different network sizes [128, 128] or [256, 256]
- [ ] Extend training time (sparse rewards take longer)

#### Problem: Unstable Learning
**Symptoms**: Return bounces up and down wildly
**Solutions**:
- [ ] Reduce learning rate to 1e-4
- [ ] Reduce clip_ratio to 0.1
- [ ] Add gradient norm clipping (max_grad_norm=0.5)

#### Problem: Fast Convergence to Bad Policy
**Symptoms**: Quickly reaches -200 but can't improve further
**Solutions**:
- [ ] This might actually be good! Check if -200 beats your threshold
- [ ] If not, increase exploration with higher entropy_coef

## Quick Start Commands 🚀

### 1. Test Setup
```bash
# Verify Acrobot works on your system
python -c "import gymnasium as gym; env = gym.make('Acrobot-v1'); print(env.reset()); print(env.step(env.action_space.sample()))"
```

### 2. Quick Functionality Test
```bash
# Test PPO doesn't crash on Acrobot
python scripts/train.py --config configs/experiments/ppo_acrobot.yaml --debug --total-timesteps 1000
```

### 3. Full Training Run
```bash  
# Train for full duration
python scripts/train.py --config configs/experiments/ppo_acrobot.yaml
```

### 4. Multi-seed Validation
```bash
# Test 3 seeds for consistency
for seed in 1 2 3; do
  python scripts/train.py --config configs/experiments/ppo_acrobot.yaml --seed $seed --name acrobot_seed_$seed
done
```

## Success Checklist ✅

Your Acrobot PPO is complete when:
- [ ] Trained PPO beats random policy by huge margin (≥-100 vs -500)
- [ ] Consistent across multiple seeds (variance <20%)  
- [ ] Learning is stable and reproducible
- [ ] Understands sparse reward credit assignment
- [ ] Ready to tackle continuous control (Pendulum next!)

## What You've Learned 🎓

By completing this homework, you've mastered:
- **Sparse Reward RL**: GAE λ tuning for delayed gratification
- **Credit Assignment**: Handling rewards far from actions
- **Exploration Balance**: Entropy scheduling for discovery vs exploitation  
- **Robustness**: Multi-seed validation and consistency metrics
- **Mac Optimization**: Memory and compute management for longer training

**Next Up**: Pendulum-v1 - Continuous control with action scaling and reward normalization!