# PPO Implementation Homework Checklist

Welcome to your first PPO implementation! I've created the boilerplate code and configuration files. Your job is to fill in the TODOs to make PPO work on CartPole.

## Files Created for You ✅

- **`src/algorithms/ppo.py`** - PPO algorithm template with detailed TODOs
- **`configs/algorithms/ppo.yaml`** - PPO hyperparameter configuration  
- **`configs/experiments/ppo_cartpole.yaml`** - Complete experiment setup

## Your Implementation Tasks

### 📋 Core Implementation (Required)

#### 1. Network Setup (`_setup_networks_and_optimizers`)
- [ ] Create actor network (ActorMLP) with correct input/output dimensions
- [ ] Create critic network (CriticMLP) with single output (value function)
- [ ] Set up Adam optimizers for both networks
- [ ] Store networks in `self.networks` dict
- [ ] Store optimizers in `self.optimizers` dict

#### 2. Action Selection (`act`)
- [ ] Forward pass through actor network to get action logits
- [ ] Create appropriate distribution (Categorical for discrete actions)
- [ ] Handle deterministic vs stochastic action selection
- [ ] Return properly formatted action tensor

#### 3. Training Data Collection (`get_action_and_value`)
- [ ] Get action logits from actor network
- [ ] Get value estimate from critic network
- [ ] Sample action and compute log probability
- [ ] Return action, log_prob, and value

#### 4. Policy Evaluation (`evaluate_actions`)
- [ ] Forward pass through both networks
- [ ] Create distribution from current policy
- [ ] Compute log probabilities of given actions under current policy
- [ ] Compute entropy of current policy
- [ ] Return log_probs, values, entropy

#### 5. PPO Loss Computation (`compute_ppo_loss`)
- [ ] Extract batch data (observations, actions, old_log_probs, advantages, returns)
- [ ] Normalize advantages if requested
- [ ] Evaluate actions under current policy
- [ ] Compute probability ratio: `exp(new_log_prob - old_log_prob)`
- [ ] Implement clipped surrogate objective: `min(ratio * adv, clipped_ratio * adv)`
- [ ] Compute value function loss (MSE between predicted and target values)
- [ ] Implement value loss clipping (optional but recommended)
- [ ] Compute entropy bonus (negative entropy for maximization)
- [ ] Return policy_loss, value_loss, entropy_loss

#### 6. Training Loop (`update`)
- [ ] Move batch to correct device
- [ ] Set up multi-epoch training loop
- [ ] Split data into minibatches for each epoch
- [ ] For each minibatch:
  - [ ] Compute PPO losses
  - [ ] Combine losses with coefficients
  - [ ] Zero gradients
  - [ ] Backward pass
  - [ ] Apply gradient clipping
  - [ ] Update parameters
- [ ] Track and return training metrics
- [ ] Increment step counter

### 🧪 Testing & Validation

#### Basic Functionality Tests
- [ ] Test network creation (no errors on initialization)
- [ ] Test action sampling (actions in correct range)
- [ ] Test forward pass (no NaN/Inf values)
- [ ] Test loss computation (finite loss values)

#### Training Tests
- [ ] Run 1000 timesteps without crashes
- [ ] Verify losses are decreasing (at least sometimes)
- [ ] Check that networks are actually updating (gradient norms > 0)
- [ ] Confirm tensorboard logging works

#### Performance Tests  
- [ ] Train for 25k timesteps
- [ ] Achieve average reward > 50 (better than random)
- [ ] Achieve average reward > 195 (CartPole solved)
- [ ] Consistent performance (multiple runs succeed)

### 🚀 Advanced Extensions (Optional)

#### Improvements
- [ ] Add learning rate scheduling
- [ ] Implement adaptive clipping based on KL divergence
- [ ] Add early stopping based on KL threshold
- [ ] Implement different value loss types (Huber loss)
- [ ] Add network weight initialization options

#### Experiments
- [ ] Try different network architectures
- [ ] Experiment with hyperparameters
- [ ] Test on other environments (MountainCar, LunarLander)
- [ ] Compare with other algorithms

## How to Test Your Implementation

### 1. Quick Functionality Test
```bash
# Test basic functionality (should not crash)
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --debug --total_timesteps 1000
```

### 2. Learning Test  
```bash  
# Test learning (should improve over time)
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --total_timesteps 25000
```

### 3. Full Training
```bash
# Full training run
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml
```

## Expected Results

- **Initial Performance**: Random policy gets ~20 reward
- **Learning Onset**: Should start improving within 5-10k timesteps
- **Solved**: Should achieve 195+ average reward within 25-50k timesteps  
- **Final Performance**: Should consistently get 200 reward (max possible)

## Common Pitfalls & Debug Tips

### If Nothing Happens (No Learning)
- Check if gradients are flowing (`grad_norm` in logs should be > 0)
- Verify action sampling returns valid actions
- Make sure advantages are being computed correctly
- Check that the policy loss has the right sign (should be negative of objective)

### If Training is Unstable  
- Try lower learning rate (1e-4 instead of 3e-4)
- Try smaller clip_ratio (0.1 instead of 0.2)
- Check for NaN/Inf values in losses
- Make sure advantages are normalized

### If Learning is Too Slow
- Try higher learning rate (1e-3)
- Try higher entropy coefficient (0.05) for more exploration
- Check if value function is learning (value_loss should decrease)
- Verify that action distribution has enough entropy

## Implementation Order Recommendation

1. Start with `_setup_networks_and_optimizers` - get the networks created
2. Implement `act` method - test action sampling works
3. Implement `get_action_and_value` - test data collection  
4. Implement `evaluate_actions` - test policy evaluation
5. Implement `compute_ppo_loss` - this is the hardest part!
6. Finally implement `update` - put it all together

## Debugging Tools

- Use `print()` statements liberally during development
- Check tensor shapes at each step
- Verify gradient flows with `torch.autograd.grad`
- Use TensorBoard to visualize learning curves
- Set `debug: true` in config for verbose logging

## Success Criteria ✅

Your implementation is complete when:
- [ ] Code runs without errors for full training
- [ ] Agent learns to solve CartPole (195+ average reward)  
- [ ] Training is stable across multiple runs
- [ ] All loss components behave as expected
- [ ] Metrics are properly logged

Good luck! This is a substantial but very educational implementation. Take it step by step, test frequently, and don't hesitate to debug thoroughly at each stage.