# Atari Pong PPO Implementation Homework

**Environment Goal**: Master PPO on real pixel environments with deep CNNs, Atari preprocessing, and long training horizons. This is your entry into the classic deep RL domain!

**Why Pong?** This environment teaches you:
- Raw pixel processing (210×160×3 RGB → 84×84×1 grayscale)
- Deep CNN architectures (Nature CNN, IMPALA)
- Atari-specific preprocessing pipeline  
- Credit assignment across long episodes (1000+ steps)
- Sample efficiency with 10-20M frame budget

Your roadmap notes: "Classic pixel control; PPO succeeds with standard preprocessing. Budget: ~10-20M frames for textbook curves."

## Phase B.1 - Atari Environment Setup

### 📋 Environment Installation & Setup

#### 1. Atari Dependencies
- [ ] Install Atari environments: `pip install "gymnasium[atari,accept-rom-license]"`
- [ ] Verify PongNoFrameskip-v4 loads: `gym.make('PongNoFrameskip-v4')`
- [ ] Test other Atari games for later: `BreakoutNoFrameskip-v4`, `SpaceInvadersNoFrameskip-v4`
- [ ] Check ROM availability and legal compliance

#### 2. Environment Analysis
- [ ] Raw observation space: `Box(0, 255, shape=(210, 160, 3))` - RGB pixels
- [ ] Action space: `Discrete(6)` - [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
- [ ] Effective actions: Often simplified to [NOOP, RIGHT, LEFT] for Pong
- [ ] Episode length: Variable, typically 1000-8000 steps
- [ ] Random policy performance: ~-21 (loses every game)

#### 3. Reward Structure Understanding
- [ ] Pong scoring: +1 for winning point, -1 for losing point, 0 otherwise
- [ ] Game ends at ±21 points (first to 21 wins)
- [ ] Episode reward typically in range [-21, +21]
- [ ] Random policy gets ~-20 to -21 (almost always loses)

### Expected Baselines 📊
- **Random Policy**: -20.5 ± 1.0 average episode return
- **Human Performance**: ~15-20 average episode return  
- **Strong PPO Target**: +15 to +20 average episode return
- **Episode Length**: 1000-8000 steps depending on skill level

## Phase B.2 - Atari Preprocessing Pipeline

### 📋 Preprocessing Implementation Tasks

Your roadmap specifies: "grayscale → resize 84×84 → frame-stack 4; sticky-actions; max-no-op starts; reward clipping to ±1"

#### 1. Standard Atari Preprocessing Chain
- [ ] **Grayscale conversion**: RGB (210,160,3) → Grayscale (210,160,1)
- [ ] **Resize**: (210,160) → (84,84) using area interpolation
- [ ] **Frame stacking**: Stack 4 consecutive frames → (84,84,4)
- [ ] **Frame skipping**: Already handled by "NoFrameskip" environment
- [ ] **Life management**: Reset on life loss (Pong doesn't have lives)

#### 2. Atari-Specific Wrappers
- [ ] **MaxNoOp**: Random no-op actions (0-30) at episode start
- [ ] **Sticky actions**: 25% chance previous action repeats
- [ ] **Reward clipping**: Clip rewards to {-1, 0, +1}
- [ ] **Episode termination**: Handle Atari episode endings properly

#### 3. Observation Normalization
- [ ] Scale pixels to [0,1] range: `obs = obs / 255.0`
- [ ] Channel-first format for PyTorch: (B,C,H,W) vs (B,H,W,C)
- [ ] Memory-efficient frame stacking (avoid copying)

### Preprocessing Implementation Templates

#### Core Atari Wrappers
```python
import cv2
import numpy as np
from collections import deque

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        
    def observation(self, observation):
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = tuple(shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )
        
    def observation(self, observation):
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high)
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation()
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info
        
    def _get_observation(self):
        return np.stack(list(self.frames), axis=0)

class RewardClipper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)  # Maps to {-1, 0, 1}

class MaxNoOpStarts(gym.Wrapper):
    def __init__(self, env, max_noop=30):
        super().__init__(env)
        self.max_noop = max_noop
        
    def reset(self):
        obs = self.env.reset()
        num_noops = np.random.randint(0, self.max_noop + 1)
        for _ in range(num_noops):
            obs, _, done, _ = self.env.step(0)  # NOOP action
            if done:
                obs = self.env.reset()
        return obs

class StickyActions(gym.ActionWrapper):
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0
        
    def action(self, action):
        if np.random.random() < self.p:
            return self.last_action
        else:
            self.last_action = action
            return action
```

#### Complete Atari Preprocessing Function
```python
def make_atari_env(env_name="PongNoFrameskip-v4"):
    env = gym.make(env_name)
    env = MaxNoOpStarts(env, max_noop=30)
    env = StickyActions(env, p=0.25)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=4)
    env = RewardClipper(env)
    
    # Scale to [0,1] and ensure channel-first
    env = ScaleAndTranspose(env)
    return env
```

## Phase B.3 - CNN Architecture Implementation

### 📋 CNN Architecture Tasks

Your roadmap suggests: "Nature-CNN or IMPALA-small; value head shares trunk"

#### 1. Nature CNN Implementation
- [ ] **Conv1**: 32 filters, 8×8 kernel, stride 4 → (32, 20, 20)
- [ ] **Conv2**: 64 filters, 4×4 kernel, stride 2 → (64, 9, 9) 
- [ ] **Conv3**: 64 filters, 3×3 kernel, stride 1 → (64, 7, 7)
- [ ] **Flatten**: → 3136 features
- [ ] **FC**: → 512 hidden units
- [ ] **Heads**: Actor (action_dim) and Critic (1) outputs

#### 2. IMPALA-Small Implementation
- [ ] **Residual blocks**: 2-3 blocks with skip connections
- [ ] **Channel progression**: 16 → 32 → 32 channels
- [ ] **Global average pooling**: More parameter efficient
- [ ] **Smaller final FC**: 256 hidden units

#### 3. Shared Trunk Architecture
- [ ] Single CNN encoder shared between actor and critic
- [ ] Separate heads for policy and value
- [ ] Test shared vs separate networks
- [ ] Monitor for optimization conflicts

### CNN Architecture Templates

#### Nature CNN
```python
class NatureCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        self.conv_output_size = 64 * 7 * 7  # 3136 for 84x84 input
        
        # Shared trunk
        self.trunk = nn.Linear(self.conv_output_size, 512)
        
        # Separate heads
        self.actor_head = nn.Linear(512, num_actions)
        self.critic_head = nn.Linear(512, 1)
        
    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))  # -> (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # -> (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # -> (batch, 64, 7, 7)
        
        x = x.view(x.size(0), -1)  # Flatten -> (batch, 3136)
        features = F.relu(self.trunk(x))  # -> (batch, 512)
        
        action_logits = self.actor_head(features)  # -> (batch, num_actions)
        value = self.critic_head(features)         # -> (batch, 1)
        
        return action_logits, value

class ImpalaCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()
        
        # IMPALA-style residual blocks
        self.block1 = self._make_block(input_channels, 16)
        self.block2 = self._make_block(16, 32)  
        self.block3 = self._make_block(32, 32)
        
        # Global average pooling instead of flatten
        self.trunk = nn.Linear(32, 256)
        
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)
        
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x)
        
        # Global average pooling
        x = x.mean(dim=(2, 3))  # -> (batch, 32)
        features = F.relu(self.trunk(x))
        
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        
        return action_logits, value
```

## Phase B.4 - PPO Configuration for Atari

### 📋 Hyperparameter Configuration Tasks

#### 1. Atari-Specific PPO Settings
- [ ] **Large batch sizes**: 256-512 minibatch, 2048-8192 total batch
- [ ] **Multiple environments**: 8-32 parallel environments for data collection
- [ ] **Extended training**: 10-20M frames (2.5-5M timesteps with 4-frame stacking)
- [ ] **Learning rate**: 2.5e-4 (slightly lower than classics)

#### 2. CNN-Specific Optimizations
- [ ] **Gradient clipping**: max_grad_norm=0.5 (important for deep networks)
- [ ] **Learning rate warmup**: Optional linear warmup for first 10k steps
- [ ] **Batch normalization**: Sometimes helps CNN training
- [ ] **Weight initialization**: Xavier/He initialization for conv layers

#### 3. Memory Management
- [ ] **Observation caching**: Avoid repeated preprocessing
- [ ] **Mixed precision**: Use float16 for CNN forward passes
- [ ] **Gradient checkpointing**: If memory constrained on Mac

### Configuration Template
```yaml
# configs/experiments/ppo_pong.yaml
experiment:
  name: "ppo_pong_nature_cnn"
  seed: 42
  device: "cuda"  # or "mps" for Apple Silicon
  
algorithm:
  name: "ppo"
  lr: 2.5e-4                  # Lower LR for deep networks
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.1             # Conservative clipping for Atari
  value_coef: 0.5
  entropy_coef: 0.01
  ppo_epochs: 4               # 4 epochs per batch
  minibatch_size: 256         # Larger minibatches
  normalize_advantages: true
  max_grad_norm: 0.5          # Essential for CNN stability
  
environment:
  name: "PongNoFrameskip-v4"
  wrapper: "atari"
  num_envs: 16                # Many parallel environments
  preprocessing:
    grayscale: true
    resize: [84, 84]
    frame_stack: 4
    reward_clipping: true
    sticky_actions: 0.25
    max_no_op: 30
    
network:
  type: "nature_cnn"          # or "impala_cnn"
  channels: 4                 # 4 stacked frames
  
buffer:
  type: "trajectory"
  size: 8192                  # Large buffer for many envs
  
training:
  total_timesteps: 10000000   # 10M timesteps (40M frames)
  eval_frequency: 250000      # Eval every 1M frames
  checkpoint_frequency: 500000
  num_eval_episodes: 10
  
# Mac-specific optimizations
system:
  mixed_precision: true       # Memory savings
  compile_model: false        # May not work on MPS
  num_workers: 4              # Parallel env workers
```

## Phase B.5 - Training & Success Criteria

### 🎯 Done-When Gates (From Your Roadmap)

**Primary Goal**: "Beats −5 → +15 trajectory on learning curve and stabilizes across 3 seeds"

#### 1. Learning Curve Targets
- [ ] **Early Learning**: Show improvement from -21 within first 2M frames
- [ ] **Mid Training**: Cross 0 average reward (neutral) by 5M frames  
- [ ] **Target Performance**: Reach +15 average reward by 10M frames
- [ ] **Stabilization**: Performance stable ±2 points around +15

#### 2. Multi-Seed Validation
- [ ] **Consistency**: All 3 seeds achieve similar learning trajectories
- [ ] **Reproducibility**: Same hyperparameters work across seeds
- [ ] **Final Performance**: All seeds end up ≥+13 average reward

#### 3. Additional Metrics
- [ ] **Sample Efficiency**: Learning visible by 1M frames, strong by 5M
- [ ] **Policy Quality**: Agent learns proper Pong strategy (positioning, hitting)
- [ ] **Exploration**: Action entropy decreases appropriately over training

### Performance Benchmarks 📊
| Training Phase | Frame Count | Expected Reward | Key Milestones |
|----------------|-------------|----------------|----------------|
| Initial | 0-500k | -21 to -15 | Random → Some learning |
| Early Learning | 500k-2M | -15 to -5 | Consistent improvement |
| Mid Training | 2M-5M | -5 to +5 | Cross neutral, beat random |
| Target Performance | 5M-10M | +5 to +15 | Approach human-level |
| Stable Performance | 10M+ | +15 ± 2 | Consistent strong play |

## Phase B.6 - Mac-Specific Optimizations

### 📋 Mac Training Tasks

#### 1. Apple Silicon Optimizations
- [ ] **MPS Backend**: Set `device="mps"` for M1/M2 acceleration
- [ ] **Memory Management**: Monitor GPU memory usage, reduce batch size if needed
- [ ] **Float32 Default**: Use `torch.set_default_dtype(torch.float32)` for MPS compatibility
- [ ] **Avoid Unsupported Ops**: Some PyTorch ops not yet supported on MPS

#### 2. CPU Fallback Strategies
- [ ] **Parallel Environments**: Reduce num_envs to 8-12 on CPU
- [ ] **Smaller Networks**: Use IMPALA-small instead of Nature CNN if needed
- [ ] **Reduced Batch Size**: 2048 total, 128 minibatch for memory limits
- [ ] **Mixed Precision**: Use autocast for memory savings

#### 3. Training Monitoring
- [ ] **TensorBoard Setup**: Monitor learning curves, CNN feature maps
- [ ] **Checkpoint Strategy**: Save every 1M frames due to long training times
- [ ] **Early Stopping**: Monitor for convergence to save compute

### Mac Optimization Code
```python
# Mac-friendly training setup
import torch

# Set device and dtype for Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_dtype(torch.float32)  # MPS requires float32
else:
    device = torch.device("cpu")
    
# Reduce parallel environments for Mac
num_envs = 8 if device.type == "cpu" else 16

# Memory-efficient CNN training
class MemoryEfficientNatureCNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()
        # ... same conv layers ...
        
        # Gradient checkpointing for memory savings
        self.use_checkpointing = True
        
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
            
    def _forward_impl(self, x):
        # ... same forward logic ...
```

## Phase B.7 - Training Process & Debugging

### 📋 Training Execution Tasks

#### 1. Progressive Training Setup
- [ ] **Start Small**: Test on single environment first
- [ ] **Scale Up**: Gradually increase to 16 parallel environments
- [ ] **Monitor Metrics**: Track FPS, memory usage, gradient norms
- [ ] **Checkpoint Strategy**: Save every 1M frames for long training

#### 2. Learning Curve Analysis
- [ ] **Early Phase (0-1M)**: Should see some deviation from pure random
- [ ] **Learning Onset (1-3M)**: Clear upward trend in episode rewards
- [ ] **Acceleration (3-7M)**: Rapid improvement phase
- [ ] **Convergence (7-10M)**: Approach final performance level

#### 3. Common Atari Training Issues

#### Problem: No Learning After 2M Frames
**Symptoms**: Reward stuck around -20, no improvement
**Solutions**:
- [ ] Check preprocessing pipeline (ensure correct shapes)
- [ ] Verify reward clipping is working
- [ ] Increase batch size to 8192
- [ ] Check CNN gradient flow (gradient norms should be > 0)

#### Problem: Learning Then Catastrophic Forgetting
**Symptoms**: Improves to +5 then drops back to -15
**Solutions**:
- [ ] Reduce learning rate to 1e-4
- [ ] Reduce clip_ratio to 0.05  
- [ ] Add gradient clipping (max_grad_norm=0.3)
- [ ] Use more conservative PPO (fewer epochs=2)

#### Problem: Unstable/Noisy Learning
**Symptoms**: Wild oscillations in performance
**Solutions**:
- [ ] Increase number of evaluation episodes
- [ ] Use longer evaluation periods (average over more episodes)
- [ ] Check for numerical instabilities (NaN in losses)
- [ ] Verify frame stacking is working correctly

#### Problem: Memory Issues During Training
**Symptoms**: OOM errors or very slow training
**Solutions**:
- [ ] Reduce batch_size and minibatch_size
- [ ] Use gradient accumulation instead of large batches
- [ ] Reduce number of parallel environments
- [ ] Enable mixed precision training

### Debugging Code Templates
```python
# Comprehensive training monitoring
class AtariTrainingMonitor:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.gradient_norms = []
        
    def log_episode(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Print running statistics
        if len(self.episode_rewards) % 100 == 0:
            recent_rewards = self.episode_rewards[-100:]
            print(f"Episodes {len(self.episode_rewards)}: "
                  f"Mean Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                  
    def log_training_step(self, metrics):
        self.gradient_norms.append(metrics.get('grad_norm', 0))
        
        # Check for training issues
        if metrics['policy_loss'] != metrics['policy_loss']:  # NaN check
            print("WARNING: NaN detected in policy loss!")
            
        if metrics['grad_norm'] == 0:
            print("WARNING: Zero gradient norm - check learning rate!")

# Preprocessing verification
def verify_preprocessing(env):
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]") 
    print(f"Observation dtype: {obs.dtype}")
    
    # Check frame stacking
    action = env.action_space.sample()
    for i in range(5):
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward}, done={done}")
        if done:
            obs = env.reset()
```

## Phase B.8 - Advanced Analysis & Evaluation

### 📋 Analysis Tasks

#### 1. Policy Visualization
- [ ] **Record Videos**: Save episode recordings at different training stages
- [ ] **Action Analysis**: Plot action distribution evolution over training
- [ ] **Strategy Analysis**: Verify agent learns proper Pong positioning
- [ ] **Failure Analysis**: Identify common failure modes

#### 2. CNN Feature Analysis  
- [ ] **Feature Maps**: Visualize what CNN layers detect
- [ ] **Activation Analysis**: Check for dead neurons or saturation
- [ ] **Gradient Analysis**: Verify all layers receive meaningful gradients
- [ ] **Architecture Comparison**: Compare Nature CNN vs IMPALA performance

#### 3. Sample Efficiency Analysis
- [ ] **Learning Speed**: Compare your results to published baselines
- [ ] **Data Efficiency**: Frames needed to reach different performance thresholds
- [ ] **Hyperparameter Sensitivity**: Which settings matter most for Pong
- [ ] **Transfer Potential**: How well does this setup work on other Atari games

## Quick Start Commands 🚀

### 1. Environment Test
```bash
# Test Atari installation and preprocessing
python -c "
import gymnasium as gym
from src.envs.atari_wrappers import make_atari_env
env = make_atari_env('PongNoFrameskip-v4')
obs = env.reset()
print(f'Preprocessed obs shape: {obs.shape}')
print(f'Action space: {env.action_space}')
env.step(1)  # Test step
"
```

### 2. CNN Architecture Test
```bash
# Test CNN forward pass
python scripts/test_network.py --arch nature_cnn --input-shape 4 84 84 --batch-size 32
```

### 3. Short Training Test
```bash
# Test full pipeline without long training
python scripts/train.py --config configs/experiments/ppo_pong.yaml --debug --total-timesteps 50000
```

### 4. Full Training  
```bash
# Start full Pong training (will take several hours)
python scripts/train.py --config configs/experiments/ppo_pong.yaml --name pong_full_training
```

### 5. Multi-Seed Validation
```bash
# Train 3 seeds for consistency (can run in parallel)
for seed in 1 2 3; do
  python scripts/train.py --config configs/experiments/ppo_pong.yaml --seed $seed --name pong_seed_$seed &
done
wait  # Wait for all to complete
```

## Success Checklist ✅

Your Atari Pong PPO is complete when:
- [ ] Learns from -21 to +15 average reward within 10M frames
- [ ] Shows consistent learning trajectory across 3 different seeds
- [ ] Demonstrates proper Pong strategy (positioning, ball tracking)
- [ ] CNN architecture processes 84×84×4 frames efficiently
- [ ] Ready for more complex Atari games (Breakout, SpaceInvaders)

## What You've Learned 🎓

By completing this homework, you've mastered:
- **Deep CNN Architectures**: Nature CNN and IMPALA for visual RL
- **Atari Preprocessing**: The full pipeline from raw pixels to training data  
- **Large-Scale Training**: Managing 10M+ timestep training runs
- **Sample Efficiency**: Achieving strong performance within reasonable budgets
- **Mac Optimization**: Training deep RL on Apple Silicon efficiently

**Next Up**: Breakout - More complex credit assignment and exploration challenges!