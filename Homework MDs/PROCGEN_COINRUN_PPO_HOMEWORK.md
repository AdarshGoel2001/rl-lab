# Procgen CoinRun PPO Implementation Homework

**Environment Goal**: Master PPO with procedural generation, train/test generalization, and data augmentation. This teaches you to build agents that generalize beyond their training distribution!

**Why Procgen CoinRun?** This environment teaches you:
- Procedural generation (infinite unique levels)
- Train/test generalization measurement
- Data augmentation for visual robustness  
- IMPALA-style architectures with batch normalization
- Sample efficiency on complex visual tasks

Your roadmap notes: "Generalization across levels; PPO is a standard baseline here. Setup: train on 200 levels, test on ∞; IMPALA-small + batch norm; no reward clipping."

## Phase B.3 - Procgen Environment Setup

### 📋 Procgen Installation & Setup

#### 1. Procgen Dependencies
- [ ] Install Procgen: `pip install procgen`
- [ ] Test CoinRun: `gym.make('procgen:procgen-coinrun-v0')`
- [ ] Verify multiple environments: Test StarPilot, Fruitbot, Dodgeball
- [ ] Check level generation: Confirm different start_level values create different layouts

#### 2. Environment Analysis
- [ ] Observation space: `Box(0, 255, shape=(64, 64, 3))` - RGB 64×64 pixels
- [ ] Action space: `Discrete(15)` - Complex action set including movement, jumping
- [ ] Episode length: Fixed at 1000 timesteps
- [ ] Task: Navigate platformer level, collect coin, reach exit
- [ ] Random policy performance: ~1-3 (very low success rate)

#### 3. Level Generation Understanding
- [ ] **Training Levels**: Specified by start_level and num_levels (e.g., start_level=0, num_levels=200)
- [ ] **Test Levels**: Use start_level=0, num_levels=0 for infinite procedural generation
- [ ] **Difficulty**: Levels have consistent difficulty but different layouts
- [ ] **Scoring**: Reward for collecting coin + reaching exit, time penalties

### Expected Baselines 📊
- **Random Policy**: ~1.5 ± 1.0 average episode return
- **Strong PPO Target**: 8-10 on training levels, 6-8 on test levels
- **Generalization Gap**: Train score - test score (should minimize this)
- **Human Performance**: ~9-10 (near perfect play)

## Phase B.4 - Generalization-Focused Setup

### 📋 Train/Test Split Configuration

Your roadmap specifies: "train on 200 levels, test on ∞"

#### 1. Training Environment Setup
- [ ] **Training Levels**: start_level=0, num_levels=200
- [ ] **Deterministic**: Use fixed seed for reproducible training set
- [ ] **Distribution Mode**: 'easy' for standard difficulty
- [ ] **Level Sampling**: Ensure all 200 levels get seen during training

#### 2. Evaluation Environment Setup
- [ ] **Test Levels**: start_level=0, num_levels=0 (infinite)
- [ ] **Different Seed**: Use different random seed from training
- [ ] **Same Difficulty**: Keep distribution_mode='easy' for fair comparison
- [ ] **Regular Testing**: Evaluate on fresh test levels every N training steps

#### 3. Generalization Metrics
- [ ] **Train Performance**: Average score on training levels
- [ ] **Test Performance**: Average score on never-seen levels  
- [ ] **Generalization Gap**: |Train Score - Test Score|
- [ ] **Level Completion Rate**: % of levels where agent reaches exit

### Environment Configuration Templates

#### Training Environment
```python
# Training: Fixed set of 200 levels
train_env = gym.make(
    'procgen:procgen-coinrun-v0',
    start_level=0,
    num_levels=200,          # Fixed training set
    distribution_mode='easy',
    render_mode=None,
    rand_seed=42             # Deterministic training levels
)

# Test Environment  
test_env = gym.make(
    'procgen:procgen-coinrun-v0', 
    start_level=0,
    num_levels=0,            # Infinite test levels
    distribution_mode='easy',
    render_mode=None,
    rand_seed=123            # Different from training seed
)
```

#### Level Curriculum Wrapper
```python
class LevelCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, num_training_levels=200):
        super().__init__(env)
        self.num_training_levels = num_training_levels
        self.level_counts = np.zeros(num_training_levels)
        
    def reset(self):
        # Ensure balanced sampling of training levels
        level_id = np.argmin(self.level_counts)
        self.level_counts[level_id] += 1
        
        # Reset to specific level
        return self.env.reset(seed=level_id)
        
    def get_level_statistics(self):
        return {
            'level_visits': self.level_counts,
            'min_visits': self.level_counts.min(),
            'max_visits': self.level_counts.max(),
            'uniform_coverage': self.level_counts.std()
        }
```

## Phase B.5 - IMPALA Architecture for Procgen

### 📋 Network Architecture Tasks

Your roadmap specifies: "IMPALA-small + batch norm; no reward clipping"

#### 1. IMPALA-Small Implementation
- [ ] **Residual Blocks**: 3 blocks with skip connections
- [ ] **Channel Progression**: 16 → 32 → 32 channels
- [ ] **Batch Normalization**: After each conv layer (crucial for Procgen)
- [ ] **Global Average Pooling**: More robust than flatten for generalization

#### 2. Batch Normalization Integration
- [ ] **Conv BatchNorm**: After each convolutional layer
- [ ] **Training Mode**: Ensure proper train/eval mode switching
- [ ] **Running Statistics**: Track moving averages across environments  
- [ ] **Generalization**: BatchNorm helps with varying visual styles

#### 3. Architecture Comparison
- [ ] **IMPALA vs Nature CNN**: Compare generalization performance
- [ ] **With/Without BatchNorm**: Measure impact on train/test gap
- [ ] **Different Channel Sizes**: Test 16→32→32 vs larger variants

### IMPALA-Small Implementation

```python
class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.skip_bn = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()
        
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
    def forward(self, x):
        residual = self.skip_bn(self.skip(x))
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        out = self.pool(out)
        
        return out

class ImpalaCNNSmall(nn.Module):
    def __init__(self, input_channels=3, num_actions=15):
        super().__init__()
        
        # IMPALA blocks
        self.block1 = ImpalaBlock(input_channels, 16)  # 64x64 -> 32x32
        self.block2 = ImpalaBlock(16, 32)              # 32x32 -> 16x16  
        self.block3 = ImpalaBlock(32, 32)              # 16x16 -> 8x8
        
        # Global average pooling + small FC
        self.trunk = nn.Linear(32, 256)
        
        # Separate heads
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)
        
    def forward(self, x):
        # x shape: (batch, 3, 64, 64) - no frame stacking needed for Procgen
        x = x.float() / 255.0  # Normalize to [0,1]
        
        x = self.block1(x)  # -> (batch, 16, 32, 32)
        x = self.block2(x)  # -> (batch, 32, 16, 16)
        x = self.block3(x)  # -> (batch, 32, 8, 8)
        
        # Global average pooling
        x = x.mean(dim=(2, 3))  # -> (batch, 32)
        features = F.relu(self.trunk(x))  # -> (batch, 256)
        
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        
        return action_logits, value.squeeze(-1)
```

## Phase B.6 - Data Augmentation Implementation

### 📋 Augmentation Tasks

Your roadmap mentions: "reduced via tiny augments (random shift)"

#### 1. Random Crop/Shift Augmentation
- [ ] **Random Shifts**: Translate image by ±4 pixels in each direction
- [ ] **Crop and Pad**: Maintain 64×64 size after shifting
- [ ] **Training Only**: Apply augmentation only during training, not evaluation
- [ ] **Probability**: Apply augmentation to 50-100% of samples

#### 2. Additional Augmentations (Optional)
- [ ] **Color Jitter**: Slight brightness/contrast variation
- [ ] **Random Horizontal Flip**: If task allows (may break some Procgen games)
- [ ] **Cutout**: Random square patches set to zero
- [ ] **Gaussian Noise**: Small amount of noise for robustness

#### 3. Augmentation Impact Analysis
- [ ] **Baseline**: Train without any augmentation
- [ ] **With Augmentation**: Train with random shifts
- [ ] **Generalization Gap**: Measure reduction in train/test gap
- [ ] **Final Performance**: Ensure augmentation doesn't hurt training performance

### Data Augmentation Implementation

```python
class RandomShiftWrapper(gym.ObservationWrapper):
    def __init__(self, env, shift_range=4):
        super().__init__(env)
        self.shift_range = shift_range
        
    def observation(self, observation):
        if not self.training:  # Only augment during training
            return observation
            
        # Random shift in both directions
        dx = np.random.randint(-self.shift_range, self.shift_range + 1)
        dy = np.random.randint(-self.shift_range, self.shift_range + 1)
        
        # Pad and crop to maintain size
        padded = np.pad(observation, ((self.shift_range, self.shift_range), 
                                    (self.shift_range, self.shift_range), (0, 0)), 
                       mode='edge')
        
        # Extract shifted region
        start_x = self.shift_range + dx
        start_y = self.shift_range + dy
        shifted = padded[start_x:start_x+64, start_y:start_y+64, :]
        
        return shifted

class ColorJitterWrapper(gym.ObservationWrapper):
    def __init__(self, env, brightness=0.1, contrast=0.1):
        super().__init__(env)
        self.brightness = brightness
        self.contrast = contrast
        
    def observation(self, observation):
        if not self.training:
            return observation
            
        # Random brightness and contrast
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        
        # Apply transformations
        obs = observation.astype(np.float32)
        obs = obs * contrast_factor + (brightness_factor - 1) * 128
        obs = np.clip(obs, 0, 255).astype(np.uint8)
        
        return obs
```

## Phase B.7 - PPO Configuration for Procgen

### 📋 Hyperparameter Configuration

#### 1. Procgen-Specific Settings
- [ ] **No Reward Clipping**: Keep raw rewards (per roadmap)
- [ ] **Larger Batch Sizes**: 64k-128k total batch with many parallel envs
- [ ] **Extended Training**: 25M-50M timesteps for full learning
- [ ] **Conservative Clipping**: clip_ratio=0.2 or lower for stability

#### 2. Batch Normalization Considerations
- [ ] **Larger Batch Sizes**: BatchNorm needs sufficient batch size to work well
- [ ] **Environment Synchronization**: Sync BatchNorm stats across parallel envs
- [ ] **Evaluation Mode**: Switch to eval mode during testing
- [ ] **Momentum**: Use appropriate momentum (0.99) for running averages

#### 3. Memory Management
- [ ] **No Frame Stacking**: Procgen doesn't need temporal stacking
- [ ] **Mixed Precision**: Use float16 for CNN forward passes
- [ ] **Gradient Checkpointing**: For deeper IMPALA networks

### Configuration Template
```yaml
# configs/experiments/ppo_coinrun.yaml
experiment:
  name: "ppo_coinrun_generalization"
  seed: 42
  device: "cuda"  # or "mps" for Apple Silicon
  
algorithm:
  name: "ppo"
  lr: 5e-4                    # Higher LR for Procgen
  gamma: 0.999                # Slightly higher discount
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  ppo_epochs: 3               # Fewer epochs for stability
  minibatch_size: 512         # Large minibatches for BatchNorm
  normalize_advantages: true
  max_grad_norm: 0.5
  
environment:
  name: "procgen:procgen-coinrun-v0"
  train_levels: 200           # Fixed training set
  test_levels: 0              # Infinite test set
  distribution_mode: "easy"
  num_envs: 64               # Many parallel environments
  augmentation:
    random_shift: 4          # ±4 pixel random shifts
    apply_prob: 0.5          # 50% of samples augmented
    
network:
  type: "impala_small"       # IMPALA architecture
  use_batch_norm: true       # Essential for Procgen
  channels: 3                # RGB, no frame stacking
  
buffer:
  type: "trajectory"
  size: 32768               # Large buffer for many envs
  
training:
  total_timesteps: 25000000  # 25M timesteps
  eval_frequency: 500000     # Eval every 500k timesteps
  checkpoint_frequency: 1000000
  num_eval_episodes: 100     # Many episodes for stable evaluation
  
# Generalization tracking
evaluation:
  train_levels: [0, 200]     # Evaluate on training levels
  test_levels: [0, 0]        # Evaluate on fresh test levels
  track_generalization_gap: true
```

## Phase B.8 - Training & Success Criteria

### 🎯 Done-When Gates (From Your Roadmap)

**Primary Goal**: "Clear train→test generalization gap measured & reduced via tiny augments (random shift)"

#### 1. Performance Thresholds
- [ ] **Training Performance**: ≥8.0 average score on 200 training levels
- [ ] **Test Performance**: ≥6.5 average score on infinite test levels
- [ ] **Generalization Gap**: <2.0 points between train and test
- [ ] **Sample Efficiency**: Achieve targets within 25M timesteps

#### 2. Augmentation Impact
- [ ] **Baseline Gap**: Measure train/test gap without augmentation
- [ ] **With Augmentation**: Measure gap reduction with random shifts
- [ ] **Gap Reduction**: ≥0.5 point improvement in generalization
- [ ] **Training Performance**: Augmentation doesn't hurt training scores

#### 3. Consistency Validation
- [ ] **Multiple Seeds**: Consistent results across 3 random seeds
- [ ] **Level Coverage**: Agent performs well across diverse test levels
- [ ] **Learning Stability**: Smooth learning curves, no catastrophic forgetting

### Performance Benchmarks 📊
| Metric | Without Augmentation | With Augmentation | Target |
|--------|---------------------|-------------------|--------|
| Train Score | 8.5 | 8.0 | ≥8.0 |
| Test Score | 5.5 | 6.5 | ≥6.5 |
| Generalization Gap | 3.0 | 1.5 | <2.0 |
| Sample Efficiency | 30M steps | 25M steps | ≤25M |

## Phase B.9 - Advanced Analysis & Insights

### 📋 Generalization Analysis Tasks

#### 1. Level Difficulty Analysis
- [ ] **Easy Levels**: Performance on simple layouts
- [ ] **Hard Levels**: Performance on complex/challenging layouts  
- [ ] **Visual Diversity**: Performance across different visual themes
- [ ] **Obstacle Types**: Success rate on different obstacle patterns

#### 2. Learned Feature Analysis
- [ ] **CNN Visualization**: What features does IMPALA learn?
- [ ] **Attention Maps**: Which parts of the image does agent focus on?
- [ ] **Feature Similarity**: Compare features between train and test levels
- [ ] **Augmentation Impact**: How does augmentation change learned features?

#### 3. Failure Mode Analysis
- [ ] **Common Failures**: What causes agent to fail on test levels?
- [ ] **Visual Differences**: How do test levels differ visually from training?
- [ ] **Strategy Generalization**: Does agent learn transferable strategies?
- [ ] **Overfitting Signs**: Does agent memorize training level layouts?

### Analysis Code Templates
```python
# Generalization gap tracking
class GeneralizationTracker:
    def __init__(self):
        self.train_scores = []
        self.test_scores = []
        self.gaps = []
        
    def log_evaluation(self, train_score, test_score):
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        gap = train_score - test_score
        self.gaps.append(gap)
        
        print(f"Train: {train_score:.2f}, Test: {test_score:.2f}, Gap: {gap:.2f}")
        
    def get_final_metrics(self):
        return {
            'final_train_score': np.mean(self.train_scores[-10:]),
            'final_test_score': np.mean(self.test_scores[-10:]),
            'final_gap': np.mean(self.gaps[-10:]),
            'gap_reduction': self.gaps[0] - self.gaps[-1]
        }

# Level difficulty analyzer
class LevelAnalyzer:
    def __init__(self, env):
        self.env = env
        self.level_scores = {}
        
    def evaluate_level_range(self, policy, start_level, num_levels):
        level_scores = []
        for level in range(start_level, start_level + num_levels):
            obs = self.env.reset(seed=level)
            episode_reward = 0
            done = False
            
            while not done:
                action = policy.act(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
            level_scores.append(episode_reward)
            
        return {
            'mean_score': np.mean(level_scores),
            'std_score': np.std(level_scores),
            'min_score': np.min(level_scores),
            'max_score': np.max(level_scores),
            'level_scores': level_scores
        }
```

## Phase B.10 - Mac Training Optimizations

### 📋 Mac-Specific Tasks

#### 1. Memory Management
- [ ] **Batch Size Tuning**: Reduce if GPU memory limited
- [ ] **Environment Count**: Use 32 envs instead of 64 on Mac
- [ ] **Mixed Precision**: Essential for large batch BatchNorm
- [ ] **Gradient Checkpointing**: For deeper IMPALA networks

#### 2. Compute Optimizations
- [ ] **MPS Backend**: Use Apple Silicon GPU acceleration
- [ ] **CPU Parallelism**: Efficient environment parallelization  
- [ ] **Model Compilation**: Use torch.compile if supported
- [ ] **Data Loading**: Optimize observation preprocessing

#### 3. Training Monitoring
- [ ] **Checkpoint Strategy**: Save frequently due to long training
- [ ] **Early Stopping**: Monitor generalization gap convergence
- [ ] **Resource Monitoring**: Track GPU memory and CPU usage

### Mac Optimization Code
```python
# Mac-friendly Procgen training setup
import torch

# Device setup for Mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)

# Reduce memory usage
config_mac = {
    'num_envs': 32,           # Reduce from 64
    'minibatch_size': 256,    # Reduce from 512
    'buffer_size': 16384,     # Reduce from 32768
    'mixed_precision': True,  # Essential for memory
}

class MacOptimizedImpalaCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ... same architecture ...
        
        # Enable gradient checkpointing for memory
        self.use_checkpoint = True
        
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
```

## Phase B.11 - Debugging & Common Issues

### 📋 Debugging Tasks

#### Problem: Poor Generalization (Large Train/Test Gap)
**Symptoms**: High training score but low test score
**Solutions**:
- [ ] Increase data augmentation strength
- [ ] Add more regularization (dropout, weight decay)
- [ ] Reduce model capacity (fewer channels)
- [ ] Increase training set diversity (more levels)

#### Problem: BatchNorm Issues
**Symptoms**: Training instability or poor performance
**Solutions**:
- [ ] Ensure large enough batch sizes (≥256)
- [ ] Sync BatchNorm across parallel environments
- [ ] Check train/eval mode switching
- [ ] Try Group Normalization instead

#### Problem: Slow Learning on Procgen
**Symptoms**: No improvement after 5M+ timesteps
**Solutions**:
- [ ] Increase learning rate to 5e-4 or 1e-3
- [ ] Use larger batch sizes (128k)
- [ ] Add exploration bonuses
- [ ] Check reward scale (don't clip rewards)

#### Problem: Memory Issues with Many Environments
**Symptoms**: OOM errors or very slow training
**Solutions**:
- [ ] Reduce num_envs from 64 to 32 or 16
- [ ] Use gradient accumulation instead of large batches
- [ ] Enable mixed precision training
- [ ] Reduce network size

## Quick Start Commands 🚀

### 1. Environment Test
```bash
# Test Procgen installation
python -c "import procgen; import gymnasium as gym; env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=1); print(env.reset()[0].shape)"
```

### 2. Level Visualization
```bash
# Generate and view different levels
python scripts/visualize_levels.py --env coinrun --levels 5
```

### 3. Architecture Test
```bash  
# Test IMPALA network
python scripts/test_network.py --arch impala_small --input-shape 3 64 64 --batch-norm
```

### 4. Training Test
```bash
# Short training test
python scripts/train.py --config configs/experiments/ppo_coinrun.yaml --debug --total-timesteps 100000
```

### 5. Full Training with Generalization Tracking
```bash
# Full training with train/test evaluation
python scripts/train.py --config configs/experiments/ppo_coinrun.yaml --track-generalization
```

## Success Checklist ✅

Your Procgen CoinRun PPO is complete when:
- [ ] Achieves ≥8.0 on training levels and ≥6.5 on test levels  
- [ ] Demonstrates clear generalization gap reduction with augmentation
- [ ] Shows smooth learning curves over 25M timesteps
- [ ] IMPALA architecture with BatchNorm trains stably
- [ ] Ready for other Procgen environments and algorithm comparisons

## What You've Learned 🎓

By completing this homework, you've mastered:
- **Generalization in RL**: Train/test splits and measuring overfitting
- **Data Augmentation**: Visual robustness through random shifts
- **IMPALA Architecture**: Residual blocks with BatchNorm for visual RL
- **Procedural Generation**: Training on diverse, algorithmically generated content
- **Long-Horizon Training**: Managing 25M+ timestep training runs

**Next Up**: Hyperparameter sweep system - Systematically optimize all your algorithms!