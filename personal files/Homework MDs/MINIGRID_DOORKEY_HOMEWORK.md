# MiniGrid DoorKey PPO Implementation Homework

**Environment Goal**: Master PPO with partial observability, exploration challenges, and visual/symbolic observations. This is your introduction to environments that require memory and strategic exploration!

**Why MiniGrid DoorKey?** This environment teaches you:
- Partial observability (7×7 egocentric view vs full grid)
- Exploration strategies (find key, then door, then goal)
- Visual observation processing (CNN vs one-hot encoding)
- Frame stacking for temporal information
- Curriculum learning (5x5 → 8x8 grid sizes)

## Phase A.3 - Partial Observability & Visual Processing

### 📋 Environment Analysis Tasks

#### 1. MiniGrid Installation & Setup
- [ ] Install minigrid: `pip install minigrid[classic]`
- [ ] Test DoorKey-5x5v0: `gym.make('MiniGrid-DoorKey-5x5-v0')`
- [ ] Test DoorKey-8x8v0: `gym.make('MiniGrid-DoorKey-8x8-v0')`
- [ ] Verify environments load without errors
- [ ] Record action space: `Discrete(7)` - [left, right, forward, pickup, drop, toggle, done]

#### 2. Observation Space Investigation
- [ ] Default obs: `Dict` with 'image' (7×7×3) and 'direction' (scalar)
- [ ] Image contains object/color/state encoding (not raw pixels)
- [ ] Direction indicates agent's facing direction (0-3)
- [ ] Measure random policy performance on 5x5: should be ~0% success
- [ ] Test episode lengths: variable (usually 50-200 steps)

#### 3. Task Structure Analysis
- [ ] **Step 1**: Agent spawns in random location
- [ ] **Step 2**: Find and collect the key (yellow key object)
- [ ] **Step 3**: Navigate to locked door (colored door)
- [ ] **Step 4**: Open door with key ('toggle' action)
- [ ] **Step 5**: Reach green goal square
- [ ] Record success reward: +1 for reaching goal, -0.01 per step otherwise

### Expected Results 📊
- **Random Policy 5x5**: 0% success rate, ~-1.0 average return
- **Random Policy 8x8**: 0% success rate, ~-2.0 average return  
- **Episode Length**: Highly variable (10-1000 steps)
- **Observation Shape**: (7, 7, 3) image + direction scalar

## Phase A.4 - Observation Processing Strategies

### 📋 Observation Wrapper Tasks

Your roadmap suggests: "one-hot encoding or simple CNN for 7×7 egocentric view; frame-stack 4; sticky-action=False"

#### 1. One-Hot Encoding Approach
- [ ] Flatten 7×7×3 observation to 147-dimensional vector
- [ ] Convert object/color/state codes to one-hot vectors
- [ ] Concatenate with direction information
- [ ] Test with standard MLP networks

#### 2. Simple CNN Approach  
- [ ] Design lightweight CNN for 7×7 input
- [ ] 2-3 conv layers with small kernels (3×3)
- [ ] Small channel counts (16, 32, 64)
- [ ] Global average pooling or flatten to FC
- [ ] Add direction as auxiliary input

#### 3. Frame Stacking Implementation
- [ ] Stack 4 consecutive frames for temporal memory
- [ ] Handle episode resets (duplicate initial frame)
- [ ] Memory management for stacked observations
- [ ] Test impact on learning vs single frame

### Implementation Templates

#### One-Hot Wrapper
```python
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define one-hot sizes for MiniGrid objects
        self.object_to_idx = {...}  # mapping of object types
        self.color_to_idx = {...}   # mapping of colors
        # New observation space: flattened one-hot vector
        
    def observation(self, obs):
        image = obs['image']  # (7, 7, 3)
        direction = obs['direction']
        
        # Convert to one-hot and flatten
        one_hot = self._to_one_hot(image)  # shape: (147 * num_categories)
        return np.concatenate([one_hot, [direction]])
```

#### Simple CNN Architecture
```python
class MiniGridCNN(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.fc = nn.Linear(64 + 1, hidden_size)  # +1 for direction
        
    def forward(self, obs):
        image = obs['image'].permute(0, 3, 1, 2)  # (B, 7, 7, 3) -> (B, 3, 7, 7)
        direction = obs['direction']
        
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.mean(dim=(2, 3))  # Global average pooling (B, 64)
        x = torch.cat([x, direction.unsqueeze(1)], dim=1)  # (B, 65)
        
        return self.fc(x)
```

#### Frame Stacking Wrapper
```python
class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
        
    def _get_obs(self):
        return np.stack(list(self.frames), axis=0)  # (k, 7, 7, 3)
```

## Phase A.5 - PPO Configuration for Exploration

### 📋 Hyperparameter Tuning Tasks

#### 1. Exploration-Focused Settings
- [ ] **High entropy coefficient**: Start with 0.05-0.1 (vs 0.01 for CartPole)
- [ ] **Larger batch sizes**: 4096-8192 for better advantage estimates
- [ ] **Extended training**: 500k-1M timesteps (exploration takes time)
- [ ] **Learning rate**: 3e-4 or 2.5e-4 (standard)

#### 2. Network Architecture Choices
- [ ] **Option A**: One-hot + MLP with [128, 128, 128] layers
- [ ] **Option B**: CNN + smaller FC layers [64, 64]
- [ ] Test both approaches and compare performance
- [ ] Consider shared vs separate actor/critic networks

#### 3. Environment Configuration
- [ ] Set `sticky_actions=False` (per roadmap recommendation)
- [ ] Start with DoorKey-5x5 for quicker iteration
- [ ] Use multiple parallel environments (4-8) for data collection

### Configuration Template
```yaml
# configs/experiments/ppo_doorkey.yaml
experiment:
  name: "ppo_doorkey_exploration"
  seed: 42
  device: "cuda"  # or "mps" for Apple Silicon
  
algorithm:
  name: "ppo"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.1           # High exploration!
  ppo_epochs: 4
  minibatch_size: 256
  normalize_advantages: true
  
environment:
  name: "MiniGrid-DoorKey-5x5-v0"
  wrapper: "minigrid"
  observation_wrapper: "onehot"  # or "cnn"
  frame_stack: 4
  sticky_actions: false
  num_envs: 8                 # Parallel environments
  
network:
  type: "mlp"  # or "cnn" 
  hidden_dims: [128, 128, 128]  # Larger for complex task
  activation: "relu"
  
buffer:
  type: "trajectory"
  size: 8192                  # Larger buffer
  
training:
  total_timesteps: 500000     # Extended training
  eval_frequency: 25000
  checkpoint_frequency: 50000
  num_eval_episodes: 20       # More eval episodes
```

## Phase A.6 - Curriculum Learning Implementation

### 📋 Curriculum Tasks

Your roadmap specifies: "≥95% success on 5x5; 8x8 reaches >70% within your step budget"

#### 1. Phase 1: Master 5x5
- [ ] Train exclusively on DoorKey-5x5 first
- [ ] Target: ≥95% success rate within 300k timesteps
- [ ] Measure success rate every 10k timesteps
- [ ] Don't move to 8x8 until 5x5 is mastered

#### 2. Phase 2: Transfer to 8x8
- [ ] Either continue training on 8x8 OR restart with 5x5 learned weights
- [ ] Target: >70% success rate within remaining timesteps  
- [ ] Compare transfer learning vs fresh start
- [ ] Document performance differences

#### 3. Curriculum Strategies
- [ ] **Strategy A**: Sequential (5x5 → 8x8)
- [ ] **Strategy B**: Mixed training (70% 5x5, 30% 8x8)
- [ ] **Strategy C**: Adaptive difficulty based on success rate
- [ ] Test which approach works best

### Curriculum Implementation
```python
class CurriculumWrapper:
    def __init__(self):
        self.envs = {
            'easy': gym.make('MiniGrid-DoorKey-5x5-v0'),
            'hard': gym.make('MiniGrid-DoorKey-8x8-v0')
        }
        self.current_level = 'easy'
        self.success_buffer = deque(maxlen=100)
        
    def should_upgrade(self):
        if len(self.success_buffer) >= 50:
            success_rate = np.mean(self.success_buffer)
            return success_rate > 0.90  # 90% success on easy
        return False
        
    def step(self, action):
        obs, reward, done, info = self.envs[self.current_level].step(action)
        
        if done:
            success = reward > 0  # Reached goal
            self.success_buffer.append(success)
            
            if self.current_level == 'easy' and self.should_upgrade():
                print("Upgrading to hard level!")
                self.current_level = 'hard'
                
        return obs, reward, done, info
```

## Phase A.7 - Training & Success Criteria

### 🎯 Done-When Gates (From Your Roadmap)

**Primary Goal**: ≥95% success on 5x5; 8x8 reaches >70% within your step budget

#### 1. 5x5 Performance Thresholds
- [ ] **Success Rate**: ≥95% over last 100 episodes
- [ ] **Sample Efficiency**: Achieve this within 300k timesteps
- [ ] **Consistency**: Stable performance across multiple seeds
- [ ] **Strategy**: Agent consistently finds key → door → goal

#### 2. 8x8 Performance Thresholds
- [ ] **Success Rate**: >70% over last 100 episodes  
- [ ] **Step Budget**: Within total 500k timestep budget
- [ ] **Transfer**: Benefit from 5x5 knowledge visible
- [ ] **Exploration**: Covers larger space efficiently

#### 3. Additional Validation
- [ ] **Episode Length**: Decreasing over training (more efficient paths)
- [ ] **Exploration**: Visits all reachable states during training
- [ ] **Generalization**: Success on unseen random seeds

### Performance Benchmarks 📊
| Environment | Random Policy | Target Performance |
|-------------|---------------|-------------------|
| DoorKey-5x5 | 0% success | ≥95% success |
| DoorKey-8x8 | 0% success | >70% success |
| Episode Length (5x5) | ~100 steps | <30 steps |
| Episode Length (8x8) | ~200 steps | <60 steps |

## Phase A.8 - Advanced Features & Analysis

### 📋 Enhancement Tasks

#### 1. Curiosity & Intrinsic Motivation
- [ ] Implement count-based exploration bonuses
- [ ] Add state visitation tracking
- [ ] Test impact on exploration efficiency
- [ ] Compare with pure entropy-based exploration

#### 2. Memory Mechanisms  
- [ ] Test LSTM/GRU policies for memory
- [ ] Compare with frame stacking approach
- [ ] Measure performance on larger grids requiring memory

#### 3. Visualization & Analysis
- [ ] Generate heatmaps of state visitation
- [ ] Visualize learned policies (trajectories)
- [ ] Plot exploration efficiency over training
- [ ] Create success rate learning curves

### Advanced Implementation Examples
```python
# Count-based exploration bonus
class ExplorationBonusWrapper(gym.Wrapper):
    def __init__(self, env, bonus_scale=0.1):
        super().__init__(env)
        self.visit_counts = {}
        self.bonus_scale = bonus_scale
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get state representation (agent position + grid state)
        state_key = self._get_state_key()
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
        
        # Add exploration bonus (inversely proportional to visit count)
        bonus = self.bonus_scale / np.sqrt(self.visit_counts[state_key])
        reward += bonus
        
        return obs, reward, done, info

# LSTM policy for memory
class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        self.encoder = MiniGridCNN()  # Process visual input
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)
        
    def forward(self, obs, hidden=None):
        features = self.encoder(obs)
        lstm_out, hidden = self.lstm(features.unsqueeze(1), hidden)
        
        action_logits = self.actor_head(lstm_out.squeeze(1))
        value = self.critic_head(lstm_out.squeeze(1))
        
        return action_logits, value, hidden
```

## Phase A.9 - Debugging Guide

### Common MiniGrid Issues 🔧

#### Problem: Agent Gets Stuck at Walls
**Symptoms**: Keeps trying to move forward into walls
**Solutions**:
- [ ] Check observation processing (agent should see walls)
- [ ] Increase exploration (higher entropy_coef)
- [ ] Add negative reward for invalid actions
- [ ] Verify action space mapping is correct

#### Problem: Can't Find Key/Door
**Symptoms**: Random wandering, low success rate
**Solutions**:
- [ ] Increase exploration bonus
- [ ] Use count-based or curiosity-driven exploration
- [ ] Extend episode length limits
- [ ] Add intermediate rewards (distance to key/door)

#### Problem: Finds Key but Can't Open Door
**Symptoms**: Picks up key but fails to use it
**Solutions**:
- [ ] Check 'toggle' action is being explored
- [ ] Add shaped rewards for approaching door with key
- [ ] Ensure frame stacking captures key pickup
- [ ] Verify action masking (if used)

#### Problem: Memory Issues (Large Frame Stacks)
**Symptoms**: OOM errors or slow training
**Solutions**:
- [ ] Reduce frame stack size from 4 to 2
- [ ] Use observation compression
- [ ] Switch to LSTM instead of frame stacking
- [ ] Reduce batch size and buffer size

### Mac-Specific Optimizations 🍎
- [ ] Use MPS for CNN training: `device="mps"`
- [ ] Reduce parallel environments if memory limited: `num_envs=4`
- [ ] Use mixed precision training for CNNs
- [ ] Profile memory usage during frame stacking

## Phase A.10 - Analysis & Transition

### 📋 Final Analysis Tasks

#### 1. Performance Documentation
- [ ] Record final success rates on both 5x5 and 8x8
- [ ] Document sample efficiency (timesteps to reach targets)
- [ ] Compare one-hot vs CNN observation processing
- [ ] Analyze curriculum vs single-environment training

#### 2. Learned Behavior Analysis
- [ ] Verify agent learns correct sequence: key → door → goal
- [ ] Check exploration patterns (visits all areas)
- [ ] Measure policy consistency across runs
- [ ] Document failure modes and their frequencies

#### 3. Preparation for Atari
- [ ] Compare CNN architectures (what worked for MiniGrid)
- [ ] Document frame stacking benefits
- [ ] Record exploration strategies that worked
- [ ] Note preprocessing techniques that helped

### Expected Insights 🧠
- **Exploration** is crucial for multi-step environments
- **Frame stacking** or memory helps with temporal dependencies
- **Curriculum learning** accelerates learning on complex tasks
- **CNN architectures** can process structured visual inputs effectively

## Quick Start Commands 🚀

### 1. Environment Test
```bash
# Test MiniGrid installation and basic functionality
python -c "import minigrid; import gymnasium as gym; env = gym.make('MiniGrid-DoorKey-5x5-v0'); print(env.reset()); print(env.step(1))"
```

### 2. Observation Processing Test
```bash
# Test observation wrappers
python scripts/test_wrappers.py --env MiniGrid-DoorKey-5x5-v0 --wrapper onehot
```

### 3. Training Tests
```bash
# Quick functionality test
python scripts/train.py --config configs/experiments/ppo_doorkey.yaml --debug --total-timesteps 5000

# 5x5 training
python scripts/train.py --config configs/experiments/ppo_doorkey_5x5.yaml

# 8x8 transfer learning  
python scripts/train.py --config configs/experiments/ppo_doorkey_8x8.yaml --load-checkpoint experiments/doorkey_5x5/checkpoints/latest.pt
```

### 4. Curriculum Training
```bash
# Automated curriculum
python scripts/train.py --config configs/experiments/ppo_doorkey_curriculum.yaml
```

## Success Checklist ✅

Your MiniGrid DoorKey PPO is complete when:
- [ ] Achieves ≥95% success rate on 5x5 within 300k timesteps
- [ ] Achieves >70% success rate on 8x8 within total budget
- [ ] Demonstrates proper exploration and task understanding
- [ ] Shows benefit from frame stacking or memory mechanisms
- [ ] Ready for pixel-based environments (Atari next!)

## What You've Learned 🎓

By completing this homework, you've mastered:
- **Partial Observability**: Handling limited visual information
- **Visual Processing**: CNNs and one-hot encoding for structured observations  
- **Exploration Strategies**: Entropy bonuses and curiosity-driven learning
- **Curriculum Learning**: Progressive difficulty for complex tasks
- **Memory & Temporal Dependencies**: Frame stacking and sequential decision making

**Next Up**: Atari Pong - Real pixel environments with deep CNNs and preprocessing pipelines!