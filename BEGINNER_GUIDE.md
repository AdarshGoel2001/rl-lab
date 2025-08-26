# Complete Beginner's Guide to the RL Codebase

## Table of Contents
1. [What Is This Codebase?](#what-is-this-codebase)
2. [Why Does Each Component Exist?](#why-does-each-component-exist)
3. [Understanding Data Flow](#understanding-data-flow)
4. [Function Language Guide](#function-language-guide)
5. [Component Deep Dive](#component-deep-dive)
6. [How Everything Connects](#how-everything-connects)
7. [Step-by-Step Execution Flow](#step-by-step-execution-flow)
8. [Common Programming Patterns Used](#common-programming-patterns-used)

---

## What Is This Codebase?

Imagine you're trying to teach a computer to play a video game. The computer needs to:
1. **See** what's happening on screen (observations)
2. **Decide** what button to press (actions)
3. **Learn** from what happens after pressing buttons (rewards)
4. **Get better** over time by remembering what worked

This is called **Reinforcement Learning (RL)**. This codebase is like a factory that builds different types of "AI students" that can learn to play games or solve problems through trial and error.

### Why Do We Need All These Files?

Think of building a car. You don't want to rebuild the engine, wheels, and steering wheel every time you make a new car. Instead, you create **reusable parts** that can be mixed and matched. This codebase does the same thing for AI:

- **Algorithms** = The "brain" that decides what to do
- **Environments** = The "world" where the AI practices
- **Networks** = The "memory" that stores what the AI learned
- **Training Loop** = The "teacher" that runs practice sessions

---

## Why Does Each Component Exist?

### 1. **Base Classes** (The Templates)
**Location**: `src/algorithms/base.py`, `src/environments/base.py`, etc.

**Why they exist**: Imagine you're running a restaurant. Every chef needs to know how to:
- Take an order (input)
- Cook food (processing)
- Serve the dish (output)

Base classes are like "job descriptions" - they define what every algorithm or environment MUST be able to do, even if they do it differently.

```python
class BaseAlgorithm:
    """This is like a job description for all AI algorithms"""
    
    def act(self, observation):
        """Every algorithm must be able to decide what to do when it sees something"""
        pass  # "You figure out HOW, but you MUST do this"
    
    def update(self, batch):
        """Every algorithm must be able to learn from experience"""
        pass  # "You figure out HOW, but you MUST do this"
```

**What they accept**: Different algorithms accept different things, but all must follow this pattern
**What they return**: Different algorithms return different things, but in predictable formats

### 2. **Registry System** (The Phone Book)
**Location**: `src/utils/registry.py`

**Why it exists**: Imagine you have 50 different AI algorithms. Without a registry, you'd need to manually import and create each one:

```python
# BAD: Manual way
if algorithm_name == "ppo":
    from algorithms.ppo import PPO
    algorithm = PPO(config)
elif algorithm_name == "sac":
    from algorithms.sac import SAC  
    algorithm = SAC(config)
# ... 48 more elif statements!
```

The registry is like a phone book - you give it a name, it gives you the right algorithm:

```python
# GOOD: Registry way
algorithm = ALGORITHM_REGISTRY[algorithm_name](config)
```

**How it works**:
```python
# Step 1: Algorithms register themselves when the file is loaded
@register_algorithm("ppo")  # This adds PPO to the phone book
class PPO(BaseAlgorithm):
    pass

# Step 2: Later, you can look up any algorithm by name
ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "sac": SAC,
    "dqn": DQN
}
```

**What it accepts**: A string name (like "ppo")
**What it returns**: The actual class that can create that type of algorithm

### 3. **Configuration System** (The Recipe Book)
**Location**: `configs/` directory with YAML files

**Why it exists**: Imagine you're baking. Instead of remembering "2 cups flour, 1 cup sugar..." every time, you write it down in a recipe. Configs are recipes for AI experiments.

```yaml
# This is like a recipe card
algorithm:
  name: "ppo"           # What type of AI brain to use
  lr: 0.001            # How fast it should learn
  gamma: 0.99          # How much it cares about future rewards

environment:
  name: "CartPole-v1"  # What game/problem to solve
  
training:
  total_timesteps: 100000  # How long to practice
```

**Why YAML instead of Python**: YAML is human-readable and can be changed without touching code. Non-programmers can modify experiments.

**What configs accept**: Key-value pairs defining experiment parameters
**What configs return**: A structured dictionary that components can read

### 4. **Environment Wrappers** (The Translator)
**Location**: `src/environments/`

**Why they exist**: Different games have different interfaces:
- Some games give you RGB images, others give you numbers
- Some games have discrete actions (press A or B), others have continuous actions (steering wheel position)
- Some games end after 1000 steps, others never end

Environment wrappers are like translators that make all games "speak the same language":

```python
class BaseEnvironment:
    def reset(self):
        """Start a new game/episode"""
        return observation  # What the AI sees at the start
    
    def step(self, action):
        """The AI takes an action, world responds"""
        return observation, reward, done, info
        # observation: What AI sees after the action
        # reward: How good/bad the action was (number)
        # done: Is the game over? (True/False)
        # info: Extra debug information (dictionary)
```

**What they accept**: Actions from the AI (could be button presses, movement commands, etc.)
**What they return**: Always the same 4-tuple format, regardless of the underlying game

### 5. **Neural Networks** (The Memory System)
**Location**: `src/networks/`

**Why they exist**: AI needs to remember patterns. Neural networks are like the AI's brain that stores memories about what actions work in different situations.

```python
class MLP(BaseNetwork):
    """Multi-Layer Perceptron - like a simple brain with layers"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        # input_size: How many numbers describe what the AI sees
        # hidden_sizes: How many "thinking layers" (like [64, 64])
        # output_size: How many possible actions
        pass
    
    def forward(self, observation):
        """Think about what to do given what you see"""
        # Takes: observation (what AI sees, like game screen data)
        # Returns: action_probabilities (which actions seem good)
        pass
```

**What they accept**: Observations (numerical representations of what the AI sees)
**What they return**: Action predictions or value estimates (numbers indicating what to do)

### 6. **Experience Buffers** (The Memory Storage)
**Location**: `src/buffers/`

**Why they exist**: Humans learn by remembering experiences. If you touch a hot stove and get burned, you remember not to do it again. AI needs the same memory system.

```python
class UniformBuffer:
    """Stores memories and randomly samples them for learning"""
    
    def add(self, observation, action, reward, next_observation, done):
        """Store a memory: I saw X, did Y, got reward Z"""
        pass
    
    def sample(self, batch_size):
        """Give me some random memories to learn from"""
        # Returns: A batch of experiences for the AI to study
        pass
```

**What they accept**: Experience tuples (observation, action, reward, next_observation, done)
**What they return**: Batches of experiences for learning

### 7. **Training Loop** (The Teacher)
**Location**: `src/core/trainer.py`

**Why it exists**: Like a driving instructor who:
1. Sets up the practice course
2. Watches the student drive
3. Gives feedback after each attempt
4. Keeps track of progress
5. Saves progress so you can continue later

```python
class Trainer:
    def __init__(self, config_path):
        """Set up the practice session"""
        # Reads the recipe (config)
        # Creates the student (algorithm)
        # Sets up the practice area (environment)
        pass
    
    def train(self):
        """Run the practice session"""
        while not_done_learning:
            # 1. Student practices (collects experience)
            # 2. Student studies (learns from experience) 
            # 3. Teacher evaluates (runs tests)
            # 4. Teacher saves progress (checkpoints)
            pass
```

**What it accepts**: A configuration file path
**What it returns**: A trained AI model and performance metrics

---

## Understanding Data Flow

Think of data flow like water flowing through pipes. Each component is a pipe section that transforms the water (data) in a specific way:

### 1. **Configuration Flow** (The Setup Phase)
```
YAML Config File → Config Loader → Component Creator → Ready System
```

**Step by step**:
1. **YAML file**: Human-readable recipe sitting on disk
2. **Config Loader**: Reads YAML, converts to Python dictionary
3. **Component Creator**: Uses registry to build actual components
4. **Ready System**: All components connected and ready to use

### 2. **Training Data Flow** (The Learning Phase)
```
Environment → Observation → Algorithm → Action → Environment → Reward
     ↑                                                          ↓
     ← Experience Buffer ← Experience Tuple ←──────────────────┘
                ↓
         Random Sample → Algorithm → Learning Update
```

**Step by step**:
1. **Environment** creates an **observation** (like a game screenshot converted to numbers)
2. **Algorithm** receives observation, thinks, produces an **action** (like "move left")
3. **Environment** receives action, simulates physics, returns **reward** and new **observation**
4. This experience gets stored in **Experience Buffer** as a memory
5. **Algorithm** randomly samples old memories from buffer to learn patterns
6. **Algorithm** updates its neural networks based on what worked/failed

### 3. **Checkpoint Flow** (The Save System)
```
Training State → Checkpoint Manager → Disk Storage
      ↑                                    ↓
Resume Point ← Checkpoint Loader ←─────────┘
```

**Why checkpoints**: If your computer crashes after 3 hours of training, you don't want to start over! Checkpoints are like save files in video games.

---

## Function Language Guide

### Input/Output Patterns

#### **Observations** (What AI Sees)
```python
# Simple environments give arrays of numbers
observation = [0.1, -0.5, 0.3, 0.8]  # Position, velocity, angle, etc.

# Complex environments give images
observation = np.array([[[255, 0, 0], [0, 255, 0]], ...])  # RGB pixel values

# What it means: These numbers represent the current "state of the world"
```

#### **Actions** (What AI Does)
```python
# Discrete actions (like button presses)
action = 0  # Could mean "move left"
action = 1  # Could mean "move right"

# Continuous actions (like steering wheel)
action = [0.5, -0.2]  # [steering_angle, acceleration]

# What it means: These numbers tell the environment what the AI wants to do
```

#### **Rewards** (Feedback Signal)
```python
reward = 1.0   # Good job!
reward = 0.0   # Neutral
reward = -1.0  # Bad move!

# What it means: This number tells the AI if its last action was good or bad
```

#### **Experience Tuples** (Memory Format)
```python
experience = {
    'observation': [0.1, 0.2, 0.3],    # What I saw
    'action': 1,                        # What I did  
    'reward': 0.5,                      # What I got
    'next_observation': [0.2, 0.3, 0.4], # What I saw after
    'done': False                       # Is episode over?
}
```

#### **Batches** (Learning Data)
```python
batch = {
    'observations': [[0.1, 0.2], [0.3, 0.4], ...],    # Array of what AI saw
    'actions': [1, 0, 1, ...],                         # Array of what AI did
    'rewards': [0.5, -0.1, 1.0, ...],                 # Array of feedback
    'next_observations': [[0.2, 0.3], [0.4, 0.5], ...], # What happened after
    'dones': [False, False, True, ...]                 # Which episodes ended
}

# Why batches: Neural networks learn faster with multiple examples at once
```

### Function Signatures Explained

#### **Algorithm Functions**
```python
def act(self, observation, deterministic=False):
    """
    INPUT:
      observation: Current state of environment (array of numbers)
      deterministic: Should AI be consistent (True) or explore randomly (False)?
    
    PROCESS:
      - Neural network processes observation
      - Decides which action has highest probability of success
      - Adds randomness if deterministic=False (for exploration)
    
    OUTPUT:
      action: What the AI wants to do (number or array of numbers)
    """
    
def update(self, batch):
    """
    INPUT:
      batch: Dictionary containing arrays of past experiences
    
    PROCESS:
      - Computes how wrong the AI's predictions were
      - Calculates gradients (which direction to adjust network weights)
      - Updates neural network weights to be more accurate
    
    OUTPUT:
      metrics: Dictionary of learning statistics (losses, accuracies, etc.)
    """
```

#### **Environment Functions**
```python
def reset(self, seed=None):
    """
    INPUT:
      seed: Random number seed for reproducible experiments (optional)
    
    PROCESS:
      - Resets game/simulation to starting state
      - Randomizes initial conditions (if appropriate)
      - Prepares first observation
    
    OUTPUT:
      observation: Initial state that AI will see
    """

def step(self, action):
    """
    INPUT:
      action: What the AI wants to do (number or array)
    
    PROCESS:
      - Applies action to simulation (move character, adjust control, etc.)
      - Simulates physics/game rules for one time step
      - Calculates reward based on what happened
      - Determines if episode should end
    
    OUTPUT:
      observation: New state after action
      reward: Numerical feedback about the action
      done: Boolean indicating if episode is over
      info: Extra debugging information (dictionary)
    """
```

#### **Buffer Functions**
```python
def add(self, observation, action, reward, next_observation, done):
    """
    INPUT:
      All the components of one experience step
    
    PROCESS:
      - Packages inputs into experience tuple
      - Adds to internal storage (list, array, or database)
      - May remove old experiences if buffer is full
    
    OUTPUT:
      None (just stores the experience)
    """

def sample(self, batch_size):
    """
    INPUT:
      batch_size: How many experiences to retrieve (like 64 or 256)
    
    PROCESS:
      - Randomly selects batch_size experiences from storage
      - Converts to arrays that neural networks can process
      - May apply transformations (normalization, etc.)
    
    OUTPUT:
      batch: Dictionary with arrays of experiences
    """
```

---

## Component Deep Dive

### The Registry Pattern (Phone Book System)

**Problem it solves**: How do you create objects from string names without huge if/else chains?

```python
# Without registry (BAD)
def create_algorithm(name, config):
    if name == "ppo":
        from algorithms.ppo import PPO
        return PPO(config)
    elif name == "sac":
        from algorithms.sac import SAC
        return SAC(config)
    # ... 50 more algorithms!

# With registry (GOOD)
ALGORITHM_REGISTRY = {}

def register_algorithm(name):
    """This is a decorator - it runs when Python loads the file"""
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls  # Add to phone book
        return cls
    return decorator

@register_algorithm("ppo")  # Automatically adds PPO to registry
class PPO(BaseAlgorithm):
    pass

def create_algorithm(name, config):
    return ALGORITHM_REGISTRY[name](config)  # Look up in phone book
```

**How the decorator works**:
1. When Python loads `algorithms/ppo.py`, it sees `@register_algorithm("ppo")`
2. This runs `register_algorithm("ppo")` which returns a decorator function
3. The decorator adds PPO to `ALGORITHM_REGISTRY["ppo"] = PPO`
4. Later, `ALGORITHM_REGISTRY["ppo"]` gives you the PPO class

### The Configuration System (Recipe Management)

**Problem it solves**: How do you run experiments with different settings without changing code?

```python
# config.yaml
algorithm:
  name: "ppo"
  learning_rate: 0.001
  
environment:
  name: "CartPole-v1"
  
training:
  total_timesteps: 100000
```

```python
# config_loader.py
import yaml

def load_config(path):
    """
    INPUT: Path to YAML file
    PROCESS: 
      - Opens file
      - Parses YAML syntax into Python dictionary
      - Validates that required fields exist
      - Sets default values for optional fields
    OUTPUT: Nested dictionary with all experiment settings
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validation
    assert 'algorithm' in config, "Must specify algorithm"
    assert 'environment' in config, "Must specify environment"
    
    # Defaults
    if 'training' not in config:
        config['training'] = {'total_timesteps': 100000}
    
    return config
```

**Why YAML**: 
- Human readable: `learning_rate: 0.001` vs `{"learning_rate": 0.001}`
- Supports comments: `# This controls how fast the AI learns`
- No syntax errors: YAML parser catches mistakes
- Version controllable: Git can track changes to experiment settings

### The Base Class System (Job Descriptions)

**Problem it solves**: How do you ensure all algorithms work with the same training loop?

```python
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):  # ABC = Abstract Base Class
    """This is like a job description that all algorithms must follow"""
    
    def __init__(self, config):
        """Every algorithm gets a config dictionary"""
        self.config = config
        
    @abstractmethod  # This decorator means "you MUST implement this"
    def act(self, observation, deterministic=False):
        """Every algorithm must be able to choose actions"""
        pass  # No implementation - subclasses must provide this
    
    @abstractmethod
    def update(self, batch):
        """Every algorithm must be able to learn from experience"""
        pass
    
    # Non-abstract method - provides default implementation
    def save_checkpoint(self, path):
        """Save algorithm state to disk"""
        checkpoint = {
            'config': self.config,
            'networks': self.get_network_states(),
            'step': self.step
        }
        torch.save(checkpoint, path)
```

```python
class PPO(BaseAlgorithm):
    """Concrete implementation of the algorithm interface"""
    
    def act(self, observation, deterministic=False):
        # MUST implement this because it's @abstractmethod in base class
        with torch.no_grad():
            action_probs = self.policy_network(observation)
            if deterministic:
                action = action_probs.argmax()
            else:
                action = torch.multinomial(action_probs, 1)
        return action.item()
    
    def update(self, batch):
        # MUST implement this because it's @abstractmethod in base class
        policy_loss = self.compute_policy_loss(batch)
        value_loss = self.compute_value_loss(batch)
        
        total_loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item()
        }
```

**Why inheritance**: The training loop can call `algorithm.act()` and `algorithm.update()` without knowing if it's PPO, SAC, or DQN. They all follow the same interface.

### The Checkpoint System (Save Game)

**Problem it solves**: How do you save and resume training without losing progress?

```python
class CheckpointManager:
    def __init__(self, experiment_dir):
        self.checkpoint_dir = experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save(self, trainer_state, name="checkpoint"):
        """
        INPUT: trainer_state - Dictionary containing everything needed to resume
        PROCESS:
          - Collects state from all components (algorithm, buffer, etc.)
          - Saves random number generator states (for reproducibility)
          - Saves to disk with timestamp
          - Creates 'latest.pt' symlink for easy loading
        OUTPUT: None (saves to disk)
        """
        checkpoint = {
            # Algorithm state (neural network weights, optimizer state)
            'algorithm': trainer_state['algorithm'].save_checkpoint(),
            
            # Experience buffer state (all stored experiences)
            'buffer': trainer_state['buffer'].save_checkpoint(),
            
            # Training progress
            'global_step': trainer_state['global_step'],
            'episode': trainer_state['episode'],
            
            # Random states (crucial for reproducibility!)
            'rng_states': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            
            # Metadata
            'config': trainer_state['config'],
            'timestamp': time.time()
        }
        
        # Save with unique name
        step = trainer_state['global_step']
        path = self.checkpoint_dir / f"{name}_{step}.pt"
        torch.save(checkpoint, path)
        
        # Update latest symlink
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(path)
    
    def load_latest(self):
        """
        INPUT: None
        PROCESS:
          - Checks if 'latest.pt' symlink exists
          - Loads checkpoint file
          - Validates checkpoint format
        OUTPUT: Checkpoint dictionary or None if no checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            checkpoint = torch.load(latest_path)
            return checkpoint
        return None
    
    def restore_training_state(self, checkpoint, trainer):
        """
        INPUT: 
          checkpoint - Dictionary loaded from disk
          trainer - Trainer object to restore state to
        PROCESS:
          - Restores algorithm state (network weights, optimizer)
          - Restores buffer state (all experiences)
          - Restores random number generators
          - Restores training counters
        OUTPUT: None (modifies trainer object in-place)
        """
        # Restore algorithm
        trainer.algorithm.load_checkpoint(checkpoint['algorithm'])
        
        # Restore buffer
        trainer.buffer.load_checkpoint(checkpoint['buffer'])
        
        # Restore training progress
        trainer.global_step = checkpoint['global_step']
        trainer.episode = checkpoint['episode']
        
        # Restore random states (CRITICAL for reproducibility)
        np.random.set_state(checkpoint['rng_states']['numpy'])
        torch.set_rng_state(checkpoint['rng_states']['torch'])
        if checkpoint['rng_states']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
```

**Why random state matters**: If you resume training, you want the AI to behave exactly as if training never stopped. Random number generators affect action selection and data sampling.

---

## How Everything Connects

### The Main Training Loop (The Conductor)

Think of the trainer as a conductor orchestrating a symphony. Each component is an instrument that needs to play its part at the right time:

```python
class Trainer:
    def __init__(self, config_path):
        """Set up the orchestra"""
        # 1. Load the sheet music (config)
        self.config = load_config(config_path)
        
        # 2. Create experiment directory (concert hall)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path("experiments") / f"{self.config.experiment.name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Hire the musicians (create components)
        self.env = ENVIRONMENT_REGISTRY[self.config.environment.wrapper](
            self.config.environment
        )
        self.algorithm = ALGORITHM_REGISTRY[self.config.algorithm.name](
            self.config.algorithm
        )
        self.buffer = BUFFER_REGISTRY[self.config.buffer.type](
            self.config.buffer
        )
        
        # 4. Set up recording equipment (logging)
        self.logger = Logger(self.exp_dir, self.config.logging)
        
        # 5. Set up save system (checkpoints)
        self.checkpoint_manager = CheckpointManager(self.exp_dir)
        
        # 6. Initialize counters
        self.global_step = 0
        self.episode = 0
        
        # 7. Check if we're resuming a previous concert
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            self.restore_from_checkpoint(checkpoint)
            print(f"Resumed from step {self.global_step}")
    
    def train(self):
        """Conduct the performance"""
        target_steps = self.config.training.total_timesteps
        
        while self.global_step < target_steps:
            # 1. COLLECTION PHASE: AI practices in environment
            episode_data = self.collect_episode()
            
            # 2. STORAGE PHASE: Save what happened for learning
            for experience in episode_data:
                self.buffer.add(**experience)
            
            # 3. LEARNING PHASE: Study past experiences (if buffer has enough data)
            if self.buffer.ready():
                batch = self.buffer.sample(self.config.training.batch_size)
                metrics = self.algorithm.update(batch)
                self.logger.log(metrics, self.global_step, prefix="train/")
            
            # 4. EVALUATION PHASE: Test how well AI is doing
            if self.global_step % self.config.training.eval_frequency == 0:
                eval_metrics = self.evaluate()
                self.logger.log(eval_metrics, self.global_step, prefix="eval/")
            
            # 5. CHECKPOINT PHASE: Save progress
            if self.global_step % self.config.training.checkpoint_frequency == 0:
                self.checkpoint_manager.save(self.get_state())
            
            self.episode += 1
    
    def collect_episode(self):
        """Let the AI practice one complete game/episode"""
        episode_experiences = []
        observation = self.env.reset()
        done = False
        
        while not done:
            # AI decides what to do
            action = self.algorithm.act(observation)
            
            # World responds to AI's action
            next_observation, reward, done, info = self.env.step(action)
            
            # Remember this experience
            experience = {
                'observation': observation,
                'action': action,
                'reward': reward,
                'next_observation': next_observation,
                'done': done
            }
            episode_experiences.append(experience)
            
            # Move to next step
            observation = next_observation
            self.global_step += 1
        
        return episode_experiences
    
    def evaluate(self):
        """Test the AI without learning (like a final exam)"""
        eval_rewards = []
        
        for _ in range(self.config.training.num_eval_episodes):
            observation = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use deterministic=True for consistent evaluation
                action = self.algorithm.act(observation, deterministic=True)
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        return {
            'reward/mean': np.mean(eval_rewards),
            'reward/std': np.std(eval_rewards),
            'reward/max': np.max(eval_rewards),
            'reward/min': np.min(eval_rewards)
        }
```

### Data Transformation Pipeline

Understanding how data flows and transforms through the system:

```python
# 1. ENVIRONMENT OUTPUT (raw game state)
raw_observation = env.reset()
# Example: [x_position, velocity, angle, angular_velocity] = [0.1, 0.0, 0.05, 0.0]

# 2. ALGORITHM INPUT (processed observation)  
processed_obs = torch.tensor(raw_observation, dtype=torch.float32)
# Converts Python list to PyTorch tensor for neural network

# 3. NEURAL NETWORK COMPUTATION
with torch.no_grad():  # Don't compute gradients during action selection
    # Pass through network layers
    hidden = torch.relu(self.layer1(processed_obs))  # First layer + activation
    hidden = torch.relu(self.layer2(hidden))         # Second layer + activation  
    action_logits = self.output_layer(hidden)        # Final layer (no activation)
    
    # Convert to probabilities
    action_probs = torch.softmax(action_logits, dim=-1)
    # Example: [0.3, 0.7] means 30% chance action 0, 70% chance action 1

# 4. ACTION SAMPLING
if deterministic:
    action = action_probs.argmax()  # Choose best action
else:
    action = torch.multinomial(action_probs, 1)  # Sample based on probabilities

# 5. CONVERT BACK TO ENVIRONMENT FORMAT
action_int = action.item()  # Convert tensor to Python integer
# Example: 1 (meaning "move right" in CartPole)

# 6. ENVIRONMENT RESPONSE
next_obs, reward, done, info = env.step(action_int)
# Example: next_obs=[0.11, 0.02, 0.048, -0.01], reward=1.0, done=False

# 7. EXPERIENCE STORAGE
experience = {
    'observation': raw_observation,      # [0.1, 0.0, 0.05, 0.0]
    'action': action_int,               # 1
    'reward': reward,                   # 1.0  
    'next_observation': next_obs,       # [0.11, 0.02, 0.048, -0.01]
    'done': done                        # False
}
buffer.add(**experience)

# 8. BATCH CREATION (for learning)
batch = buffer.sample(64)  # Get 64 random experiences
# batch = {
#     'observations': [[0.1, 0.0, 0.05, 0.0], [0.2, 0.1, 0.03, 0.02], ...],  # 64 observations
#     'actions': [1, 0, 1, ...],                                               # 64 actions
#     'rewards': [1.0, 1.0, 0.0, ...],                                        # 64 rewards
#     'next_observations': [[0.11, 0.02, 0.048, -0.01], ...],                # 64 next observations
#     'dones': [False, False, True, ...]                                      # 64 done flags
# }

# 9. LEARNING (gradient computation)
observations = torch.tensor(batch['observations'])     # Shape: [64, 4]
actions = torch.tensor(batch['actions'])              # Shape: [64]
rewards = torch.tensor(batch['rewards'])              # Shape: [64]

# Compute current policy predictions
current_action_probs = policy_network(observations)   # Shape: [64, 2]
selected_action_probs = current_action_probs.gather(1, actions.unsqueeze(1))  # Shape: [64, 1]

# Compute loss (how wrong were our predictions?)
loss = -torch.log(selected_action_probs) * rewards   # Negative log likelihood weighted by rewards

# Update network weights
optimizer.zero_grad()  # Clear previous gradients
loss.mean().backward()  # Compute gradients
optimizer.step()       # Update weights
```

---

## Step-by-Step Execution Flow

### Startup Sequence

1. **User runs command**: `python scripts/train.py --config configs/experiments/ppo_cartpole.yaml`

2. **Config Loading**:
   ```python
   config = load_config("configs/experiments/ppo_cartpole.yaml")
   # Reads YAML file, validates structure, sets defaults
   ```

3. **Component Registration Discovery**:
   ```python
   # Python imports all files in src/algorithms/, src/environments/, etc.
   # This triggers @register_algorithm decorators, populating registries
   ALGORITHM_REGISTRY = {"ppo": PPO, "sac": SAC, ...}
   ENVIRONMENT_REGISTRY = {"gym": GymWrapper, "dm_control": DMControlWrapper, ...}
   ```

4. **Experiment Directory Creation**:
   ```python
   exp_dir = Path("experiments/ppo_cartpole_20241201_143022")
   exp_dir.mkdir(parents=True)
   # Creates subdirectories: checkpoints/, logs/, videos/, configs/
   ```

5. **Component Instantiation**:
   ```python
   env = ENVIRONMENT_REGISTRY["gym"](config.environment)
   algorithm = ALGORITHM_REGISTRY["ppo"](config.algorithm)  
   buffer = BUFFER_REGISTRY["uniform"](config.buffer)
   ```

6. **Checkpoint Check**:
   ```python
   checkpoint = checkpoint_manager.load_latest()
   if checkpoint:
       restore_from_checkpoint(checkpoint)
   ```

### Training Loop Execution

**Outer Loop**: Episodes (complete games/tasks)
```python
for episode in range(max_episodes):
    episode_experiences = []
    observation = env.reset()  # Start new game
    done = False
    
    # Inner Loop: Steps within one episode
    while not done:
        action = algorithm.act(observation)
        next_obs, reward, done, info = env.step(action)
        
        # Store experience
        experience = (observation, action, reward, next_obs, done)
        episode_experiences.append(experience)
        buffer.add(*experience)
        
        observation = next_obs
        global_step += 1
        
        # Learning update (may happen multiple times per episode)
        if buffer.ready() and global_step % update_frequency == 0:
            batch = buffer.sample(batch_size)
            metrics = algorithm.update(batch)
            logger.log(metrics, global_step)
    
    # End-of-episode tasks
    if episode % eval_frequency == 0:
        eval_metrics = evaluate()
        logger.log(eval_metrics, global_step, prefix="eval/")
    
    if episode % checkpoint_frequency == 0:
        checkpoint_manager.save(get_state())
```

### Learning Update Detail

When `algorithm.update(batch)` is called:

1. **Forward Pass**: 
   ```python
   # Compute what the AI currently thinks about the batch of experiences
   current_values = value_network(batch['observations'])
   current_action_probs = policy_network(batch['observations'])
   ```

2. **Loss Computation**:
   ```python
   # How wrong were our predictions?
   value_loss = (current_values - batch['returns']).pow(2).mean()
   policy_loss = compute_policy_loss(current_action_probs, batch['actions'], batch['advantages'])
   total_loss = value_loss + policy_loss
   ```

3. **Backward Pass**:
   ```python
   optimizer.zero_grad()        # Clear old gradients
   total_loss.backward()        # Compute new gradients
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)  # Prevent exploding gradients
   optimizer.step()             # Update network weights
   ```

4. **Metrics Collection**:
   ```python
   return {
       'loss/value': value_loss.item(),
       'loss/policy': policy_loss.item(), 
       'grad_norm': compute_grad_norm(),
       'explained_variance': compute_explained_variance()
   }
   ```

---

## Common Programming Patterns Used

### 1. **Factory Pattern** (Component Creation)
**Problem**: Need to create different types of objects based on string names
**Solution**: Registry system that maps names to classes

```python
# Instead of giant if/else chains
def create_algorithm(name, config):
    return ALGORITHM_REGISTRY[name](config)

# Usage
ppo = create_algorithm("ppo", config)
sac = create_algorithm("sac", config)
```

### 2. **Template Method Pattern** (Base Classes)  
**Problem**: Different algorithms have different implementations but same overall structure
**Solution**: Abstract base class defines the interface, subclasses implement details

```python
class BaseAlgorithm:
    def train_step(self):  # Template method - same for all algorithms
        batch = self.buffer.sample()
        metrics = self.update(batch)  # Subclass implements this
        return metrics
    
    @abstractmethod
    def update(self, batch):  # Each algorithm implements differently
        pass
```

### 3. **Observer Pattern** (Logging)
**Problem**: Multiple components need to report metrics without tight coupling
**Solution**: Logger that multiple components can send data to

```python
class Logger:
    def __init__(self):
        self.subscribers = []  # List of places to send data
    
    def log(self, metrics, step):
        for subscriber in self.subscribers:
            subscriber.log(metrics, step)

# Usage
logger.add_subscriber(TensorBoardLogger())
logger.add_subscriber(WandBLogger())
logger.log({'reward': 10.5}, step=1000)  # Goes to both TensorBoard and WandB
```

### 4. **Strategy Pattern** (Interchangeable Components)
**Problem**: Want to swap different implementations of same functionality
**Solution**: Common interface with multiple implementations

```python
# All buffers implement same interface but work differently
class UniformBuffer:
    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)

class PrioritizedBuffer:  
    def sample(self, batch_size):
        return self.sample_by_priority(batch_size)  # Different sampling strategy
```

### 5. **Decorator Pattern** (Registration)
**Problem**: Need to automatically register classes when files are loaded
**Solution**: Decorators that run code when class is defined

```python
def register_algorithm(name):
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls  # This runs when Python loads the file
        return cls
    return decorator

@register_algorithm("ppo")  # Automatically registers PPO when file is imported
class PPO(BaseAlgorithm):
    pass
```

### 6. **Context Manager Pattern** (Resource Management)
**Problem**: Need to ensure cleanup happens even if errors occur
**Solution**: `with` statements and context managers

```python
class GradientContext:
    def __enter__(self):
        torch.set_grad_enabled(True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(False)

# Usage
with GradientContext():
    loss.backward()  # Gradients enabled
# Gradients automatically disabled after with block
```

### 7. **Builder Pattern** (Complex Object Construction)
**Problem**: Creating objects with many optional parameters
**Solution**: Configuration objects that build components step by step

```python
class AlgorithmBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_networks(self):
        self.policy_net = create_network(self.config.policy_network)
        self.value_net = create_network(self.config.value_network)
        return self
    
    def build_optimizers(self):
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters())
        return self
    
    def build(self):
        return PPO(self.policy_net, self.value_net, self.policy_optimizer, self.value_optimizer)

# Usage
algorithm = AlgorithmBuilder(config).build_networks().build_optimizers().build()
```

---

## Key Takeaways for Beginners

### 1. **Everything is Data Transformation**
- **Environment**: Game state → Numbers (observations)
- **Algorithm**: Numbers → Decision (action)  
- **Learning**: Past decisions → Better future decisions
- **Logging**: Training metrics → Human-readable plots

### 2. **Interfaces Enable Modularity**
- Base classes define "what" must happen
- Concrete classes define "how" it happens
- Training loop works with any algorithm that follows the interface

### 3. **Configuration Drives Behavior**
- Code defines the capabilities
- Configs define which capabilities to use and how
- Same code can run completely different experiments

### 4. **State Management is Critical**
- Checkpoints save everything needed to resume
- Random states ensure reproducibility
- Metrics track progress over time

### 5. **Error Handling Happens at Boundaries**
- Config validation catches setup errors early
- Network forward passes check for NaN/Inf values
- Checkpoint loading validates file format

This codebase is designed to let you focus on the interesting research questions (what makes a good learning algorithm?) while handling all the infrastructure automatically (where do I save results? how do I compare experiments? how do I resume training?).