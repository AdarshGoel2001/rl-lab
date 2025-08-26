# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Modular RL Mono-Repo: Complete Architecture & Implementation Plan

## Executive Summary
A highly modular reinforcement learning mono-repo designed for researchers to easily implement algorithms, run experiments, and reproduce results across different environments, function approximators, and dynamics models.

## Core Design Principles
1. **Plug-and-Play Components**: Every component (algorithm, environment, network, logger) should be swappable
2. **Configuration-Driven**: All experiments defined via YAML/JSON configs, no code changes needed
3. **Reproducibility First**: Automatic seed management, versioning, and experiment tracking
4. **Researcher-Friendly**: Focus on implementing algorithms, not infrastructure
5. **Resume-Anywhere**: Seamless checkpoint system for multi-day training runs

## Project Structure
```
rl-monorepo/
├── configs/                    # Experiment configurations
│   ├── algorithms/            # Algorithm-specific configs
│   ├── environments/          # Environment configs
│   ├── networks/             # Network architecture configs
│   ├── experiments/          # Full experiment configs
│   └── sweeps/              # Hyperparameter sweep configs
├── src/
│   ├── algorithms/           # RL algorithm implementations
│   │   ├── base.py          # Abstract base algorithm
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── dqn.py
│   │   └── world_models/
│   │       ├── dreamer.py
│   │       └── planet.py
│   ├── environments/        # Environment wrappers
│   │   ├── base.py         # Abstract base environment
│   │   ├── gym_wrapper.py
│   │   ├── dm_control_wrapper.py
│   │   └── atari_wrapper.py
│   ├── networks/           # Neural network architectures
│   │   ├── base.py        # Abstract base network
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── rnn.py
│   │   └── transformers.py
│   ├── dynamics/          # World model dynamics
│   │   ├── base.py       # Abstract dynamics model
│   │   ├── deterministic.py
│   │   └── stochastic.py
│   ├── buffers/          # Experience replay buffers
│   │   ├── base.py
│   │   ├── uniform.py
│   │   ├── prioritized.py
│   │   └── trajectory.py
│   ├── utils/
│   │   ├── checkpoint.py  # Checkpoint management
│   │   ├── logger.py     # Logging utilities
│   │   ├── metrics.py    # Metric tracking
│   │   ├── seeding.py    # Reproducibility utilities
│   │   └── registry.py   # Component registration
│   └── core/
│       ├── trainer.py    # Main training loop
│       ├── evaluator.py  # Evaluation logic
│       └── experiment.py # Experiment orchestration
├── scripts/
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   ├── sweep.py         # Hyperparameter sweep script
│   └── analyze.py       # Result analysis script
├── experiments/          # Experiment outputs
│   └── {exp_name}_{timestamp}/
│       ├── checkpoints/
│       ├── logs/
│       ├── videos/
│       ├── configs/
│       └── metrics.json
└── tests/               # Unit tests

```

## Component APIs and Interfaces

### 1. Base Algorithm Interface
```python
class BaseAlgorithm:
    """All algorithms must implement this interface"""
    
    def __init__(self, config):
        self.config = config
        self.networks = {}  # Dictionary of networks
        self.optimizers = {}  # Dictionary of optimizers
        self.step = 0
        
    def act(self, observation, deterministic=False):
        """Select action given observation"""
        pass
        
    def update(self, batch):
        """Update algorithm with batch of experiences"""
        pass
        
    def save_checkpoint(self, path):
        """Save all algorithm state"""
        checkpoint = {
            'networks': {k: v.state_dict() for k, v in self.networks.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'step': self.step,
            'config': self.config
        }
        return checkpoint
        
    def load_checkpoint(self, checkpoint):
        """Load algorithm state from checkpoint"""
        pass
        
    def get_metrics(self):
        """Return current training metrics"""
        pass
```

### 2. Environment Wrapper Interface
```python
class BaseEnvironment:
    """Unified interface for all environments"""
    
    def __init__(self, config):
        self.config = config
        
    def reset(self, seed=None):
        """Reset environment, return initial observation"""
        pass
        
    def step(self, action):
        """Execute action, return (obs, reward, done, info)"""
        pass
        
    @property
    def observation_space(self):
        """Return observation space specification"""
        pass
        
    @property
    def action_space(self):
        """Return action space specification"""
        pass
```

### 3. Network Registry System
```python
# Registry for automatic component discovery
NETWORK_REGISTRY = {}
ALGORITHM_REGISTRY = {}
ENVIRONMENT_REGISTRY = {}

def register_network(name):
    def decorator(cls):
        NETWORK_REGISTRY[name] = cls
        return cls
    return decorator

# Usage example:
@register_network("mlp")
class MLP(BaseNetwork):
    pass
```

## Configuration System

### Master Experiment Config (YAML)
```yaml
# configs/experiments/ppo_cartpole.yaml
experiment:
  name: "ppo_cartpole_baseline"
  seed: 42
  device: "cuda"
  
algorithm:
  name: "ppo"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  
environment:
  name: "CartPole-v1"
  wrapper: "gym"
  normalize_obs: true
  normalize_reward: true
  
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
  size: 2048
  
training:
  total_timesteps: 1000000
  eval_frequency: 10000
  checkpoint_frequency: 50000
  num_eval_episodes: 10
  
logging:
  terminal: true
  tensorboard: true
  wandb:
    enabled: true
    project: "rl-experiments"
    tags: ["ppo", "cartpole"]
```

## Checkpoint & Resume System

### Checkpoint Manager
```python
class CheckpointManager:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = experiment_dir / "checkpoints"
        
    def save(self, trainer_state, name="checkpoint"):
        """Save complete training state"""
        checkpoint = {
            'algorithm': trainer_state['algorithm'].save_checkpoint(),
            'buffer': trainer_state['buffer'].save_checkpoint(),
            'global_step': trainer_state['global_step'],
            'episode': trainer_state['episode'],
            'metrics': trainer_state['metrics'],
            'rng_states': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
            }
        }
        path = self.checkpoint_dir / f"{name}_{global_step}.pt"
        torch.save(checkpoint, path)
        
        # Save latest symlink
        latest_link = self.checkpoint_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(path)
        
    def load_latest(self):
        """Load most recent checkpoint"""
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            return torch.load(latest)
        return None
        
    def auto_save(self, trainer_state, frequency=10000):
        """Automatically save at specified frequency"""
        if trainer_state['global_step'] % frequency == 0:
            self.save(trainer_state, name=f"auto_{trainer_state['global_step']}")
```

## Training Orchestration

### Main Trainer Class
```python
class Trainer:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.setup_experiment()
        
    def setup_experiment(self):
        # Create experiment directory
        self.exp_dir = Path("experiments") / f"{self.config.experiment.name}_{timestamp()}"
        
        # Initialize components from registry
        self.env = ENVIRONMENT_REGISTRY[self.config.environment.wrapper](self.config.environment)
        self.algorithm = ALGORITHM_REGISTRY[self.config.algorithm.name](self.config.algorithm)
        self.buffer = BUFFER_REGISTRY[self.config.buffer.type](self.config.buffer)
        
        # Setup logging
        self.logger = Logger(self.exp_dir, self.config.logging)
        self.checkpoint_manager = CheckpointManager(self.exp_dir)
        
        # Check for resume
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            self.resume_from_checkpoint(checkpoint)
            
    def train(self):
        """Main training loop with automatic checkpointing"""
        while self.global_step < self.config.training.total_timesteps:
            # Collect experience
            trajectory = self.collect_trajectory()
            self.buffer.add(trajectory)
            
            # Update algorithm
            if self.buffer.ready():
                metrics = self.algorithm.update(self.buffer.sample())
                self.logger.log(metrics, self.global_step)
                
            # Evaluation
            if self.global_step % self.config.training.eval_frequency == 0:
                eval_metrics = self.evaluate()
                self.logger.log(eval_metrics, self.global_step, prefix="eval/")
                
            # Checkpoint
            if self.global_step % self.config.training.checkpoint_frequency == 0:
                self.checkpoint_manager.save(self.get_state())
                
    def resume_from_checkpoint(self, checkpoint):
        """Resume training from checkpoint"""
        self.algorithm.load_checkpoint(checkpoint['algorithm'])
        self.buffer.load_checkpoint(checkpoint['buffer'])
        self.global_step = checkpoint['global_step']
        
        # Restore RNG states for reproducibility
        np.random.set_state(checkpoint['rng_states']['numpy'])
        torch.set_rng_state(checkpoint['rng_states']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
```

## Daily Workflow Solutions

### 1. Running Experiments
```bash
# Start new experiment
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml

# Resume interrupted experiment
python scripts/train.py --resume experiments/ppo_cartpole_2024/

# Run with different seeds
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --seed 1,2,3,4,5

# Quick test run
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --debug --total_timesteps 1000
```

### 2. Hyperparameter Sweeps
```yaml
# configs/sweeps/ppo_sweep.yaml
base_config: configs/experiments/ppo_cartpole.yaml
sweep:
  algorithm.lr: [1e-4, 3e-4, 1e-3]
  algorithm.clip_ratio: [0.1, 0.2, 0.3]
  network.actor.hidden_dims: [[64, 64], [128, 128], [256, 256]]
parallel_jobs: 4
```

```bash
python scripts/sweep.py --config configs/sweeps/ppo_sweep.yaml
```

### 3. Experiment Tracking & Analysis
```python
class ExperimentTracker:
    """Automatic experiment versioning and comparison"""
    
    def __init__(self):
        self.experiments_db = "experiments/registry.json"
        
    def register_experiment(self, config, exp_dir):
        """Register new experiment with metadata"""
        metadata = {
            'id': generate_id(),
            'name': config.experiment.name,
            'algorithm': config.algorithm.name,
            'environment': config.environment.name,
            'timestamp': timestamp(),
            'git_hash': get_git_hash(),
            'config_hash': hash_config(config),
            'directory': str(exp_dir)
        }
        
    def find_experiments(self, **filters):
        """Find experiments matching criteria"""
        # e.g., find_experiments(algorithm='ppo', environment='CartPole-v1')
        pass
        
    def compare_experiments(self, exp_ids):
        """Generate comparison plots/tables"""
        pass
```

### 4. Debugging & Monitoring
```python
class DebugMode:
    """Special mode for algorithm debugging"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.enable_gradient_checking = True
        self.log_network_weights = True
        self.check_nan_inf = True
        self.profile_performance = True
        
    def wrap_update(self, update_fn):
        def wrapped(*args, **kwargs):
            # Pre-update checks
            self.check_gradients()
            self.check_inputs(*args)
            
            # Profile update
            with torch.profiler.profile() as prof:
                result = update_fn(*args, **kwargs)
                
            # Post-update checks
            self.check_outputs(result)
            self.log_diagnostics()
            
            return result
        return wrapped
```

## Advanced Features

### 1. Distributed Training Support
```python
class DistributedTrainer(Trainer):
    """Extension for multi-GPU/multi-node training"""
    
    def setup_distributed(self):
        # Automatic detection and setup
        if torch.cuda.device_count() > 1:
            self.algorithm = nn.DataParallel(self.algorithm)
        # Or use DistributedDataParallel for multi-node
```

### 2. Automatic Hyperparameter Tuning
```python
class AutoTuner:
    """Bayesian optimization for hyperparameters"""
    
    def suggest_next_config(self, history):
        # Use optuna or similar
        pass
        
    def update_with_result(self, config, performance):
        pass
```

### 3. Experiment Reproducibility Verification
```python
class ReproducibilityChecker:
    """Verify experiments can be reproduced"""
    
    def verify_experiment(self, exp_dir):
        # Load original config and checkpoint
        # Re-run with same seed
        # Compare trajectories and metrics
        pass
```

## Common Pitfalls & Solutions

### 1. Memory Management
- **Problem**: OOM errors during long training runs
- **Solution**: Automatic buffer size adjustment, gradient accumulation, memory profiling

### 2. Numerical Instability
- **Problem**: NaN/Inf values crashing training
- **Solution**: Automatic detection, gradient clipping, safe math operations

### 3. Environment Compatibility
- **Problem**: Different environments have different interfaces
- **Solution**: Unified wrapper system with automatic space conversion

### 4. Experiment Organization
- **Problem**: Losing track of experiments
- **Solution**: Automatic naming, tagging, and searchable database

### 5. Config Management
- **Problem**: Config changes breaking old experiments
- **Solution**: Config versioning and migration system

## Implementation Priority Order

1. **Phase 1: Core Infrastructure** (Week 1)
   - Base classes for Algorithm, Environment, Network
   - Config loading system
   - Basic training loop

2. **Phase 2: Checkpoint System** (Week 2)
   - Save/load functionality
   - Automatic checkpointing
   - Resume capability

3. **Phase 3: First Algorithm** (Week 3)
   - Implement PPO as reference
   - Test on CartPole
   - Verify reproducibility

4. **Phase 4: Logging & Monitoring** (Week 4)
   - TensorBoard integration
   - Metric tracking
   - Experiment database

5. **Phase 5: Additional Algorithms** (Weeks 5-6)
   - SAC, DQN, etc.
   - World models (Dreamer, PlaNet)

6. **Phase 6: Advanced Features** (Weeks 7-8)
   - Hyperparameter sweeps
   - Distributed training
   - Auto-tuning

## Quick Start Guide for Beginners

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Your First Experiment
```bash
# Use pre-made config
python scripts/train.py --config configs/quickstart/ppo_cartpole.yaml
```

### 3. Implement Your Own Algorithm
```python
# src/algorithms/my_algorithm.py
from src.algorithms.base import BaseAlgorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(BaseAlgorithm):
    def act(self, observation):
        # Your implementation
        pass
        
    def update(self, batch):
        # Your implementation
        pass
```

### 4. Create Config for Your Algorithm
```yaml
# configs/algorithms/my_algorithm.yaml
algorithm:
  name: "my_algorithm"
  # Your hyperparameters
```

### 5. Train!
```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock environments and networks
- Verify checkpoint save/load preserves state exactly

### Integration Tests
- Test full training loop with small configs
- Verify reproducibility with fixed seeds
- Test resume functionality

### Performance Tests
- Benchmark training speed
- Memory usage monitoring
- Profiling bottlenecks

## Monitoring & Alerting

### Real-time Monitoring
```python
class TrainingMonitor:
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_conditions = {
            'nan_loss': lambda m: np.isnan(m['loss']),
            'slow_progress': lambda m: m['reward'] < threshold,
            'memory_high': lambda m: m['memory_usage'] > 0.9
        }
        
    def check_alerts(self, metrics):
        for name, condition in self.alert_conditions.items():
            if condition(metrics):
                self.send_alert(name, metrics)
```

## File Storage & Organization

### Automatic Cleanup
```python
class ExperimentCleaner:
    """Manage disk space by cleaning old experiments"""
    
    def cleanup_old_experiments(self, keep_best_n=10, keep_recent_days=30):
        # Keep best performing experiments
        # Keep recent experiments
        # Delete others but keep configs and final metrics
        pass
```

## CLI Tools

### Experiment Management CLI
```bash
# List experiments
python -m rlrepo list --filter algorithm=ppo --sort reward

# Compare experiments
python -m rlrepo compare exp1 exp2 exp3 --metrics reward,loss

# Export results
python -m rlrepo export exp1 --format latex

# Clean old experiments
python -m rlrepo cleanup --keep-best 10 --keep-recent 30
```

## Dependencies (requirements.txt)
```
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.28.0
dm-control>=1.0.0
tensorboard>=2.13.0
wandb>=0.15.0
hydra-core>=1.3.0
optuna>=3.2.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.3.0
black>=23.3.0
pyyaml>=6.0
```

## Final Notes for Implementation

1. **Start Simple**: Begin with PPO on CartPole, get the full pipeline working
2. **Test Early**: Write tests as you go, especially for checkpointing
3. **Document Everything**: Every config option should be documented
4. **Version Control**: Use git tags for stable versions
5. **Community Friendly**: Include examples and tutorials

This system is designed to handle the complete lifecycle of RL research, from quick prototyping to long-running experiments, with a focus on modularity and ease of use for researchers who want to focus on algorithms rather than infrastructure.