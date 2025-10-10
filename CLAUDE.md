# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL Lab is a modular reinforcement learning research framework designed for rapid experimentation with different RL paradigms. The framework emphasizes:
- **Paradigm-based architecture**: Model-free (PPO) and world model-based approaches
- **Component modularity**: Encoders, policy heads, value functions, and dynamics models are pluggable via config
- **Registry-based composition**: Components are registered and instantiated from YAML configs

## Common Commands

### Training
```bash
# Start a new training run
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml

# Resume from checkpoint
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --resume experiments/ppo_cartpole_20240101_120000/

# Override config values
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --seed 42 --device cuda --total-timesteps 1000000

# Dry run (validate config without training)
python scripts/train.py --config configs/experiments/ppo_cartpole.yaml --dry-run
```

### Sweeps
```bash
# Run hyperparameter sweep
python scripts/sweep.py --config configs/sweeps/ppo_extensive_sweep.yaml

# Analyze sweep results
python scripts/analyze_sweep.py --sweep-dir experiments/sweep_ppo_20240101/
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

### Installation
```bash
# Install in development mode
pip install -e .

# Install with all extras
pip install -e ".[all]"

# Install specific extras
pip install -e ".[envs,dev,vision]"
```

## Architecture

### Paradigm System

The framework is organized around **paradigms** - high-level approaches to RL:

1. **Model-Free Paradigm** (`src/paradigms/model_free/`):
   - Direct policy learning from experience
   - PPO implementation with modular components
   - Trainer: `src/paradigms/model_free/trainer.py`

2. **World Model Paradigm** (`src/paradigms/world_models/`):
   - Learns dynamics model of environment
   - Supports imagination-based training and planning
   - Modular system designed for Dreamer, TD-MPC, MuZero-style architectures
   - System orchestrator: `src/paradigms/world_models/system.py`

Each paradigm is selected via the `paradigm` field in experiment config (e.g., `paradigm: model_free` or `paradigm: world_model`).

### Component System

Components are the building blocks registered via decorators and instantiated from config:

**Core Components** (`src/components/`):
- **Encoders**: Transform observations to features (CNN, MLP)
- **Policy Heads**: Generate action distributions from representations
- **Value Functions**: Estimate state values (critics)
- **Representation Learners**: Process encoder output (identity, autoencoders, RSSM)

**World Model Components** (`src/components/world_models/`):
- **Dynamics Models**: Predict next states (RSSM, deterministic MLP)
- **Controllers**: Action selection (policy-based or planners)
- **Latents**: State representations (continuous, discrete)
- **Rollout**: Imagination trajectory generation

### Registry Pattern

Components register themselves using decorators:
```python
@register_encoder("mlp")
class MLPEncoder(BaseEncoder):
    ...

@register_policy_head("categorical_mlp")
class CategoricalMLPPolicyHead(BasePolicyHead):
    ...
```

Factory creates components from config via `src/paradigms/factory.py`:
```yaml
encoder:
  type: mlp  # looks up registered "mlp" encoder
  config:
    hidden_dims: [64, 64]
```

### Configuration Flow

1. **YAML Config** → `src/utils/config.py` loads and validates
2. **ComponentFactory** (`src/paradigms/factory.py`) creates paradigm and components
3. **Trainer** (paradigm-specific) orchestrates training loop
4. Training entry point: `scripts/train.py` routes to appropriate trainer based on `paradigm` field

### Key Abstractions

**Paradigm Interface** (`src/paradigms/base.py`):
- `forward()`: Get action distribution from observations
- `get_value()`: Get value estimate
- `compute_loss()`: Calculate training losses

**World Model System** (`src/paradigms/world_models/system.py`):
- Orchestrates encoder → representation → dynamics → policy/value pipeline
- `act()`: Action selection (via policy or planner)
- `imagine()`: Generate imagined trajectories in latent space
- `compute_losses()`: Model loss, reward loss, value loss, policy loss

### Checkpoint & Logging Architecture

**Checkpointing** (`src/utils/checkpoint.py`):
- Components save state via component-specific methods
- CheckpointManager handles serialization and RNG states
- Resume from checkpoint with `--resume` flag

**Logging** (`src/utils/logger.py`):
- Multi-backend: TensorBoard, W&B, terminal output
- Frequency-gated to reduce overhead
- Bash log file for CLI output tracking

**Known Issues** (see `docs/checkpoint_logging_architecture_analysis.md`):
- Logging frequency control split between trainer and logger
- Multiple logging paths can cause metric overwrites in TensorBoard/W&B

### Environment System

**Wrappers** (`src/environments/`):
- `gym_wrapper.py`: Basic Gym environments
- `atari_wrapper.py`: Atari with frame stacking, grayscale, etc.
- `vectorized_gym_wrapper.py`: Parallel environments
- `minigrid_wrapper.py`: MiniGrid environments

Environments selected via config:
```yaml
environment:
  name: CartPole-v1
  wrapper: gym  # or atari, vectorized_gym, minigrid
```

### World Model Development

**Universal World Model Design** (see `docs/worldModel_Universal_build_guidelines.md`):

The world model paradigm is architected to support multiple SOTA approaches (Dreamer, TD-MPC, MuZero, IRIS, TransDreamer) through configurable axes:

1. **Latent representation**: Continuous (Dreamer) vs discrete (IRIS)
2. **Dynamics model**: RSSM, Transformer, deterministic
3. **Observation prediction**: With decoder (Dreamer) vs without (MuZero, TD-MPC)
4. **Control strategy**: Learned policy vs planner (MPC/MCTS)
5. **Training regime**: Online vs offline pretraining

Key interfaces:
- `BaseDynamicsModel`: `step(state, action) -> next_state`
- `BaseController`: `act(state) -> action` (policy or planner)
- Predictive heads: decoder, reward model, value, policy

## Project-Specific Conventions

### Adding New Components

1. Create class inheriting from appropriate base (e.g., `BaseEncoder`)
2. Register with decorator: `@register_encoder("my_encoder")`
3. Add to appropriate `__init__.py` for auto-import
4. Use in config: `encoder: {type: "my_encoder", config: {...}}`

### Config Organization

- `configs/algorithms/`: Algorithm hyperparameters (e.g., PPO defaults)
- `configs/experiments/`: Complete experiment configs
- `configs/sweeps/`: Hyperparameter sweep definitions

### Experiment Outputs

All experiments save to `experiments/<name>_<timestamp>/`:
- `config.yaml`: Complete config used
- `checkpoints/`: Model checkpoints
- `logs/`: TensorBoard logs
- `bash_log.txt`: Terminal output log

### World Model Workflow (Atari)

See `docs/atari_ppo_workflow.md` for Atari-specific workflow. Key points:
- Use `wrapper: atari` for proper preprocessing
- Frame stacking and skip handled by wrapper
- Vectorized execution via `num_environments`
- Evaluation uses single env with same preprocessing

## Development Notes

### Trainer Selection

The training script (`scripts/train.py`) dynamically imports the correct trainer based on the `paradigm` field:
- `paradigm: model_free` → `src/paradigms/model_free/trainer.py`
- `paradigm: world_model` → `src/paradigms/world_model/trainer.py` (if exists)
- Falls back to `src/core/trainer.py`

### Component Wiring

Components are connected through dimension matching in factory:
1. Encoder outputs `output_dim`
2. Representation learner uses encoder's `output_dim` as `feature_dim`
3. Policy/value heads use representation learner's `representation_dim`

### World Model Training Phases

World model paradigm supports multi-phase training:
- Warmup: Train world model only (`world_model_warmup_steps`)
- Full training: Joint optimization of model, policy, value
- Configurable update ratios per component

### Debugging

- Use `--debug` flag for detailed logging
- `--dry-run` validates config and initializes components without training
- Check `bash_log.txt` for complete terminal output history
- TensorBoard logs track all metrics: `tensorboard --logdir experiments/`
