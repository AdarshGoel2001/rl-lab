# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL Lab is a modular reinforcement learning research framework designed for rapid experimentation with world model algorithms. The framework emphasizes:

- **Clean separation**: Infrastructure (training loops, logging, checkpointing) is completely separate from algorithm logic
- **Component modularity**: Encoders, dynamics models, controllers, and decoders are pluggable via Hydra configs
- **Multi-algorithm support**: Designed to support Dreamer, TD-MPC, MuZero, IRIS, and custom variants without code duplication
- **Config-driven experiments**: Architecture and hyperparameters controlled via YAML, not hardcoded

## Common Commands

### Training

```bash
# Start a new world model training run (Hydra-based)
python scripts/train.py experiment=dreamer_cartpole

# Override config values
python scripts/train.py experiment=dreamer_cartpole experiment.seed=42 training.total_timesteps=1000000

# Use different components
python scripts/train.py experiment=dreamer_cartpole components/encoder=cnn

# Dry run (validate config without training)
python scripts/train.py experiment=dreamer_cartpole --dry-run
```

**Note:** The framework uses Hydra for configuration. The old `--config` flag is replaced by `experiment=<name>`.

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_world_model_system.py

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

### Three-Layer Design

The framework uses a clean three-layer architecture:

```
┌─────────────────────────────────────────────┐
│  Entry Point: scripts/train.py             │
│  • Loads Hydra configs                      │
│  • Instantiates components via _target_     │
│  • Creates orchestrator and runs training   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Infrastructure: WorldModelOrchestrator     │
│  (src/orchestration/world_model_orchestrator.py)
│  • Training loop (run() method)             │
│  • Phase scheduling (warmup, online, eval)  │
│  • Buffer routing                           │
│  • Checkpointing, logging, evaluation       │
│  • NO algorithm-specific logic              │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Algorithm: Workflow (e.g., DreamerWorkflow)│
│  (src/workflows/world_models/)              │
│  • collect_step(): environment interaction  │
│  • update_world_model(): model learning     │
│  • update_controller(): policy learning     │
│  • imagine(): latent rollouts               │
│  • Pure algorithm logic, no infrastructure  │
└─────────────────────────────────────────────┘
```

**Key Design Principle:** Orchestrator knows nothing about algorithms. Workflow knows nothing about infrastructure.

### Component System

Components are instantiated via Hydra's `_target_` mechanism (not decorator-based registry):

**Core Components** (`src/components/`):
- **Encoders** (`encoders/`): Transform observations to features
  - `MLPEncoder`: Dense layers for vector observations
  - `CNNEncoder`: Convolutional layers for image observations

**World Model Components** (`src/components/world_models/`):
- **Representation Learners** (`representation_learners/`): Manage latent state
  - `RSSMRepresentationLearner`: Dreamer-style stochastic recurrent state
  - `IdentityRepresentationLearner`: Pass-through for simple models
  - Base protocol: `observe()`, `observe_sequence()`, `imagine_step()`

- **Dynamics Models** (`dynamics/`): Predict next latent state
  - `RSSMDynamicsModel`: Recurrent state-space model
  - Base interface: `forward(state, action) -> next_state`

- **Controllers** (`controllers/`): Action selection
  - `DreamerActorController`: Learned policy network
  - `DreamerCriticController`: Value function network
  - Each controller owns its optimizer
  - Base interface: `act(latent_state) -> action/distribution`

- **Decoders** (`decoders/observation/`): Reconstruct observations
  - `MLPObservationDecoder`: For vector observations
  - `AtariObservationDecoder`: For image observations
  - Base interface: `forward(latent) -> observation`

- **Predictors** (`reward_predictors/`, `value_predictors/`): Auxiliary heads
  - Reward prediction for world model training
  - Value prediction for planning/critic learning

### Hydra Configuration System

Components are instantiated using `_target_` paths pointing to Python classes:

**Example Component Config** (`configs/components/encoder/mlp.yaml`):
```yaml
_target_: src.components.encoders.simple_mlp.MLPEncoder
input_dim: ???  # Must be provided by experiment config
hidden_dims: [128, 128]
activation: elu
```

**Example Experiment Config** (`configs/experiment/dreamer_cartpole.yaml`):
```yaml
defaults:
  - /workflow: dreamer
  - /components/encoder: mlp
  - /components/representation_learner: rssm
  - /controller@controllers.actor: dreamer_actor
  - /controller@controllers.critic: dreamer_critic
  - /training: default
  - /buffer: world_model_sequence
  - /environment: cartpole
  - _self_

experiment:
  name: dreamer_cartpole
  seed: 42
  device: auto
  paradigm: world_model

_dims:
  observation: 4
  action: 2
  encoder_output: 128
  deterministic: 200
  stochastic: 32
  representation: ${add:${_dims.deterministic},${_dims.stochastic}}

algorithm:
  world_model_lr: 2.0e-4
  actor_lr: 3.0e-4
  imagination_horizon: 15
  gamma: 0.99

components:
  encoder:
    input_dim: ${_dims.observation}
    hidden_dims: [128, 128]

controllers:
  actor:
    representation_dim: ${_dims.representation}
    action_dim: ${_dims.action}
```

**Key Features:**
- **Compositional**: Mix and match components via defaults list
- **Dimension tracking**: `_dims` section ensures consistency
- **Interpolation**: Use `${_dims.observation}` to reference values
- **Type-safe**: Hydra validates structure at runtime

### Workflow Contract

All world model algorithms implement `WorldModelWorkflow` base class (`src/workflows/world_models/base.py`):

```python
class WorldModelWorkflow(ABC):
    @abstractmethod
    def initialize(self, context: WorkflowContext) -> None:
        """Bind components and initialize state."""

    @abstractmethod
    def update_world_model(self, batch, *, phase) -> Dict[str, float]:
        """Update dynamics model, decoder, reward predictor. Return metrics."""

    def collect_step(self, step, *, phase) -> Optional[CollectResult]:
        """Interact with environment, return trajectories."""

    def update_controller(self, batch, *, phase) -> Dict[str, float]:
        """Update policy/value networks. Return metrics."""

    def imagine(self, *, observations=None, latent=None, horizon) -> Dict[str, Any]:
        """Generate imagined rollouts in latent space."""
```

**WorkflowContext** (`src/workflows/world_models/context.py`):
Immutable dataclass providing workflows with:
- `config`: Full experiment configuration
- `device`: torch device (cuda/cpu)
- `train_environment`, `eval_environment`: Environment instances
- `components`: WorldModelComponents bundle (encoder, RSSM, decoder, etc.)
- `optimizers`: Dict of optimizers keyed by name
- `buffers`: Dict of replay buffers
- `controller_manager`: Manages actor/critic/planner controllers
- `checkpoint_manager`: Handles saving/loading
- `experiment_logger`: Logging backend (TensorBoard, W&B)

### Phase-Based Training

Training is controlled by **PhaseScheduler** (`src/orchestration/phase_scheduler.py`), enabling complex curricula:

**Example Phase Config**:
```yaml
training:
  total_timesteps: 10000000
  phases:
    - name: warmup
      type: online
      duration: 1000000  # steps
      hooks:
        - collect: {every: 1, steps: 1}
        - update_world_model: {every: 1, updates: 1}
        # No policy updates during warmup

    - name: joint_training
      type: online
      duration: 9000000
      hooks:
        - collect: {every: 1, steps: 1}
        - update_world_model: {every: 1, updates: 1}
        - update_controller: {every: 1, updates: 1}
```

**Phase Actions:**
- `collect`: Call `workflow.collect_step()`, add to buffer
- `update_world_model`: Sample batch, call `workflow.update_world_model()`
- `update_controller`: Sample batch, call `workflow.update_controller()`
- `evaluate`: Run evaluation episodes

**Use Cases:**
- Model warmup before policy training (Dreamer V2/V3)
- Offline pretraining then online finetuning (MuZero)
- Alternating model/policy updates at different ratios
- Periodic evaluation without hardcoding in workflow

## Implementing a New World Model Algorithm

### Example: Adding TD-MPC

**Files to CREATE:**

1. **Workflow** (`src/workflows/world_models/tdmpc.py`):
```python
class TDMPCWorkflow(WorldModelWorkflow):
    def initialize(self, context):
        # Bind components from context
        self.encoder = context.components.encoder
        self.dynamics_model = context.components.dynamics_model
        self.reward_predictor = context.components.reward_predictor
        self.value_function = context.controller_manager.get("critic")
        self.planner = context.controller_manager.get("planner")
        self.world_model_optimizer = context.optimizers["world_model"]

    def collect_step(self, step, *, phase):
        # Encode observation → deterministic latent
        latent = self.encoder(self.current_obs)

        # Plan action via MPC
        action = self.planner.act(latent, workflow=self)

        # Step environment
        next_obs, reward, done, info = self.environment.step(action)
        return CollectResult(steps=1, extras={"replay": trajectory})

    def update_world_model(self, batch, *, phase):
        # Deterministic dynamics loss + reward loss
        next_latent_pred = self.dynamics_model(latent, action)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target)
        # ... optimize

    def update_controller(self, batch, *, phase):
        # TD value learning (no actor)
        td_target = reward + gamma * next_value
        value_loss = F.mse_loss(value_pred, td_target)
        # ... optimize
```

2. **MPC Planner** (`src/components/world_models/controllers/mpc_planner.py`):
```python
class MPCPlanner(BaseController):
    def act(self, latent_state, *, workflow=None, **kwargs):
        # Cross-Entropy Method planning
        for _ in range(self.cem_iterations):
            action_sequences = self.sample_sequences()
            values = [workflow.imagine(latent, action_seq) for action_seq in action_sequences]
            self.refit_distribution(top_k_sequences)
        return self.action_mean[0]
```

3. **Deterministic Dynamics** (`src/components/world_models/dynamics/deterministic_mlp.py`):
```python
class DeterministicMLPDynamics(BaseDynamicsModel):
    def forward(self, state, action):
        input_tensor = torch.cat([state, action], dim=-1)
        return self.net(input_tensor)  # Direct prediction, no stochastic sampling
```

4. **Config Files** (4 YAML files in `configs/`):
   - `workflow/tdmpc.yaml`: Points to TDMPCWorkflow class
   - `controller/mpc_planner.yaml`: MPC hyperparameters
   - `components/dynamics_model/deterministic_mlp.yaml`: Network architecture
   - `experiment/tdmpc_cartpole.yaml`: Full experiment config

**Files to MODIFY: NONE**

**Total Effort:** ~700 lines of new code, zero infrastructure changes.

### Key Abstractions to Follow

**When implementing a new workflow:**
1. Inherit from `WorldModelWorkflow`
2. Implement required methods: `initialize()`, `update_world_model()`
3. Optionally implement: `collect_step()`, `update_controller()`, `imagine()`
4. Access components via `context` in `initialize()`
5. Return metrics as `Dict[str, float]` from update methods
6. Let orchestrator handle logging, checkpointing, evaluation

**When implementing a new component:**
1. Inherit from appropriate base class (e.g., `BaseEncoder`, `BaseDynamicsModel`)
2. Implement required abstract methods
3. Add `__init__` accepting `config` dict or kwargs
4. Create corresponding YAML config with `_target_` pointing to class
5. Component will be instantiated by Hydra and passed to workflow via context

## Config Organization

```
configs/
├── config.yaml                  # Hydra root (minimal)
├── experiment/                  # Complete experiment configs
│   ├── dreamer_cartpole.yaml
│   └── your_experiment.yaml
├── workflow/                    # Workflow selection
│   └── dreamer.yaml
├── components/                  # Component defaults
│   ├── encoder/
│   │   ├── mlp.yaml
│   │   └── cnn.yaml
│   ├── representation_learner/
│   │   └── rssm.yaml
│   └── dynamics_model/
│       └── rssm.yaml
├── controller/                  # Controller configs
│   ├── dreamer_actor.yaml
│   └── dreamer_critic.yaml
├── buffer/
│   └── world_model_sequence.yaml
├── training/                    # Training hyperparameters
│   └── default.yaml
├── environment/                 # Environment configs
│   ├── cartpole.yaml
│   └── atari.yaml
└── logging/
    └── tensorboard.yaml
```

## Experiment Outputs

All experiments save to `experiments/<name>_<timestamp>/`:
- `configs/config.yaml`: Complete resolved config (all Hydra interpolations resolved)
- `checkpoints/`: Model checkpoints
  - `step_<N>.pt`: Periodic checkpoints
  - `final.pt`: Final checkpoint
- Hydra creates working directory and changes to it during training

## Known Issues and Workarounds

### 1. RSSM Coupling in Dreamer Workflow

**Issue:** `src/workflows/world_models/dreamer.py` directly calls RSSM-specific methods:
```python
latent_step = self.rssm.observe(...)  # Assumes RSSM interface
```

**Impact:** Cannot swap RSSM for Transformer-based representation learner without modifying workflow.

**Workaround:** If implementing non-RSSM Dreamer variant:
1. Create new workflow inheriting from `WorldModelWorkflow`
2. Use duck typing to call your representation learner's methods
3. Or: Refactor Dreamer to use Protocol-based interface (future work)

**Future Fix:** Define `RepresentationLearnerProtocol` and use `self.representation_learner` instead of `self.rssm`.

### 2. Controller Interface Ambiguity

**Issue:** Some controllers return `Distribution` (actor), others return `action` tensor (planner).

**Workaround:** In workflow, check controller type or add mode parameter:
```python
if isinstance(controller, ActorController):
    action = controller.act(latent).rsample()
else:
    action = controller.act(latent)  # Planner returns tensor directly
```

**Future Fix:** Standardize controller interface with explicit return type contracts.

### 3. Optimizer Construction Location

**Issue:** Optimizers are built in `scripts/train.py` (lines 168-198), not via Hydra configs.

**Workaround:** To use different optimizer:
```python
# In workflow.initialize():
self.world_model_optimizer = RMSprop(params, lr=self.config.algorithm.world_model_lr)
```

**Future Fix:** Create optimizer configs: `configs/optimizer/adam.yaml`, instantiate via Hydra.

## Environment System

**Wrappers** (`src/environments/`):
- `gym_wrapper.py`: Basic Gym environments (CartPole, MountainCar, etc.)
- `atari_wrapper.py`: Atari with standard preprocessing (frame stacking, grayscale, etc.)
- `vectorized_gym_wrapper.py`: Parallel environments for faster collection
- `minigrid_wrapper.py`: MiniGrid environments

**Important:** Environments automatically reset on `done=True`. Workflows don't call `env.reset()` explicitly.

**Config Example**:
```yaml
environment:
  _target_: src.environments.gym_wrapper.GymWrapper
  name: CartPole-v1
  num_envs: 8
  transforms:
    - normalize_observations: true
```

## Buffer System

**WorldModelSequenceBuffer** (`src/buffers/world_model_sequence.py`):
- Stores vectorized trajectories in per-environment deques
- Returns contiguous sequences `(B, T, ...)` for world model training
- Supports configurable sequence length and stride
- Optional return computation (n-step, TD-λ, etc.)

**Config Example**:
```yaml
buffer:
  _target_: src.buffers.world_model_sequence.WorldModelSequenceBuffer
  capacity: 100000
  batch_size: 32
  sequence_length: 16
  sequence_stride: 8
  num_envs: ${environment.num_envs}
```

**Buffer Interface:**
- `add(trajectory=...)`: Add vectorized trajectory dict
- `sample(batch_size) -> Dict[str, torch.Tensor]`: Sample batch of sequences
- `ready() -> bool`: Check if enough data to sample
- `clear()`: Reset buffer

## Debugging and Development

### Debugging Tips

1. **Validate config without training:**
   ```bash
   python scripts/train.py experiment=dreamer_cartpole --cfg job
   ```
   This prints the full resolved config without running training.

2. **Check component dimensions:**
   ```bash
   python scripts/train.py experiment=dreamer_cartpole --dry-run
   ```
   Instantiates all components and validates dimension consistency.

3. **Monitor training:**
   ```bash
   tensorboard --logdir experiments/
   ```

4. **Check logs:**
   - Hydra outputs to `experiments/<name>_<timestamp>/`
   - Check `.hydra/config.yaml` for resolved config
   - Check `.hydra/overrides.yaml` for command-line overrides

### Common Errors

**"Config missing required key '_dims.X'":**
- Solution: Add dimension to `_dims` section in experiment config
- Dimensions must be defined: `observation`, `action`, `encoder_output`, `representation`

**"Encoder output dim doesn't match representation learner input dim":**
- Solution: Use Hydra interpolation: `feature_dim: ${_dims.encoder_output}`
- Ensures consistency across component configs

**"Buffer not ready" warning during training:**
- Solution: Buffer doesn't have enough sequences yet
- This is normal during initial collection phase
- Training will start once buffer reaches `batch_size` sequences

**"Controller not found: 'actor'":**
- Solution: Add controller to config:
  ```yaml
  defaults:
    - /controller@controllers.actor: dreamer_actor
  ```

### Logging and Metrics

Metrics are logged with prefixes:
- `collect/*`: Environment interaction metrics (reward, episode length)
- `train/*`: World model training metrics (reconstruction loss, KL divergence)
- `controller/*`: Policy/value training metrics (actor loss, critic loss, entropy)
- `eval/*`: Evaluation metrics (return_mean, return_std)
- `workflow/*`: Workflow-specific metrics (total episodes, runtime)

## Best Practices

### When Implementing New Algorithms

1. **Start with workflow skeleton:**
   - Copy `base.py` method signatures
   - Implement minimal `initialize()` and `update_world_model()`
   - Test with simple environment before adding complexity

2. **Keep workflows pure:**
   - No logging setup (use orchestrator's logger via context)
   - No checkpoint management (orchestrator handles this)
   - No training loop logic (orchestrator controls phases)
   - Focus on algorithm: losses, updates, imagination

3. **Use Hydra interpolations:**
   - Reference `_dims` for all dimension configs
   - Use `${add:...}` or `${multiply:...}` for computed dimensions
   - This prevents dimension mismatch errors

4. **Test components independently:**
   - Create components in test file
   - Verify input/output shapes
   - Check gradient flow before integrating into workflow

### When Modifying Existing Code

1. **Preserve orchestrator/workflow boundary:**
   - Don't add algorithm logic to orchestrator
   - Don't add infrastructure logic to workflow
   - Use `WorkflowContext` for communication

2. **Update configs when adding parameters:**
   - Add new parameters to component YAML configs
   - Document default values and valid ranges
   - Use `???` for required parameters (Hydra will error if missing)

3. **Maintain backward compatibility:**
   - When changing component interfaces, update all implementations
   - Consider adding optional parameters with defaults instead of breaking changes

## Reference: File Locations

**Key Files:**
- Orchestrator: `src/orchestration/world_model_orchestrator.py:42` (WorldModelOrchestrator class)
- Workflow base: `src/workflows/world_models/base.py:34` (WorldModelWorkflow class)
- Dreamer workflow: `src/workflows/world_models/dreamer.py:28` (DreamerWorkflow class)
- Context: `src/workflows/world_models/context.py` (WorkflowContext dataclass)
- Phase scheduler: `src/orchestration/phase_scheduler.py` (PhaseScheduler class)
- Buffer: `src/buffers/world_model_sequence.py:17` (WorldModelSequenceBuffer class)
- Entry point: `scripts/train.py` (Hydra main function)

**Component Base Classes:**
- Encoder: `src/components/encoders/base.py:15`
- Representation learner: `src/components/world_models/representation_learners/base.py:147`
- Dynamics model: `src/components/world_models/dynamics/base.py:12`
- Controller: `src/components/world_models/controllers/base.py:15`
- Decoder: `src/components/world_models/decoders/observation/base.py`

## Quick Start: Implementing Your Algorithm

1. **Create workflow file:** `src/workflows/world_models/your_algorithm.py`
2. **Inherit from base:** `class YourWorkflow(WorldModelWorkflow):`
3. **Implement methods:** `initialize()`, `update_world_model()`, optionally `collect_step()`, `update_controller()`, `imagine()`
4. **Create components:** If needed, implement custom encoders/dynamics/controllers
5. **Create configs:**
   - `configs/workflow/your_algorithm.yaml` with `_target_` pointing to workflow
   - `configs/experiment/your_algorithm_<env>.yaml` with full experiment config
6. **Run:** `python scripts/train.py experiment=your_algorithm_<env>`

**That's it!** The orchestrator handles training loop, checkpointing, logging, evaluation automatically.
