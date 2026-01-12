# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other LLMs working on this repository. Read this entire file before making changes.

## Project Overview

RL Lab is a modular reinforcement learning research framework for rapid experimentation with world model algorithms. The framework supports Dreamer, TD-MPC, Original World Models (Ha & Schmidhuber 2018), diffusion policies, and custom variants.

### Core Design Philosophy

1. **Infrastructure is separate from algorithms** - The Orchestrator handles training loops, checkpointing, and logging. Workflows handle algorithm logic. They know nothing about each other's internals.

2. **Phase-based training** - Complex training curricula (warmup, pretraining, joint training) are expressed declaratively in YAML, not hardcoded in workflows.

3. **Components are loosely organized** - Component folders (`controllers/`, `dynamics/`, etc.) are for organization, NOT strict interfaces. Workflows know how to use their components; components don't need to follow rigid base classes.

4. **Config-driven experiments** - Architecture and hyperparameters are controlled via Hydra YAML configs. Swapping components means changing config, not code.

---

## Architecture

### The Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│  scripts/train.py (Entry Point)                                 │
│                                                                 │
│  Responsibilities:                                              │
│  • Load Hydra config                                            │
│  • Instantiate components, controllers, optimizers, buffers     │
│  • Create Orchestrator with all resources                       │
│  • Call orchestrator.run()                                      │
│                                                                 │
│  This is the "experiment setup script". All wiring happens here.│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator (src/orchestration/orchestrator.py)               │
│                                                                 │
│  Responsibilities:                                              │
│  • Own the training loop (run() method)                         │
│  • Ask PhaseScheduler "what action next?"                       │
│  • Call workflow methods: collect_step, update_world_model, etc │
│  • Route collected data to appropriate buffer                   │
│  • Handle checkpointing (save/load all state)                   │
│  • Handle logging (metrics to TensorBoard/W&B)                  │
│                                                                 │
│  KNOWS NOTHING about algorithm internals.                       │
│  Just calls workflow methods and handles infrastructure.        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PhaseScheduler (src/orchestration/phase_scheduler.py)          │
│                                                                 │
│  Responsibilities:                                              │
│  • Track current training phase                                 │
│  • Return next action: "collect", "update_world_model", etc     │
│  • Advance phase when duration criteria met                     │
│  • Provide phase context to workflows                           │
│                                                                 │
│  A finite state machine for training curricula.                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Workflow (src/workflows/*.py)                                  │
│                                                                 │
│  Responsibilities:                                              │
│  • initialize(): Bind components from context, setup state      │
│  • collect_step(): Interact with environment, return trajectory │
│  • update_world_model(): Compute losses, backprop, return metrics│
│  • update_controller(): Update policy/value networks            │
│  • imagine(): Generate latent rollouts (optional)               │
│                                                                 │
│  PURE ALGORITHM LOGIC. No training loop, no checkpointing,      │
│  no logging setup. Just: given batch, compute loss, update.     │
└─────────────────────────────────────────────────────────────────┘
```

### Information Flow

```
1. Orchestrator asks PhaseScheduler: "What's the next action?"
2. PhaseScheduler returns: "collect" (or "update_world_model", etc.)
3. Orchestrator calls: workflow.collect_step(step, phase=phase_config)
4. Workflow does the work, returns: CollectResult(steps=N, trajectory={...})
5. Orchestrator routes trajectory to buffer, logs metrics
6. Orchestrator tells PhaseScheduler: "I did 'collect' with N steps"
7. PhaseScheduler updates internal counters, may advance to next phase
8. Loop continues...
```

### Why This Design?

- **New algorithm = new workflow file only** - Orchestrator unchanged
- **New training curriculum = YAML change only** - No code changes
- **Testing is easy** - Inject mock components into Orchestrator
- **Debugging is easy** - Clear boundaries, logs show exactly what's happening

---

## Directory Structure

```
rl-lab/
├── scripts/
│   └── train.py                 # Entry point - instantiates everything, runs training
│
├── src/
│   ├── orchestration/           # Training infrastructure
│   │   ├── orchestrator.py      # Main training loop, checkpointing, logging
│   │   └── phase_scheduler.py   # Phase progression and hook scheduling
│   │
│   ├── workflows/               # Algorithm implementations
│   │   ├── utils/               # Shared workflow utilities
│   │   │   ├── base.py          # WorldModelWorkflow ABC, CollectResult
│   │   │   ├── context.py       # WorkflowContext, WorldModelComponents
│   │   │   └── controllers.py   # ControllerManager
│   │   ├── dreamer.py           # Dreamer v1/v2/v3 workflow
│   │   ├── tdmpc.py             # TD-MPC workflow
│   │   ├── og_wm.py             # Original World Models (2018) workflow
│   │   └── diffusion_policy_workflow.py  # Diffusion policy (WIP)
│   │
│   ├── components/              # Pluggable algorithm building blocks
│   │   ├── controllers/         # Action selection (actors, critics, planners)
│   │   │   ├── dreamer.py       # DreamerActorController, DreamerCriticController
│   │   │   ├── mpc_planner.py   # MPC-based planning
│   │   │   ├── cma_es.py        # CMA-ES evolutionary controller
│   │   │   └── random_policy.py # Random action selection
│   │   ├── dynamics/            # Next-state prediction models
│   │   │   ├── deterministic_mlp.py
│   │   │   └── mdn_rnn.py       # Mixture Density Network RNN
│   │   ├── representation_learners/  # Latent state management
│   │   │   ├── rssm.py          # Recurrent State-Space Model (Dreamer)
│   │   │   ├── identity.py      # Pass-through (no learning)
│   │   │   └── conv_vae.py      # Convolutional VAE
│   │   └── return_computers/    # Return computation strategies
│   │       ├── discounted.py    # Monte Carlo returns
│   │       ├── n_step.py        # N-step returns
│   │       └── td_lambda.py     # TD-lambda returns
│   │
│   ├── environments/            # Environment wrappers
│   │   ├── base.py              # BaseEnvironment ABC
│   │   ├── gym_wrapper.py       # Gymnasium environments
│   │   ├── atari_wrapper.py     # Atari with preprocessing
│   │   ├── carracing_wrapper.py # CarRacing-v3
│   │   ├── dmc_wrapper.py       # DeepMind Control Suite
│   │   └── minigrid_wrapper.py  # MiniGrid environments
│   │
│   ├── buffers/                 # Experience storage
│   │   ├── base.py              # BaseBuffer ABC
│   │   ├── world_model_sequence.py  # Main buffer for world models
│   │   ├── disk_buffer.py       # Disk-backed buffer for large datasets
│   │   └── offline.py           # Offline dataset loading
│   │
│   └── utils/                   # Utilities
│       ├── checkpoint.py        # CheckpointManager
│       ├── config.py            # Config helpers
│       └── logger.py            # TensorBoard/W&B logging
│
├── configs/                     # Hydra YAML configurations
│   ├── config.yaml              # Root config (minimal)
│   ├── experiment/              # Complete experiment configs
│   │   ├── og_wm_carracing.yaml
│   │   └── dreamer_cartpole.yaml
│   ├── workflow/                # Workflow selection (_target_ to class)
│   ├── components/              # Component configs
│   │   ├── representation_learner/
│   │   └── dynamics_model/
│   ├── controller/              # Controller configs
│   ├── buffer/                  # Buffer configs
│   └── environment/             # Environment configs
│
├── tests/                       # Test suite
├── experiments/                 # Training outputs (gitignored)
└── datasets/                    # Generated datasets (gitignored)
```

---

## Key Abstractions

### WorldModelWorkflow (src/workflows/utils/base.py)

The base class all algorithms inherit from:

```python
class WorldModelWorkflow(ABC):
    @abstractmethod
    def initialize(self, context: WorkflowContext) -> None:
        """Called once at start. Bind components, setup state."""

    @abstractmethod
    def update_world_model(self, batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]:
        """Train world model on batch. Return metrics dict."""

    def collect_step(self, step: int, *, phase: PhaseConfig) -> Optional[CollectResult]:
        """Interact with environment. Return trajectory and metrics."""

    def update_controller(self, batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]:
        """Train controller/policy. Return metrics dict."""

    def imagine(self, *, horizon: int, **kwargs) -> Dict[str, Any]:
        """Generate imagined rollouts in latent space."""

    def get_state(self) -> Dict[str, Any]:
        """Return custom state for checkpointing."""

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore custom state from checkpoint."""
```

**Important**: The base class provides episode tracking utilities:
- `_reset_episode_tracking(num_envs)` - Initialize tracking for vectorized envs
- `_update_episode_stats(rewards, dones, infos)` - Update running stats
- `_snapshot_episode_tracking()` / `_restore_episode_tracking()` - For checkpointing

### WorkflowContext (src/workflows/utils/context.py)

Immutable bundle of resources passed to workflow.initialize():

```python
@dataclass(frozen=True)
class WorkflowContext:
    config: DictConfig              # Full experiment config
    device: str                     # "cuda" or "cpu"
    train_environment: Any          # Training environment
    eval_environment: Any           # Evaluation environment
    components: WorldModelComponents # All instantiated components
    buffers: Dict[str, Any]         # Named buffers
    optimizers: Dict[str, Any]      # Named optimizers
    controller_manager: ControllerManager
    checkpoint_manager: CheckpointManager
    experiment_logger: Any          # TensorBoard/W&B logger
    initial_observation: Any        # First obs from env.reset()
    initial_dones: Any              # Initial done flags
    global_step: int                # Current training step
```

Access components via: `context.components.vae`, `context.components.dynamics_model`, etc.

### PhaseScheduler (src/orchestration/phase_scheduler.py)

Controls training curriculum. Each phase specifies:
- **name**: Identifier for the phase
- **type**: "online", "offline", or "eval_only"
- **duration**: How long (steps, updates, or cycles)
- **hooks**: Which workflow methods to call

```python
# PhaseScheduler provides:
scheduler.current_phase()  # Returns PhaseDefinition
scheduler.next_action()    # Returns "collect", "update_world_model", etc.
scheduler.advance(action, steps=N)  # Update counters
scheduler.get_state() / set_state()  # For checkpointing
```

### CollectResult (src/workflows/utils/base.py)

Returned by workflow.collect_step():

```python
@dataclass
class CollectResult:
    steps: int                      # Environment steps taken
    episodes: int = 0               # Episodes completed
    metrics: Dict[str, float] = {}  # Metrics to log
    trajectory: Optional[Dict] = None  # Data for buffer
    extras: Dict[str, Any] = {}     # Additional data
```

---

## Phase Configuration

Phases are defined in experiment YAML under `training.phases`:

```yaml
training:
  total_timesteps: 1000000
  phases:
    # Phase 1: Collect random data
    - name: data_collection
      type: online
      buffer: replay              # Which buffer to use
      duration_steps: 10000       # Run for 10k env steps
      workflow_hooks:
        - collect                 # Only collect, no training

    # Phase 2: Train world model
    - name: train_world_model
      type: offline              # No collection
      buffer: replay
      duration_updates: 50000    # 50k gradient updates
      workflow_hooks:
        - update_world_model

    # Phase 3: Joint training
    - name: joint_training
      type: online
      buffer: replay
      duration_steps: 500000
      workflow_hooks:
        - collect
        - update_world_model
        - update_controller
```

**Duration options:**
- `duration_steps`: Environment steps
- `duration_updates`: Gradient updates
- `duration_cycles`: Complete hook cycles

**Hook types:**
- `collect`: Call workflow.collect_step()
- `update_world_model`: Call workflow.update_world_model()
- `update_controller`: Call workflow.update_controller()
- `evaluate`: Run evaluation episodes

**Workflow receives phase context:**
```python
def update_world_model(self, batch, *, phase):
    if phase['name'] == 'converge_vae':
        # VAE-specific training
    elif phase['name'] == 'converge_mdn':
        # MDN-specific training
```

---

## Implementing a New Algorithm

### Step 1: Create the Workflow

Create `src/workflows/your_algorithm.py`:

```python
from .utils.base import WorldModelWorkflow, CollectResult, Batch, PhaseConfig
from .utils.context import WorkflowContext

class YourWorkflow(WorldModelWorkflow):
    def __init__(self):
        super().__init__()

    def initialize(self, context: WorkflowContext) -> None:
        """Bind components and setup state."""
        self._bind_context(context)
        self._reset_episode_tracking(self.num_envs, clear_history=True)
        self._reset_rollout_state(
            initial_obs=context.initial_observation,
            initial_dones=context.initial_dones,
        )

    def _bind_context(self, context: WorkflowContext) -> None:
        """Extract what you need from context. Use your own variable names."""
        self.config = context.config
        self.device = context.device
        self.environment = context.train_environment
        self.num_envs = int(getattr(self.environment, "num_envs", 1))

        # Get components - use whatever names make sense for your algorithm
        components = context.components
        self.encoder = getattr(components, "encoder", None)
        self.dynamics = getattr(components, "dynamics_model", None)

        # Get optimizers
        self.optimizer = context.optimizers.get("world_model")

    def collect_step(self, step: int, *, phase: PhaseConfig) -> CollectResult:
        """One step of environment interaction."""
        with torch.no_grad():
            # Your collection logic
            action = self.select_action(self.current_obs)
            next_obs, reward, done, info = self.environment.step(action)

            # Track episode stats (provided by base class)
            self._update_episode_stats(reward, done, info)

            # Build trajectory
            trajectory = {
                "observations": self.current_obs,
                "actions": action,
                "rewards": reward,
                "dones": done,
            }

            self.current_obs = next_obs

        return CollectResult(
            steps=self.num_envs,
            trajectory=trajectory,
            metrics={"collect/reward": float(reward.mean())},
        )

    def update_world_model(self, batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]:
        """One gradient step on world model."""
        # Your loss computation
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
```

### Step 2: Create Config Files

**Workflow config** (`configs/workflow/your_algorithm.yaml`):
```yaml
_target_: src.workflows.your_algorithm.YourWorkflow
```

**Experiment config** (`configs/experiment/your_algorithm_env.yaml`):
```yaml
# @package _global_
defaults:
  - /workflow: your_algorithm
  - /components/representation_learner@components.encoder: your_encoder
  - /controller@controllers.actor: your_controller
  - /buffer: world_model_sequence
  - /environment: cartpole
  - _self_

experiment:
  name: your_algorithm_cartpole
  seed: 42
  device: auto

_dims:
  observation: 4
  action: 2
  latent: 64

algorithm:
  learning_rate: 1e-4
  # Your hyperparameters

components:
  encoder:
    input_dim: ${_dims.observation}
    latent_dim: ${_dims.latent}

training:
  total_timesteps: 100000
  phases:
    - name: training
      type: online
      buffer: replay
      duration_steps: 100000
      workflow_hooks:
        - collect
        - update_world_model
```

### Step 3: Run

```bash
python scripts/train.py +experiment=your_algorithm_env
```

---

## Common Commands

### Training

```bash
# Run experiment
python scripts/train.py +experiment=og_wm_carracing

# Override config values
python scripts/train.py +experiment=og_wm_carracing experiment.seed=123

# Resume from checkpoint
python scripts/train.py +experiment=og_wm_carracing training.resume_path=path/to/checkpoint.pt

# Print resolved config (no training)
python scripts/train.py +experiment=og_wm_carracing --cfg job
```

### Testing

```bash
pytest                           # All tests
pytest tests/test_file.py        # Specific file
pytest -v                        # Verbose
pytest -x                        # Stop on first failure
```

### Monitoring

```bash
tensorboard --logdir experiments/
```

---

## Component Organization

Components are organized by function, NOT by strict interfaces:

```
src/components/
├── controllers/       # Things that select actions
├── dynamics/          # Things that predict next states
├── representation_learners/  # Things that manage latent states
└── return_computers/  # Things that compute returns
```

**Key principle**: Workflows know how to use their components. Components don't need to follow rigid base classes.

For example, `og_wm.py` uses:
- `self.vae` with methods: `observe()`, `observe_sequence()`, `decode()`
- `self.mdn_rnn` with methods: `observe()`, `observe_sequence()`, `reset_state()`
- `self.controller` with method: `act()`

While `dreamer.py` uses:
- `self.rssm` with methods: `observe()`, `observe_sequence()`, `imagine_step()`
- `self.actor` and `self.critic` controllers

Each workflow documents (via its `_bind_context()`) what it expects from its components.

---

## Checkpointing

The Orchestrator handles all checkpointing. It saves:

```python
{
    "version": 1,
    "global_step": 12345,
    "workflow_name": "OriginalWorldModelsWorkflow",
    "phase_state": {...},           # PhaseScheduler state
    "components": {...},            # All component state_dicts
    "controllers": {...},           # All controller state_dicts
    "optimizers": {...},            # All optimizer state_dicts
    "workflow_custom": {...},       # workflow.get_state() output
}
```

To add custom state to checkpoints, implement in your workflow:
```python
def get_state(self) -> Dict[str, Any]:
    return {
        "my_counter": self.my_counter,
        "episode_returns": list(self.episode_returns),
    }

def set_state(self, state: Dict[str, Any]) -> None:
    self.my_counter = state.get("my_counter", 0)
    self.episode_returns = state.get("episode_returns", [])
```

---

## Hydra Configuration

### Key Patterns

**Composition via defaults:**
```yaml
defaults:
  - /workflow: dreamer
  - /components/encoder@components.encoder: mlp
  - /controller@controllers.actor: dreamer_actor
```

**Dimension tracking:**
```yaml
_dims:
  observation: 4
  action: 2
  latent: ${add:${_dims.observation}, 10}  # Computed dimension

components:
  encoder:
    input_dim: ${_dims.observation}  # Reference
```

**Partial instantiation (for optimizers):**
```yaml
optimizers:
  world_model:
    _target_: torch.optim.Adam
    _partial_: true  # Creates partial, params added by train.py
    lr: 1e-4
```

### Config Resolution

When you run `python scripts/train.py +experiment=og_wm_carracing`:

1. Hydra loads `configs/config.yaml` (root)
2. Loads `configs/experiment/og_wm_carracing.yaml`
3. Follows `defaults` list, loading each referenced config
4. Merges everything, resolving `${...}` interpolations
5. Final config passed to `main(cfg)`

---

## Debugging

### Config Issues

```bash
# Print full resolved config
python scripts/train.py +experiment=your_exp --cfg job

# Print just the job config (no Hydra internals)
python scripts/train.py +experiment=your_exp --cfg job --package _global_
```

### Training Issues

1. **Buffer not ready** - Normal during initial collection. Training starts when buffer has enough data.

2. **Dimension mismatch** - Check `_dims` section, ensure all interpolations resolve correctly.

3. **Component not found** - Check component name in `_bind_context()` matches config key.

### Logging

Metrics are logged with prefixes:
- `collect/*` - Environment interaction
- `train/*` - World model training (from `update_world_model`)
- `controller/*` - Controller training (from `update_controller`)
- `eval/*` - Evaluation metrics
- `workflow/*` - Custom workflow metrics

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train.py` | Entry point - instantiation and wiring |
| `src/orchestration/orchestrator.py` | Training loop, checkpointing, logging |
| `src/orchestration/phase_scheduler.py` | Phase progression FSM |
| `src/workflows/utils/base.py` | WorldModelWorkflow ABC, CollectResult |
| `src/workflows/utils/context.py` | WorkflowContext dataclass |
| `src/workflows/utils/controllers.py` | ControllerManager |
| `src/workflows/dreamer.py` | Dreamer implementation |
| `src/workflows/og_wm.py` | Original World Models implementation |
| `src/buffers/world_model_sequence.py` | Main replay buffer |
| `src/utils/checkpoint.py` | CheckpointManager |

---

## Design Decisions

### Why is `_bind_context()` in each workflow, not the base class?

Each workflow extracts different things and uses different names. Keeping it explicit:
- Reader sees exactly what's being used
- Workflow author chooses their own names (`self.vae` vs `self.encoder`)
- No hidden magic

### Why does train.py do all the instantiation?

train.py is the "experiment setup script". Orchestrator is the "execution engine". This separation:
- Makes testing easier (inject mocks into Orchestrator)
- Keeps Orchestrator focused on execution
- Allows custom instantiation logic per experiment

### Why `update_world_model` and `update_controller` as hook names?

These cover most RL algorithms:
- World models: `update_world_model` = VAE/dynamics, `update_controller` = policy
- Model-free: `update_world_model` = value network, `update_controller` = policy
- VLAs: `update_world_model` = main model, `update_controller` = optional

If you need more hooks (e.g., GAN discriminator), the system is extensible.

### Why no strict interfaces for components?

This is a research framework. Strict interfaces:
- Limit experimentation
- Force artificial categorization
- Add boilerplate

Instead, workflows document what they need, and components are organized loosely by function.

---

## Existing Workflows

### DreamerWorkflow (`src/workflows/dreamer.py`)
- RSSM-based world model
- Actor-critic in imagination
- Supports Dreamer v1/v2/v3 variants

### TDMPCWorkflow (`src/workflows/tdmpc.py`)
- Deterministic latent dynamics
- MPC planning with learned value function
- TD learning for value

### OriginalWorldModelsWorkflow (`src/workflows/og_wm.py`)
- VAE for visual encoding
- MDN-RNN for dynamics
- CMA-ES for controller (WIP)
- Follows Ha & Schmidhuber 2018

### DiffusionPolicyWorkflow (`src/workflows/diffusion_policy_workflow.py`)
- Diffusion-based action generation
- Behavioral cloning from demonstrations
- Work in progress

---

## Quick Reference

### Running an experiment
```bash
python scripts/train.py +experiment=<name>
```

### Adding a new algorithm
1. Create `src/workflows/your_algo.py` inheriting from `WorldModelWorkflow`
2. Create `configs/workflow/your_algo.yaml` with `_target_`
3. Create `configs/experiment/your_algo_env.yaml`

### Adding a new component
1. Create `src/components/<category>/your_component.py`
2. Create `configs/components/<category>/your_component.yaml` with `_target_`
3. Reference in experiment config

### Key methods to implement
```python
initialize(context)           # Required - bind components
update_world_model(batch, phase)  # Required - train model
collect_step(step, phase)     # Optional - env interaction
update_controller(batch, phase)   # Optional - train policy
get_state() / set_state()     # Optional - custom checkpointing
```

---

## Notes for LLMs

1. **Always check actual file paths** - This doc is accurate as of the last update, but verify files exist.

2. **The orchestrator/workflow boundary is sacred** - Don't add algorithm logic to Orchestrator, don't add infrastructure to workflows.

3. **Phase config goes to workflows** - Workflows receive `phase` dict with name, type, progress. Branch on `phase['name']` for phase-specific behavior.

4. **Components are duck-typed** - Workflows call whatever methods they need. No strict interfaces.

5. **Hydra instantiation** - Components are created via `instantiate(cfg.component)`. The `_target_` field points to the Python class.

6. **Buffer routing** - Orchestrator routes `CollectResult.trajectory` to the buffer specified in phase config.

7. **Episode tracking** - Use base class utilities: `_reset_episode_tracking()`, `_update_episode_stats()`.
