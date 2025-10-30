"""Comprehensive end-to-end test suite for world model training system.

This test suite validates:
1. Component instantiation and dimension wiring
2. Data flow through all layers (orchestrator → workflow → components)
3. Multi-phase training (warmup, joint training, evaluation)
4. Buffer management (collection, sampling, sequence extraction)
5. State management (RSSM state, episode tracking, optimizer state)
6. Checkpointing and resumption
7. Vectorized environments
8. Edge cases (buffer not ready, empty batches, etc.)

The tests use real components with a simple environment (CartPole) for fast execution
while accurately emulating production training scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.buffers.world_model_sequence import WorldModelSequenceBuffer
from src.components.encoders.simple_mlp import MLPEncoder
from src.components.world_models.controllers.dreamer import (
    DreamerActorController,
    DreamerCriticController,
)
from src.components.world_models.decoders.observation.mlp import MLPObservationDecoder
from src.components.world_models.representation_learners.rssm import (
    RSSMRepresentationLearner,
)
from src.environments.gym_wrapper import GymWrapper
from src.orchestration.phase_scheduler import PhaseScheduler
from src.orchestration.world_model_orchestrator import WorldModelOrchestrator
from src.utils.checkpoint import CheckpointManager
from src.workflows.world_models.context import WorkflowContext, WorldModelComponents
from src.workflows.world_models.controllers import ControllerManager
from src.workflows.world_models.dreamer import DreamerWorkflow


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiment outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cartpole_config():
    """Minimal config for CartPole Dreamer training."""
    config = {
        "experiment": {
            "name": "test_dreamer_cartpole",
            "seed": 42,
            "device": "cpu",
            "paradigm": "world_model",
        },
        "_dims": {
            "observation": 4,
            "action": 2,
            "encoder_output": 64,
            "deterministic": 128,
            "stochastic": 32,
            "representation": 160,  # deterministic + stochastic
        },
        "algorithm": {
            "world_model_lr": 1e-3,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "imagination_horizon": 10,
            "gamma": 0.99,
            "lambda_return": 0.95,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "free_nats": 0.0,
            "kl_scale": 1.0,
            "collect_length": 8,
        },
        "components": {
            "encoder": {
                "input_dim": 4,
                "hidden_dims": [64, 64],
                "activation": "relu",
            },
            "representation_learner": {
                "feature_dim": 64,
                "deterministic_dim": 128,
                "stochastic_dim": 32,
                "action_dim": 2,
                "hidden_dim": 256,
            },
            "dynamics_model": {
                "feature_dim": 64,
                "deterministic_dim": 128,
                "stochastic_dim": 32,
                "action_dim": 2,
                "hidden_dim": 256,
            },
            "observation_decoder": {
                "representation_dim": 160,
                "output_dim": 4,
                "hidden_dims": [64, 64],
            },
            "reward_predictor": {
                "representation_dim": 160,
                "output_dim": 1,
                "hidden_dims": [64],
            },
        },
        "controllers": {
            "actor": {
                "representation_dim": 160,
                "action_dim": 2,
                "discrete_actions": True,
                "hidden_dims": [128, 128],
                "lr": 3e-4,
            },
            "critic": {
                "representation_dim": 160,
                "hidden_dims": [128, 128],
                "lr": 3e-4,
            },
        },
        "buffer": {
            "capacity": 10000,
            "batch_size": 16,
            "sequence_length": 8,
            "sequence_stride": 4,
            "num_envs": 4,
            "gamma": 0.99,
        },
        "environment": {
            "name": "CartPole-v1",
            "num_envs": 4,
        },
        "training": {
            "total_timesteps": 1000,
            "checkpoint_frequency": 0,
            "eval_frequency": 0,
            "policy_warmup_updates": 0,
            "phases": [],
        },
        "logging": {
            "log_frequency": 100,
            "backends": [],
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def simple_reward_predictor():
    """Simple MLP reward predictor."""
    class SimpleRewardPredictor(nn.Module):
        def __init__(self, representation_dim, output_dim=1, hidden_dims=None):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [64]
            layers = []
            input_dim = representation_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return SimpleRewardPredictor


class TestComponentInstantiation:
    """Test that all components instantiate correctly with proper dimensions."""

    def test_encoder_instantiation(self, cartpole_config):
        """Test encoder creates correct output dimensions."""
        encoder = MLPEncoder(**cartpole_config.components.encoder)

        # Test forward pass
        batch_size = 8
        obs = torch.randn(batch_size, cartpole_config._dims.observation)
        features = encoder(obs)

        assert features.shape == (batch_size, cartpole_config._dims.encoder_output)
        assert encoder.output_dim == cartpole_config._dims.encoder_output

    def test_rssm_instantiation(self, cartpole_config):
        """Test RSSM representation learner instantiation."""
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)

        # Test observe single step
        batch_size = 8
        features = torch.randn(batch_size, cartpole_config._dims.encoder_output)
        actions = torch.randn(batch_size, cartpole_config._dims.action)

        latent_step = rssm.observe(features, prev_action=actions)
        latent_tensor = latent_step.posterior.to_tensor()

        assert latent_tensor.shape == (batch_size, cartpole_config._dims.representation)
        assert rssm.representation_dim == cartpole_config._dims.representation

    def test_dynamics_model_instantiation(self, cartpole_config):
        """Test RSSM representation learner (also serves as dynamics model in Dreamer)."""
        # In Dreamer, RSSM handles both representation and dynamics
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)

        # Test imagine_step (dynamics function)
        batch_size = 8
        features = torch.randn(batch_size, cartpole_config._dims.encoder_output)
        actions = torch.randn(batch_size, cartpole_config._dims.action)

        latent_step = rssm.observe(features, prev_action=actions)
        next_latent_step = rssm.imagine_step(latent_step.posterior, actions)

        # imagine_step returns LatentStep with prior and posterior
        assert next_latent_step.prior.to_tensor().shape == (batch_size, cartpole_config._dims.representation)

    def test_decoder_instantiation(self, cartpole_config):
        """Test observation decoder reconstructs correct shape."""
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)

        batch_size = 8
        latent = torch.randn(batch_size, cartpole_config._dims.representation)

        recon_obs = decoder(latent)

        assert recon_obs.shape == (batch_size, cartpole_config._dims.observation)

    def test_controller_instantiation(self, cartpole_config):
        """Test actor and critic controllers instantiate correctly."""
        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)

        batch_size = 8
        latent = torch.randn(batch_size, cartpole_config._dims.representation)

        # Test actor forward
        dist = actor.forward(latent)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        assert action.shape == (batch_size, cartpole_config._dims.action)
        assert log_prob.shape == (batch_size,)

        # Test critic forward
        value = critic.forward(latent)
        assert value.shape == (batch_size, 1) or value.shape == (batch_size,)

    def test_dimension_consistency(self, cartpole_config):
        """Test that encoder → RSSM → decoder dimensions are consistent."""
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)

        batch_size = 8
        obs = torch.randn(batch_size, cartpole_config._dims.observation)
        actions = torch.randn(batch_size, cartpole_config._dims.action)

        # Forward pass through entire pipeline
        features = encoder(obs)
        latent_step = rssm.observe(features, prev_action=actions)
        latent = latent_step.posterior.to_tensor()
        recon_obs = decoder(latent)

        # Verify dimensions match
        assert features.shape[1] == encoder.output_dim
        assert latent.shape[1] == rssm.representation_dim
        assert recon_obs.shape == obs.shape


class TestBufferDataFlow:
    """Test buffer collection, storage, and sampling."""

    def test_buffer_add_trajectory(self, cartpole_config):
        """Test adding vectorized trajectories to buffer."""
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        num_envs = cartpole_config.buffer.num_envs
        time_steps = 16

        # Create fake trajectory
        trajectory = {
            "observations": np.random.randn(time_steps, num_envs, 4).astype(np.float32),
            "next_observations": np.random.randn(time_steps, num_envs, 4).astype(np.float32),
            "actions": np.random.randn(time_steps, num_envs, 2).astype(np.float32),
            "rewards": np.random.randn(time_steps, num_envs).astype(np.float32),
            "dones": np.random.randint(0, 2, (time_steps, num_envs)).astype(bool),
        }

        # Add to buffer
        buffer.add(trajectory=trajectory)

        # Check buffer size
        assert buffer.size > 0
        assert not buffer.ready()  # Not enough for batch yet

    def test_buffer_sampling_sequences(self, cartpole_config):
        """Test sampling contiguous sequences from buffer."""
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        num_envs = cartpole_config.buffer.num_envs
        sequence_length = cartpole_config.buffer.sequence_length

        # Add enough trajectories to make buffer ready
        for _ in range(10):
            trajectory = {
                "observations": np.random.randn(16, num_envs, 4).astype(np.float32),
                "next_observations": np.random.randn(16, num_envs, 4).astype(np.float32),
                "actions": np.random.randn(16, num_envs, 2).astype(np.float32),
                "rewards": np.random.randn(16, num_envs).astype(np.float32),
                "dones": np.zeros((16, num_envs), dtype=bool),
            }
            buffer.add(trajectory=trajectory)

        assert buffer.ready()

        # Sample batch
        batch_size = 8
        batch = buffer.sample(batch_size)

        # Check batch shapes
        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "dones" in batch

        # Should be (batch_size, sequence_length, ...)
        assert batch["observations"].shape[0] == batch_size
        assert batch["observations"].shape[1] == sequence_length
        assert batch["actions"].shape == (batch_size, sequence_length, 2)
        assert batch["rewards"].shape == (batch_size, sequence_length)

    def test_buffer_not_ready_handling(self, cartpole_config):
        """Test buffer behavior when not ready to sample."""
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        # Buffer should not be ready initially
        assert not buffer.ready()

        # Should raise error if trying to sample
        with pytest.raises(ValueError, match="does not contain enough data"):
            buffer.sample()


class TestWorkflowDataFlow:
    """Test data flow through workflow methods."""

    @pytest.fixture
    def workflow_with_components(self, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Create workflow with all components."""
        # Create components
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        # In Dreamer, RSSM serves both as representation learner and dynamics model
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        # Create controllers
        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({
            "actor": actor,
            "critic": critic,
        })

        # Create environment
        env = GymWrapper(name=cartpole_config.environment.name, num_envs=cartpole_config.environment.num_envs)

        # Create buffer
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        # Create optimizers
        world_model_params = (
            list(encoder.parameters()) +
            list(rssm.parameters()) +
            list(dynamics.parameters()) +
            list(decoder.parameters()) +
            list(reward_predictor.parameters())
        )
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=cartpole_config.algorithm.world_model_lr)

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(temp_experiment_dir, auto_save_frequency=0)

        # Create mock logger
        mock_logger = MagicMock()

        # Create context
        context = WorkflowContext(
            config=cartpole_config,
            device="cpu",
            train_environment=env,
            eval_environment=env,
            components=components,
            checkpoint_manager=checkpoint_manager,
            experiment_logger=mock_logger,
            buffers={"replay": buffer},
            controllers={"actor": actor, "critic": critic},
            controller_manager=controller_manager,
            optimizers={"world_model": world_model_optimizer},
            initial_observation=env.reset(),
            initial_dones=np.zeros(cartpole_config.environment.num_envs, dtype=bool),
            global_step=0,
        )

        # Create workflow
        workflow = DreamerWorkflow()
        workflow.initialize(context)

        return workflow, context

    def test_workflow_collect_step(self, workflow_with_components):
        """Test workflow collect_step produces correct data structure."""
        workflow, context = workflow_with_components

        phase = {"type": "online", "collect_length": 4}
        result = workflow.collect_step(step=0, phase=phase)

        assert result is not None
        assert result.steps > 0
        assert "replay" in result.extras

        trajectory = result.extras["replay"]["trajectory"]
        assert "observations" in trajectory
        assert "actions" in trajectory
        assert "rewards" in trajectory
        assert "dones" in trajectory

        # Check shapes
        collect_length = 4
        num_envs = context.config.environment.num_envs
        assert trajectory["observations"].shape == (collect_length, num_envs, 4)
        assert trajectory["actions"].shape == (collect_length, num_envs, 2)
        assert trajectory["rewards"].shape == (collect_length, num_envs)
        assert trajectory["dones"].shape == (collect_length, num_envs)

    def test_workflow_update_world_model(self, workflow_with_components):
        """Test workflow update_world_model computes losses correctly."""
        workflow, context = workflow_with_components

        # Create fake batch
        batch_size = 8
        sequence_length = 8
        batch = {
            "observations": torch.randn(batch_size, sequence_length, 4),
            "actions": torch.randn(batch_size, sequence_length, 2),
            "rewards": torch.randn(batch_size, sequence_length),
            "dones": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        }

        phase = {"type": "online"}
        metrics = workflow.update_world_model(batch, phase=phase)

        # Check metrics exist
        assert "world_model/total_loss" in metrics
        assert "world_model/kl_loss" in metrics
        assert "world_model/recon_loss" in metrics
        assert "world_model/reward_loss" in metrics

        # Check metrics are finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

        # Check workflow update counter incremented
        assert workflow.world_model_updates == 1

    def test_workflow_update_controller(self, workflow_with_components):
        """Test workflow update_controller via imagination."""
        workflow, context = workflow_with_components

        # Need world model updates first for actor warmup
        workflow.world_model_updates = 100
        workflow.actor_warmup_updates = 0

        # Create fake batch
        batch_size = 8
        sequence_length = 8
        batch = {
            "observations": torch.randn(batch_size, sequence_length, 4),
            "actions": torch.randn(batch_size, sequence_length, 2),
            "dones": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        }

        phase = {"type": "online", "imagination_horizon": 10}
        metrics = workflow.update_controller(batch, phase=phase)

        # Check metrics exist
        assert "controller/actor_loss" in metrics
        assert "controller/critic_loss" in metrics
        assert "controller/entropy" in metrics

        # Check metrics are finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_workflow_imagine(self, workflow_with_components):
        """Test workflow imagination rollouts."""
        workflow, context = workflow_with_components

        batch_size = 4
        horizon = 10

        # Create start latent state
        obs = torch.randn(batch_size, 4)
        actions = torch.randn(batch_size, 2)
        latent_step = workflow.rssm.observe(obs, prev_action=actions)
        start_latent = latent_step.posterior

        # Imagine
        rollout = workflow.imagine(latent=start_latent, horizon=horizon, deterministic=False)

        # Check rollout structure
        assert "states" in rollout
        assert "actions" in rollout
        assert "rewards" in rollout
        assert "values" in rollout
        assert "log_probs" in rollout
        assert "entropies" in rollout
        assert "bootstrap" in rollout

        # Check shapes
        assert rollout["states"].shape == (batch_size, horizon, context.config._dims.representation)
        assert rollout["actions"].shape == (batch_size, horizon, 2)
        assert rollout["rewards"].shape == (batch_size, horizon)
        assert rollout["values"].shape == (batch_size, horizon)
        assert rollout["bootstrap"].shape == (batch_size,)


class TestOrchestratorIntegration:
    """Test full orchestrator integration with all components."""

    @pytest.fixture
    def full_system(self, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Create full system: orchestrator + workflow + components."""
        # Create components
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        # Create controllers
        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({
            "actor": actor,
            "critic": critic,
        })

        # Create environments
        train_env = GymWrapper(name=cartpole_config.environment.name, num_envs=cartpole_config.environment.num_envs)
        eval_env = GymWrapper(name=cartpole_config.environment.name, num_envs=1)

        # Create buffer
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        # Create optimizers
        world_model_params = (
            list(encoder.parameters()) +
            list(rssm.parameters()) +
            list(dynamics.parameters()) +
            list(decoder.parameters()) +
            list(reward_predictor.parameters())
        )
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=cartpole_config.algorithm.world_model_lr)

        # Create workflow
        workflow = DreamerWorkflow()

        # Create orchestrator
        orchestrator = WorldModelOrchestrator(
            config=cartpole_config,
            workflow=workflow,
            experiment_dir=temp_experiment_dir,
            components=components,
            optimizers={"world_model": world_model_optimizer},
            train_environment=train_env,
            eval_environment=eval_env,
            buffers={"replay": buffer},
            controller_manager=controller_manager,
        )

        return orchestrator

    def test_orchestrator_initialization(self, full_system):
        """Test orchestrator initializes workflow correctly."""
        orchestrator = full_system

        orchestrator.initialize()

        # Check workflow is initialized
        assert orchestrator.workflow is not None
        assert orchestrator.workflow.encoder is not None
        assert orchestrator.workflow.rssm is not None
        assert orchestrator.workflow.actor_controller is not None
        assert orchestrator.workflow.critic_controller is not None
        assert orchestrator.global_step == 0

    @patch("src.utils.logger.create_logger")
    def test_orchestrator_simple_training_loop(self, mock_create_logger, full_system):
        """Test orchestrator runs simple training loop."""
        # Mock logger to avoid file I/O
        mock_logger = MagicMock()
        mock_create_logger.return_value = mock_logger

        orchestrator = full_system
        orchestrator.config.training.total_timesteps = 100  # Very short training
        orchestrator.config.logging.log_frequency = 0

        # Run training
        final_metrics = orchestrator.run()

        # Check training completed
        assert orchestrator.global_step > 0
        assert orchestrator.global_step <= 100
        assert "wall_time" in final_metrics

        # Check logger was called
        assert mock_logger.log_metrics.called

    @patch("src.utils.logger.create_logger")
    def test_orchestrator_buffer_routing(self, mock_create_logger, full_system):
        """Test orchestrator routes collected data to buffer correctly."""
        mock_logger = MagicMock()
        mock_create_logger.return_value = mock_logger

        orchestrator = full_system
        orchestrator.initialize()

        # Manually trigger collect step
        phase = {"type": "online", "collect_length": 8}
        result = orchestrator.workflow.collect_step(step=0, phase=phase)

        # Route to buffer
        orchestrator._route_collect_result(result)

        # Check buffer received data
        buffer = orchestrator.buffers["replay"]
        assert buffer.size > 0


class TestMultiPhaseTraining:
    """Test different phase configurations and transitions."""

    @pytest.fixture
    def phase_configs(self):
        """Different phase configurations to test."""
        return {
            "warmup_then_joint": [
                {
                    "name": "warmup",
                    "type": "online",
                    "duration": {"steps": 100},
                    "hooks": [
                        {"collect": {"every": 1, "steps": 1}},
                        {"update_world_model": {"every": 1, "updates": 1}},
                    ],
                },
                {
                    "name": "joint",
                    "type": "online",
                    "duration": {"steps": 100},
                    "hooks": [
                        {"collect": {"every": 1, "steps": 1}},
                        {"update_world_model": {"every": 1, "updates": 1}},
                        {"update_controller": {"every": 1, "updates": 1}},
                    ],
                },
            ],
            "alternating_updates": [
                {
                    "name": "alternating",
                    "type": "online",
                    "duration": {"steps": 200},
                    "hooks": [
                        {"collect": {"every": 1, "steps": 1}},
                        {"update_world_model": {"every": 1, "updates": 2}},  # 2x world model updates
                        {"update_controller": {"every": 2, "updates": 1}},  # Half controller updates
                    ],
                },
            ],
        }

    def test_phase_scheduler_warmup_then_joint(self, phase_configs):
        """Test phase scheduler with warmup then joint training."""
        scheduler = PhaseScheduler(phase_configs["warmup_then_joint"])

        # Should start in warmup phase
        phase = scheduler.current_phase()
        assert phase.name == "warmup"

        # Collect actions in warmup
        actions_seen = set()
        for _ in range(150):  # Run through warmup phase
            action = scheduler.next_action()
            if action:
                actions_seen.add(action)
                if action == "collect":
                    scheduler.advance(action, steps=4)  # 4 envs
                else:
                    scheduler.advance(action, updates=1)

            if scheduler.current_phase().name != "warmup":
                break

        # Should have seen collect and update_world_model, but NOT update_controller
        assert "collect" in actions_seen
        assert "update_world_model" in actions_seen
        assert "update_controller" not in actions_seen

        # Should transition to joint phase
        phase = scheduler.current_phase()
        assert phase.name == "joint"

        # Now should see update_controller
        action = scheduler.next_action()
        while action != "update_controller" and not scheduler.is_finished():
            scheduler.advance(action, steps=4 if action == "collect" else None, updates=1 if action != "collect" else None)
            action = scheduler.next_action()

        assert action == "update_controller"

    def test_phase_scheduler_alternating_ratios(self, phase_configs):
        """Test phase with different update ratios."""
        scheduler = PhaseScheduler(phase_configs["alternating_updates"])

        # Count actions
        action_counts = {"collect": 0, "update_world_model": 0, "update_controller": 0}

        for _ in range(300):
            action = scheduler.next_action()
            if action and action in action_counts:
                action_counts[action] += 1
                if action == "collect":
                    scheduler.advance(action, steps=4)
                else:
                    scheduler.advance(action, updates=1)

            if scheduler.is_finished():
                break

        # Should have ~2x more world_model updates than controller updates
        # (Due to every=1 vs every=2 and updates=2 vs updates=1)
        ratio = action_counts["update_world_model"] / max(action_counts["update_controller"], 1)
        assert ratio >= 3.0  # Should be ~4x due to both frequency and batch updates


class TestCheckpointingAndResumption:
    """Test checkpoint saving and loading."""

    @pytest.fixture
    def trainable_system(self, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Create system and run brief training."""
        # Create components
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        # Create controllers
        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({
            "actor": actor,
            "critic": critic,
        })

        # Create environment
        train_env = GymWrapper(name=cartpole_config.environment.name, num_envs=cartpole_config.environment.num_envs)

        # Create buffer
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        # Create optimizers
        world_model_params = (
            list(encoder.parameters()) +
            list(rssm.parameters()) +
            list(dynamics.parameters()) +
            list(decoder.parameters()) +
            list(reward_predictor.parameters())
        )
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=cartpole_config.algorithm.world_model_lr)

        # Create workflow
        workflow = DreamerWorkflow()

        # Create orchestrator
        orchestrator = WorldModelOrchestrator(
            config=cartpole_config,
            workflow=workflow,
            experiment_dir=temp_experiment_dir,
            components=components,
            optimizers={"world_model": world_model_optimizer},
            train_environment=train_env,
            eval_environment=train_env,
            buffers={"replay": buffer},
            controller_manager=controller_manager,
        )

        return orchestrator, temp_experiment_dir

    @patch("src.utils.logger.create_logger")
    def test_checkpoint_save_and_load(self, mock_create_logger, trainable_system):
        """Test saving and loading checkpoints."""
        mock_logger = MagicMock()
        mock_create_logger.return_value = mock_logger

        orchestrator, exp_dir = trainable_system
        orchestrator.config.training.total_timesteps = 200
        orchestrator.config.training.checkpoint_frequency = 100

        # Run training
        orchestrator.run()

        # Check checkpoint was saved
        checkpoint_dir = exp_dir / "checkpoints"
        assert checkpoint_dir.exists()

        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0

        # Load checkpoint
        final_checkpoint = checkpoint_dir / "final.pt"
        if not final_checkpoint.exists():
            final_checkpoint = checkpoint_files[-1]

        checkpoint_data = torch.load(final_checkpoint)

        # Check checkpoint structure
        assert "workflow" in checkpoint_data
        assert "controllers" in checkpoint_data
        assert "buffers" in checkpoint_data
        assert "world_model_updates" in checkpoint_data

    @patch("src.utils.logger.create_logger")
    def test_workflow_state_dict(self, mock_create_logger, trainable_system):
        """Test workflow state_dict contains all necessary state."""
        mock_logger = MagicMock()
        mock_create_logger.return_value = mock_logger

        orchestrator, _ = trainable_system
        orchestrator.initialize()

        # Run a few updates
        orchestrator.config.training.total_timesteps = 50
        orchestrator.run()

        # Get workflow state
        state = orchestrator.workflow.state_dict(mode="checkpoint")

        # Check state contains optimizer state
        assert "world_model_optimizer" in state
        assert "world_model_updates" in state

        # Check optimizer state is valid
        opt_state = state["world_model_optimizer"]
        assert "state" in opt_state
        assert "param_groups" in opt_state


class TestVectorizedEnvironments:
    """Test handling of vectorized environments."""

    def test_vectorized_collection(self, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Test collection with multiple parallel environments."""
        # Increase num_envs
        cartpole_config.environment.num_envs = 8
        cartpole_config.buffer.num_envs = 8

        # Create minimal workflow
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({"actor": actor, "critic": critic})

        train_env = GymWrapper(name=cartpole_config.environment.name, num_envs=8)
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        world_model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(dynamics.parameters()) + list(decoder.parameters()) + list(reward_predictor.parameters())
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=1e-3)

        checkpoint_manager = CheckpointManager(temp_experiment_dir, auto_save_frequency=0)
        mock_logger = MagicMock()

        context = WorkflowContext(
            config=cartpole_config,
            device="cpu",
            train_environment=train_env,
            eval_environment=train_env,
            components=components,
            checkpoint_manager=checkpoint_manager,
            experiment_logger=mock_logger,
            buffers={"replay": buffer},
            controller_manager=controller_manager,
            optimizers={"world_model": world_model_optimizer},
            initial_observation=train_env.reset(),
            initial_dones=np.zeros(8, dtype=bool),
            global_step=0,
        )

        workflow = DreamerWorkflow()
        workflow.initialize(context)

        # Collect step
        phase = {"type": "online", "collect_length": 4}
        result = workflow.collect_step(step=0, phase=phase)

        # Check vectorized output
        trajectory = result.extras["replay"]["trajectory"]
        assert trajectory["observations"].shape[1] == 8  # num_envs
        assert trajectory["rewards"].shape[1] == 8
        assert result.steps == 32  # 4 steps * 8 envs


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch_handling(self, cartpole_config):
        """Test handling of edge case where buffer is empty."""
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        # Buffer should not be ready
        assert not buffer.ready()

        # Should raise error
        with pytest.raises(ValueError):
            buffer.sample()

    def test_episode_termination_tracking(self, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Test that episodes are tracked correctly with dones."""
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({"actor": actor, "critic": critic})

        train_env = GymWrapper(name=cartpole_config.environment.name, num_envs=4)
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        world_model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(dynamics.parameters()) + list(decoder.parameters()) + list(reward_predictor.parameters())
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=1e-3)

        checkpoint_manager = CheckpointManager(temp_experiment_dir, auto_save_frequency=0)
        mock_logger = MagicMock()

        context = WorkflowContext(
            config=cartpole_config,
            device="cpu",
            train_environment=train_env,
            eval_environment=train_env,
            components=components,
            checkpoint_manager=checkpoint_manager,
            experiment_logger=mock_logger,
            buffers={"replay": buffer},
            controller_manager=controller_manager,
            optimizers={"world_model": world_model_optimizer},
            initial_observation=train_env.reset(),
            initial_dones=np.zeros(4, dtype=bool),
            global_step=0,
        )

        workflow = DreamerWorkflow()
        workflow.initialize(context)

        initial_total_episodes = workflow.total_episodes

        # Collect for a while to ensure some episodes finish
        for _ in range(10):
            phase = {"type": "online", "collect_length": 8}
            result = workflow.collect_step(step=0, phase=phase)

        # Should have completed at least one episode
        assert workflow.total_episodes > initial_total_episodes


@pytest.mark.slow
class TestFullTrainingScenarios:
    """End-to-end tests simulating real training scenarios."""

    @patch("src.utils.logger.create_logger")
    def test_full_dreamer_training_loop(self, mock_create_logger, cartpole_config, simple_reward_predictor, temp_experiment_dir):
        """Full Dreamer training with warmup, joint training, and evaluation."""
        mock_logger = MagicMock()
        mock_create_logger.return_value = mock_logger

        # Configure phases
        cartpole_config.training.total_timesteps = 500
        cartpole_config.training.phases = [
            {
                "name": "warmup",
                "type": "online",
                "duration": {"steps": 200},
                "hooks": [
                    {"collect": {"every": 1, "steps": 1}},
                    {"update_world_model": {"every": 1, "updates": 1}},
                ],
            },
            {
                "name": "joint",
                "type": "online",
                "duration": {"steps": 300},
                "hooks": [
                    {"collect": {"every": 1, "steps": 1}},
                    {"update_world_model": {"every": 1, "updates": 1}},
                    {"update_controller": {"every": 1, "updates": 1}},
                ],
            },
        ]

        # Create system
        encoder = MLPEncoder(**cartpole_config.components.encoder)
        rssm = RSSMRepresentationLearner(**cartpole_config.components.representation_learner)
        dynamics = RSSMRepresentationLearner(**cartpole_config.components.dynamics_model)
        decoder = MLPObservationDecoder(**cartpole_config.components.observation_decoder)
        reward_predictor = simple_reward_predictor(**cartpole_config.components.reward_predictor)

        components = WorldModelComponents(
            encoder=encoder,
            representation_learner=rssm,
            dynamics_model=dynamics,
            observation_decoder=decoder,
            reward_predictor=reward_predictor,
        )

        actor = DreamerActorController(**cartpole_config.controllers.actor)
        critic = DreamerCriticController(**cartpole_config.controllers.critic)
        controller_manager = ControllerManager({"actor": actor, "critic": critic})

        train_env = GymWrapper(name=cartpole_config.environment.name, num_envs=4)
        eval_env = GymWrapper(name=cartpole_config.environment.name, num_envs=1)
        buffer = WorldModelSequenceBuffer(**cartpole_config.buffer)

        world_model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(dynamics.parameters()) + list(decoder.parameters()) + list(reward_predictor.parameters())
        world_model_optimizer = torch.optim.Adam(world_model_params, lr=cartpole_config.algorithm.world_model_lr)

        workflow = DreamerWorkflow()

        orchestrator = WorldModelOrchestrator(
            config=cartpole_config,
            workflow=workflow,
            experiment_dir=temp_experiment_dir,
            components=components,
            optimizers={"world_model": world_model_optimizer},
            train_environment=train_env,
            eval_environment=eval_env,
            buffers={"replay": buffer},
            controller_manager=controller_manager,
        )

        # Run training
        final_metrics = orchestrator.run()

        # Validate training completed
        assert orchestrator.global_step >= 500
        assert workflow.world_model_updates > 0
        assert "wall_time" in final_metrics

        # Check logger was called with correct prefixes
        logged_prefixes = set()
        for call in mock_logger.log_metrics.call_args_list:
            if len(call[0]) > 2:
                prefix = call[0][2] if call[1].get("prefix") is None else call[1]["prefix"]
                logged_prefixes.add(prefix)

        assert "collect" in logged_prefixes or "train" in logged_prefixes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
