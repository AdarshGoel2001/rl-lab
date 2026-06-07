from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from scripts.validate_experiment import validate_experiment_config
from src.components.controllers.mpc_planner import MPCPlanner
from src.components.prediction_heads.mlp import MLPHead
from src.components.representation_learners.base import RSSMState
from src.components.representation_learners.rssm import RSSMRepresentationLearner
from src.workflows.utils.context import WorkflowContext, WorldModelComponents
from src.workflows.planet import PlaNetWorkflow


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_mlp_head_forward_backward_with_sequence_inputs():
    head = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    inputs = torch.randn(2, 4, 12)

    preds = head(inputs)
    loss = preds.pow(2).mean()

    assert preds.shape == (2, 4, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(param.grad is not None for param in head.parameters())


def test_mpc_planner_instantiates_with_buffers():
    planner = MPCPlanner(representation_dim=12, action_dim=2, horizon=3, num_samples=8, top_k=2)

    assert planner.action_mean.shape == (3, 2)
    assert planner.action_std.shape == (3, 2)


def test_planet_workflow_state_round_trips_world_model_update_count():
    workflow = PlaNetWorkflow()
    workflow.world_model_updates = 17

    state = workflow.get_state()

    restored = PlaNetWorkflow()
    restored.set_state(state)

    assert restored.world_model_updates == 17


def test_mpc_planner_scores_rollouts_with_reward_sum_by_default():
    planner = MPCPlanner(representation_dim=12, action_dim=2, horizon=3, num_samples=2, top_k=1)
    rewards = torch.ones(3, 1)
    bootstrap = torch.tensor([10.0])

    value = planner._score_rollout(
        {
            "rewards": rewards,
            "continues": torch.ones(3, 1),
            "bootstrap": bootstrap,
        },
        gamma=0.99,
    )

    assert torch.isclose(value, torch.tensor(3.0))


def test_mpc_planner_scores_rollouts_with_continuation_when_enabled():
    planner = MPCPlanner(
        representation_dim=12,
        action_dim=2,
        horizon=3,
        num_samples=2,
        top_k=1,
        use_continuation=True,
    )
    rewards = torch.ones(3, 1)
    bootstrap = torch.tensor([10.0])

    terminating = planner._score_rollout(
        {
            "rewards": rewards,
            "continues": torch.tensor([[0.0], [1.0], [1.0]]),
            "bootstrap": bootstrap,
        },
        gamma=0.99,
    )

    assert torch.isclose(terminating, torch.tensor(1.0))


def test_planet_workflow_update_and_imagine_on_fake_batch():
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    reward = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    continuation = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(
        list(rssm.parameters()) + list(reward.parameters()) + list(continuation.parameters()),
        lr=1e-3,
    )

    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.rssm = rssm
    workflow.reward_predictor = reward
    workflow.continue_predictor = continuation
    workflow.world_model_optimizer = optimizer
    workflow.action_dim = 2
    workflow.gamma = 0.99
    workflow.free_nats = 0.0
    workflow.kl_scale = 1.0

    batch = {
        "observations": torch.randn(3, 5, 4),
        "actions": torch.nn.functional.one_hot(torch.randint(0, 2, (3, 5)), num_classes=2).float(),
        "rewards": torch.randn(3, 5),
        "dones": torch.zeros(3, 5, dtype=torch.bool),
    }

    metrics = workflow.update_world_model(batch, phase={"name": "train_world_model"})
    start = rssm.initial_state(batch_size=2)
    actions = torch.zeros(2, 3, 2)
    actions[..., 0] = 1.0
    rollout = workflow.imagine(latent=start, horizon=3, action_sequence=actions, deterministic=True)

    assert metrics["world_model/total_loss"] > 0.0
    assert rollout["rewards"].shape == (2, 3, 1)
    assert rollout["continues"].shape == (2, 3, 1)
    assert rollout["bootstrap"].shape == (2, 1)


def test_planet_world_model_trains_without_continue_predictor():
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    reward = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(list(rssm.parameters()) + list(reward.parameters()), lr=1e-3)

    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.rssm = rssm
    workflow.reward_predictor = reward
    workflow.continue_predictor = None
    workflow.world_model_optimizer = optimizer
    workflow.action_dim = 2
    workflow.free_nats = 0.0
    workflow.kl_scale = 1.0

    batch = {
        "observations": torch.randn(2, 4, 4),
        "actions": torch.nn.functional.one_hot(torch.randint(0, 2, (2, 4)), num_classes=2).float(),
        "rewards": torch.randn(2, 4),
        "dones": torch.zeros(2, 4, dtype=torch.bool),
    }

    metrics = workflow.update_world_model(batch, phase={"name": "train_world_model"})

    assert metrics["world_model/total_loss"] > 0.0
    assert "world_model/continue_loss" not in metrics


def test_planet_world_model_uses_configurable_loss_scales():
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    reward = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    continuation = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(
        list(rssm.parameters()) + list(reward.parameters()) + list(continuation.parameters()),
        lr=1e-3,
    )

    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.rssm = rssm
    workflow.reward_predictor = reward
    workflow.continue_predictor = continuation
    workflow.world_model_optimizer = optimizer
    workflow.action_dim = 2
    workflow.free_nats = 0.0
    workflow.kl_scale = 0.5
    workflow.reward_loss_scale = 10.0
    workflow.continue_loss_scale = 2.0
    workflow.observation_loss_scale = 1.0

    batch = {
        "observations": torch.randn(2, 4, 4),
        "actions": torch.nn.functional.one_hot(torch.randint(0, 2, (2, 4)), num_classes=2).float(),
        "rewards": torch.randn(2, 4),
        "dones": torch.zeros(2, 4, dtype=torch.bool),
    }

    metrics = workflow.update_world_model(batch, phase={"name": "train_world_model"})
    expected_total = (
        10.0 * metrics["world_model/reward_loss"]
        + 0.5 * metrics["world_model/kl_loss"]
        + 2.0 * metrics["world_model/continue_loss"]
    )

    assert metrics["world_model/reward_loss_scale"] == 10.0
    assert metrics["world_model/continue_loss_scale"] == 2.0
    assert abs(metrics["world_model/total_loss"] - expected_total) < 1e-5


def test_planet_binds_and_trains_observation_predictor_when_present():
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    reward = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    observation = MLPHead(input_dim=12, output_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(
        list(rssm.parameters()) + list(reward.parameters()) + list(observation.parameters()),
        lr=1e-3,
    )
    env = SimpleNamespace(num_envs=1)
    context = WorkflowContext(
        config=SimpleNamespace(
            _dims=SimpleNamespace(action=2),
            algorithm=SimpleNamespace(collect_length=1, gamma=0.99, free_nats=0.0, kl_scale=1.0),
        ),
        device="cpu",
        train_environment=env,
        eval_environment=env,
        components=WorldModelComponents(
            {
                "representation_learner": rssm,
                "reward_predictor": reward,
                "observation_predictor": observation,
            }
        ),
        checkpoint_manager=None,
        experiment_logger=None,
        optimizers={"world_model": optimizer},
        initial_observation=torch.zeros(1, 4).numpy(),
        initial_dones=torch.zeros(1, dtype=torch.bool).numpy(),
    )
    workflow = PlaNetWorkflow()
    workflow.initialize(context)

    batch = {
        "observations": torch.randn(2, 4, 4),
        "actions": torch.nn.functional.one_hot(torch.randint(0, 2, (2, 4)), num_classes=2).float(),
        "rewards": torch.randn(2, 4),
        "dones": torch.zeros(2, 4, dtype=torch.bool),
    }

    metrics = workflow.update_world_model(batch, phase={"name": "train_world_model"})

    assert workflow.observation_predictor is observation
    assert metrics["world_model/observation_loss"] > 0.0
    assert any(param.grad is not None for param in observation.parameters())


def test_planet_imagine_scores_successor_state_after_candidate_action():
    class ActionCopyRSSM:
        deterministic_dim = 1
        stochastic_dim = 1
        min_std = 0.1

        def imagine_step(self, state, action, *, deterministic=False):
            del deterministic
            value = action[:, :1]
            std = torch.ones_like(value) * self.min_std
            return RSSMState(deterministic=value, stochastic=torch.zeros_like(value), mean=torch.zeros_like(value), std=std)

    class DeterministicReward(nn.Module):
        def forward(self, latent):
            return latent[..., :1]

    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.rssm = ActionCopyRSSM()
    workflow.reward_predictor = DeterministicReward()
    workflow.continue_predictor = None

    start = RSSMState(
        deterministic=torch.zeros(1, 1),
        stochastic=torch.zeros(1, 1),
        mean=torch.zeros(1, 1),
        std=torch.ones(1, 1) * 0.1,
    )
    actions = torch.tensor([[[0.25], [0.75]]])

    rollout = workflow.imagine(latent=start, horizon=2, action_sequence=actions, deterministic=True)

    assert torch.allclose(rollout["rewards"].squeeze(0), actions.squeeze(0))
    assert "continues" not in rollout
    assert torch.allclose(rollout["bootstrap"], torch.tensor([[0.75]]))


def test_mpc_planner_does_not_warm_start_cem_across_act_calls():
    class RewardByActionWorkflow:
        gamma = 0.99

        def get_action_bounds(self):
            return torch.zeros(1), torch.ones(1)

        def imagine(self, *, latent, horizon, action_sequence):
            del latent, horizon
            return {
                "rewards": action_sequence.clone(),
                "bootstrap": torch.zeros(action_sequence.shape[0], 1),
            }

    planner = MPCPlanner(
        representation_dim=2,
        action_dim=1,
        horizon=2,
        num_samples=8,
        top_k=2,
        iterations=1,
        std_init=0.5,
    )
    initial_mean = planner.action_mean.clone()
    initial_std = planner.action_std.clone()

    planner.act(torch.zeros(1, 2), workflow=RewardByActionWorkflow())

    assert torch.allclose(planner.action_mean, initial_mean)
    assert torch.allclose(planner.action_std, initial_std)


def test_mpc_planner_checkpoint_state_excludes_cem_working_distribution():
    planner = MPCPlanner(
        representation_dim=2,
        action_dim=1,
        horizon=8,
        num_samples=6,
        top_k=2,
        iterations=2,
    )

    state = planner.state_dict()

    assert "action_mean" not in state
    assert "action_std" not in state


def test_mpc_planner_load_ignores_legacy_cem_buffers_with_different_horizon():
    planner = MPCPlanner(
        representation_dim=2,
        action_dim=1,
        horizon=12,
        num_samples=6,
        top_k=2,
        iterations=2,
    )
    legacy_state = {
        "action_mean": torch.zeros(8, 1),
        "action_std": torch.ones(8, 1),
    }

    incompatible = planner.load_state_dict(legacy_state)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert planner.action_mean.shape == (12, 1)
    assert planner.action_std.shape == (12, 1)


def test_mpc_planner_batches_candidate_imagination_per_iteration():
    class CountingWorkflow:
        gamma = 0.99

        def __init__(self):
            self.calls = 0

        def get_action_bounds(self):
            return torch.zeros(1), torch.ones(1)

        def imagine(self, *, latent, horizon, action_sequence):
            self.calls += 1
            assert latent.shape[0] == action_sequence.shape[0]
            assert action_sequence.shape == (6, horizon, 1)
            return {
                "rewards": action_sequence.clone(),
                "bootstrap": torch.zeros(action_sequence.shape[0], 1),
            }

    workflow = CountingWorkflow()
    planner = MPCPlanner(
        representation_dim=2,
        action_dim=1,
        horizon=2,
        num_samples=6,
        top_k=2,
        iterations=2,
        std_init=0.5,
    )

    planner.act(torch.zeros(1, 2), workflow=workflow)

    assert workflow.calls == 2


def test_mpc_planner_batches_candidates_across_parallel_envs():
    class CountingWorkflow:
        gamma = 0.99

        def __init__(self):
            self.calls = 0

        def get_action_bounds(self):
            return torch.zeros(1), torch.ones(1)

        def imagine(self, *, latent, horizon, action_sequence):
            self.calls += 1
            assert latent.shape == (15, 2)
            assert action_sequence.shape == (15, horizon, 1)
            return {
                "rewards": action_sequence.clone(),
                "bootstrap": torch.zeros(action_sequence.shape[0], 1),
            }

    workflow = CountingWorkflow()
    planner = MPCPlanner(
        representation_dim=2,
        action_dim=1,
        horizon=2,
        num_samples=5,
        top_k=2,
        iterations=2,
        std_init=0.5,
    )

    action = planner.act(torch.zeros(3, 2), workflow=workflow)

    assert workflow.calls == 2
    assert action.shape == (3, 1)


def test_mpc_planner_uses_planet_imagination():
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.rssm = rssm
    workflow.reward_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.continue_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.gamma = 0.99

    planner = MPCPlanner(
        representation_dim=12,
        action_dim=2,
        horizon=3,
        num_samples=4,
        top_k=2,
        iterations=1,
    )
    latent = rssm.initial_state(batch_size=2)

    action = planner.act(latent.to_tensor(), workflow=workflow)

    assert action.shape == (2, 2)
    assert torch.all(action >= 0.0)
    assert torch.all(action <= 1.0)


def test_planet_evaluate_logs_planner_return_with_fake_env():
    class FakeEnv:
        num_envs = 1

        def __init__(self):
            self.steps = 0

        def reset(self, seed=None):
            self.steps = 0
            return torch.zeros(1, 4).numpy()

        def step(self, action):
            self.steps += 1
            done = self.steps >= 3
            return (
                torch.zeros(1, 4).numpy(),
                torch.ones(1).numpy(),
                torch.tensor([done]).numpy(),
                [{}],
            )

    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.environment = FakeEnv()
    workflow.eval_environment = workflow.environment
    workflow.rssm = rssm
    workflow.reward_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.continue_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.controllers = {
        "planner": MPCPlanner(
            representation_dim=12,
            action_dim=2,
            horizon=2,
            num_samples=4,
            top_k=2,
            iterations=1,
        )
    }
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.gamma = 0.99

    metrics = workflow.evaluate(num_eval_batches=2, max_steps_per_episode=5)

    assert metrics["return_mean"] == 3.0
    assert metrics["length_mean"] == 3.0
    assert metrics["episodes"] == 2.0
    assert metrics["eval_episode_batches"] == 2.0
    assert metrics["eval_num_envs"] == 1.0
    assert metrics["eval_total_episodes"] == 2.0


def test_planet_evaluate_logs_vectorized_episode_multiplication():
    class FakeVectorEnv:
        num_envs = 3
        is_vectorized = True

        def __init__(self):
            self.steps = 0

        def reset(self, seed=None):
            self.steps = 0
            return torch.zeros(3, 4).numpy()

        def step(self, action):
            self.steps += 1
            done = self.steps >= 2
            return (
                torch.zeros(3, 4).numpy(),
                torch.ones(3).numpy(),
                torch.tensor([done, done, done]).numpy(),
                [{}, {}, {}],
            )

    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.environment = FakeVectorEnv()
    workflow.eval_environment = workflow.environment
    workflow.rssm = rssm
    workflow.reward_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.continue_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.controllers = {
        "planner": MPCPlanner(
            representation_dim=12,
            action_dim=2,
            horizon=2,
            num_samples=4,
            top_k=2,
            iterations=1,
        )
    }
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.gamma = 0.99

    metrics = workflow.evaluate(num_eval_batches=2, max_steps_per_episode=5)

    assert metrics["eval_episode_batches"] == 2.0
    assert metrics["eval_num_envs"] == 3.0
    assert metrics["eval_total_episodes"] == 6.0
    assert metrics["episodes"] == 6.0
    assert metrics["return_mean"] == 2.0


def test_planet_collect_step_can_use_planner_controller():
    class FakeEnv:
        num_envs = 2

        def __init__(self):
            self.last_action = None

        def step(self, action):
            self.last_action = action
            return (
                torch.zeros(2, 4).numpy(),
                torch.ones(2).numpy(),
                torch.zeros(2, dtype=torch.bool).numpy(),
                [{}, {}],
            )

    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    env = FakeEnv()
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.environment = env
    workflow.current_obs = torch.zeros(2, 4).numpy()
    workflow.current_dones = torch.zeros(2, dtype=torch.bool).numpy()
    workflow.rssm = rssm
    workflow.reward_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.continue_predictor = MLPHead(input_dim=12, output_dim=1, hidden_dim=16)
    workflow.controllers = {
        "planner": MPCPlanner(
            representation_dim=12,
            action_dim=2,
            horizon=2,
            num_samples=4,
            top_k=2,
            iterations=1,
        )
    }
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.num_envs = 2
    workflow._reset_episode_tracking(2, clear_history=True)

    result = workflow.collect_step(0, phase={"collect_length": 1, "controller": "planner"})

    assert result.steps == 2
    assert env.last_action.shape == (2, 2)
    assert result.trajectory["actions"].shape == (1, 2, 2)


def test_planet_collect_step_adds_planner_exploration_noise_and_clamps():
    class FakeRSSM:
        def observe(self, obs, prev_action, reset_mask, *, detach_posteriors):
            del prev_action, reset_mask, detach_posteriors
            return SimpleNamespace(posterior=SimpleNamespace(to_tensor=lambda: torch.zeros(obs.shape[0], 2)))

    class ConstantPlanner:
        def act(self, latent, *, workflow):
            del workflow
            return torch.full((latent.shape[0], 2), 0.5)

    class FakeEnv:
        num_envs = 1

        def __init__(self):
            self.last_action = None

        def step(self, action):
            self.last_action = action.detach().clone()
            return (
                torch.zeros(1, 4).numpy(),
                torch.ones(1).numpy(),
                torch.zeros(1, dtype=torch.bool).numpy(),
                {},
            )

    env = FakeEnv()
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.environment = env
    workflow.current_obs = torch.zeros(1, 4).numpy()
    workflow.current_dones = torch.zeros(1, dtype=torch.bool).numpy()
    workflow.prev_action = torch.zeros(1, 2)
    workflow.rssm = FakeRSSM()
    workflow.controllers = {"planner": ConstantPlanner()}
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.num_envs = 1
    workflow._reset_episode_tracking(1, clear_history=True)

    original_randn_like = torch.randn_like
    torch.randn_like = lambda tensor: torch.ones_like(tensor)
    try:
        result = workflow.collect_step(0, phase={"collect_length": 1, "controller": "planner", "exploration_noise": 0.75})
    finally:
        torch.randn_like = original_randn_like

    assert torch.allclose(env.last_action, torch.ones(1, 2))
    assert result.trajectory["actions"].shape == (1, 1, 2)
    assert torch.allclose(torch.as_tensor(result.trajectory["actions"][0]), torch.ones(1, 2))


def test_planet_collect_step_resets_single_env_after_done():
    class FakeRSSM:
        def observe(self, obs, prev_action, reset_mask, *, detach_posteriors):
            del prev_action, reset_mask, detach_posteriors
            return SimpleNamespace(posterior=SimpleNamespace(to_tensor=lambda: torch.zeros(obs.shape[0], 2)))

    class ZeroPlanner:
        def act(self, latent, *, workflow):
            del workflow
            return torch.zeros(latent.shape[0], 2)

    class DoneEnv:
        num_envs = 1

        def __init__(self):
            self.steps = 0
            self.resets = 0
            self.stepped_after_done = False
            self.done = False

        def reset(self, seed=None):
            del seed
            self.resets += 1
            self.done = False
            return torch.full((1, 4), 7.0).numpy()

        def step(self, action):
            del action
            if self.done:
                self.stepped_after_done = True
            self.steps += 1
            self.done = True
            return (
                torch.full((1, 4), float(self.steps)).numpy(),
                torch.ones(1).numpy(),
                torch.ones(1, dtype=torch.bool).numpy(),
                {},
            )

    env = DoneEnv()
    workflow = PlaNetWorkflow()
    workflow.device = "cpu"
    workflow.environment = env
    workflow.current_obs = torch.zeros(1, 4).numpy()
    workflow.current_dones = torch.zeros(1, dtype=torch.bool).numpy()
    workflow.prev_action = torch.zeros(1, 2)
    workflow.rssm = FakeRSSM()
    workflow.controllers = {"planner": ZeroPlanner()}
    workflow.action_dim = 2
    workflow.action_low = torch.zeros(2)
    workflow.action_high = torch.ones(2)
    workflow.num_envs = 1
    workflow._reset_episode_tracking(1, clear_history=True)

    result = workflow.collect_step(0, phase={"collect_length": 2, "controller": "planner"})

    assert result.steps == 2
    assert env.resets == 2
    assert env.stepped_after_done is False
    assert result.trajectory["dones"].shape == (2, 1)
    assert result.trajectory["observations"][1].tolist() == [[7.0, 7.0, 7.0, 7.0]]


def test_planet_cartpole_config_resolves_with_tiny_budget():
    result = validate_experiment_config("planet_cartpole", budget="planet_tiny")

    assert result["ok"] is True
    assert result["targets"]["workflow"] == "src.workflows.planet.PlaNetWorkflow"
    assert (
        result["targets"]["components.representation_learner"]
        == "src.components.representation_learners.rssm.RSSMRepresentationLearner"
    )
    assert result["targets"]["components.reward_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["targets"]["components.continue_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["phases"] == ["data_collection", "train_world_model", "planning_eval"]


def test_planet_cartpole_config_swaps_planner_dimensions():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=planet_cartpole", "budget=planet_tiny"])

    assert cfg.controllers.actor.action_dim == 2
    assert cfg.controllers.planner.action_dim == 2
    assert cfg.controllers.planner.representation_dim == 12
