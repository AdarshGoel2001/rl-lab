import numpy as np
import torch
import torch.nn as nn

from src.components.prediction_heads.mlp import MLPHead
from src.components.representation_learners.base import RSSMState
from src.components.representation_learners.rssm import RSSMRepresentationLearner
from src.workflows.utils.base import CollectResult
from src.workflows.dreamer import DreamerV1Workflow, lambda_returns


class TinyActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, action_dim)
        self.calls: list[dict[str, bool]] = []

    def act(self, latent_tensor, deterministic=False, return_log_prob=False):
        self.calls.append(
            {
                "deterministic": bool(deterministic),
                "return_log_prob": bool(return_log_prob),
            }
        )
        action = torch.tanh(self.net(latent_tensor))
        if return_log_prob:
            return action, torch.zeros(action.shape[:-1], device=action.device)
        return action


class StepCountingOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _make_workflow():
    torch.manual_seed(0)
    rssm = RSSMRepresentationLearner(
        feature_dim=4,
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        hidden_dim=16,
        min_std=0.1,
        device="cpu",
    )
    latent_dim = rssm.representation_dim
    reward = MLPHead(input_dim=latent_dim, output_dim=1, hidden_dim=16)
    continuation = MLPHead(input_dim=latent_dim, output_dim=1, hidden_dim=16)
    observation = MLPHead(input_dim=latent_dim, output_dim=4, hidden_dim=16)
    actor = TinyActor(latent_dim, 2)
    critic = nn.Linear(latent_dim, 1)
    world_params = list(rssm.parameters()) + list(reward.parameters()) + list(continuation.parameters()) + list(
        observation.parameters()
    )
    workflow = DreamerV1Workflow()
    workflow.device = "cpu"
    workflow.rssm = rssm
    workflow.reward_predictor = reward
    workflow.continue_predictor = continuation
    workflow.observation_predictor = observation
    workflow.controllers = {"actor": actor, "critic": critic}
    workflow.world_model_optimizer = StepCountingOptimizer(world_params, lr=1e-3)
    workflow.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    workflow.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    workflow.action_dim = 2
    workflow.gamma = 0.99
    workflow.lambda_ = 0.95
    workflow.free_nats = 0.0
    workflow.kl_scale = 1.0
    workflow.imagination_horizon = 3
    return workflow


def _fake_batch(batch_size=3, horizon=5):
    return {
        "observations": torch.randn(batch_size, horizon, 4),
        "actions": torch.randn(batch_size, horizon, 2).tanh(),
        "rewards": torch.randn(batch_size, horizon),
        "dones": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }


def _params_clone(module: nn.Module):
    return [param.detach().clone() for param in module.parameters()]


def _params_changed(before, module: nn.Module) -> bool:
    return any(not torch.allclose(old, new.detach()) for old, new in zip(before, module.parameters()))


def test_lambda_returns_accumulates_simple_discounted_case():
    rewards = torch.ones(1, 3, 1)
    values = torch.zeros(1, 3, 1)
    continues = torch.ones(1, 3, 1)
    bootstrap = torch.zeros(1, 1)

    returns = lambda_returns(rewards, values, continues, bootstrap, discount=1.0, lambda_=1.0)

    assert torch.allclose(returns, torch.tensor([[[3.0], [2.0], [1.0]]]))


def test_imagined_rollout_returns_expected_shapes():
    workflow = _make_workflow()
    start = workflow.rssm.initial_state(batch_size=2)

    rollout = workflow.imagine_rollout(start, horizon=4, deterministic_actor=True, deterministic_model=True)

    assert rollout["states"].shape == (2, 4, 12)
    assert rollout["actions"].shape == (2, 4, 2)
    assert rollout["rewards"].shape == (2, 4, 1)
    assert rollout["continues"].shape == (2, 4, 1)
    assert rollout["values"].shape == (2, 4, 1)
    assert rollout["bootstrap"].shape == (2, 1)


def test_imagined_rollout_bootstrap_uses_one_extra_state():
    class ZeroActor(nn.Module):
        def act(self, latent_tensor, deterministic=False, return_log_prob=False):
            del deterministic, return_log_prob
            return torch.zeros(latent_tensor.shape[:-1] + (1,), device=latent_tensor.device)

    class StateValue(nn.Module):
        def forward(self, latent_tensor):
            return latent_tensor[..., :1]

    class CounterRSSM(nn.Module):
        deterministic_dim = 1
        stochastic_dim = 1
        min_std = 0.1

        @property
        def representation_dim(self):
            return 2

        def initial_state(self, batch_size):
            zeros = torch.zeros(batch_size, 1)
            return RSSMState(
                deterministic=zeros,
                stochastic=zeros,
                mean=zeros,
                std=torch.ones_like(zeros) * self.min_std,
            )

        def imagine_step(self, state, action, *, deterministic=False):
            del action, deterministic
            deterministic_state = state.deterministic + 1.0
            return RSSMState(
                deterministic=deterministic_state,
                stochastic=state.stochastic,
                mean=state.mean,
                std=state.std,
            )

    workflow = DreamerV1Workflow()
    workflow.device = "cpu"
    workflow.rssm = CounterRSSM()
    workflow.reward_predictor = StateValue()
    workflow.continue_predictor = None
    workflow.controllers = {"actor": ZeroActor(), "critic": StateValue()}

    rollout = workflow.imagine_rollout(workflow.rssm.initial_state(batch_size=1), horizon=3, deterministic_actor=True)

    assert torch.allclose(rollout["values"].squeeze(-1), torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(rollout["bootstrap"], torch.tensor([[4.0]]))


def test_dreamer_world_model_update_returns_finite_losses():
    workflow = _make_workflow()

    metrics = workflow.update_world_model(_fake_batch(), phase={"name": "train_world_model"})

    assert metrics["world_model/total_loss"] > 0.0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_controller_update_changes_actor_and_critic_without_world_model_step():
    workflow = _make_workflow()
    actor_before = _params_clone(workflow.controllers["actor"])
    critic_before = _params_clone(workflow.controllers["critic"])
    world_step_calls = workflow.world_model_optimizer.step_calls

    metrics = workflow.update_controller(_fake_batch(), phase={"horizon": 3})

    assert metrics["controller/actor_loss"] != 0.0
    assert metrics["controller/critic_loss"] >= 0.0
    assert _params_changed(actor_before, workflow.controllers["actor"])
    assert _params_changed(critic_before, workflow.controllers["critic"])
    assert workflow.world_model_optimizer.step_calls == world_step_calls


def test_dreamer_evaluate_uses_deterministic_actor_not_collect_step():
    class FakeEnv:
        num_envs = 1

        def __init__(self):
            self.steps = 0
            self.actions = []

        def reset(self, seed=None):
            del seed
            self.steps = 0
            self.actions.clear()
            return np.zeros((1, 4), dtype=np.float32)

        def step(self, action):
            self.steps += 1
            self.actions.append(np.asarray(action))
            done = self.steps >= 2
            return (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.asarray([done]),
                [{}],
            )

    workflow = _make_workflow()
    env = FakeEnv()
    workflow.environment = env
    workflow.eval_environment = env
    workflow.collect_step = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("collect_step used"))

    metrics = workflow.evaluate(num_eval_batches=1, max_steps_per_episode=5, deterministic=True)

    assert metrics["return_mean"] == 2.0
    assert len(env.actions) == 2
    assert workflow.controllers["actor"].calls
    assert all(call["deterministic"] for call in workflow.controllers["actor"].calls)


def test_dreamer_collect_with_actor_observes_latent_not_raw_observation():
    class RecordingActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def act(self, latent_tensor, deterministic=False, return_log_prob=False):
            del deterministic, return_log_prob
            self.inputs.append(latent_tensor.detach().clone())
            if latent_tensor.shape[-1] != 12:
                raise AssertionError(f"actor received raw observation shape {latent_tensor.shape}")
            return torch.zeros(latent_tensor.shape[0], 2)

    class FakeEnv:
        num_envs = 1
        is_vectorized = False

        def __init__(self):
            self.steps = 0

        def step(self, action):
            self.steps += 1
            assert np.asarray(action).shape == (1, 2)
            return (
                np.ones((1, 4), dtype=np.float32) * self.steps,
                np.asarray([1.0], dtype=np.float32),
                np.asarray([False]),
                [{}],
            )

    workflow = _make_workflow()
    actor = RecordingActor()
    workflow.controllers["actor"] = actor
    workflow.environment = FakeEnv()
    workflow.current_obs = np.zeros((1, 4), dtype=np.float32)
    workflow.current_dones = np.asarray([False])
    workflow.prev_action = torch.zeros(1, 2)
    workflow.num_envs = 1

    result = workflow.collect_step(0, phase={"controller": "actor", "collect_length": 2})

    assert isinstance(result, CollectResult)
    assert result.steps == 2
    assert len(actor.inputs) == 2
    assert all(input_tensor.shape == (1, 12) for input_tensor in actor.inputs)
