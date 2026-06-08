from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.components.controllers.dreamer_actor import DreamerActor
from src.components.controllers.dreamer_critic import DreamerCritic


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_dreamer_actor_returns_bounded_deterministic_and_stochastic_actions():
    actor = DreamerActor(
        action_dim=2,
        latent_dim=12,
        hidden_dim=16,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )
    latent = torch.randn(5, 12)

    deterministic = actor.act(latent, deterministic=True)
    stochastic = actor.act(latent, deterministic=False)

    assert deterministic.shape == (5, 2)
    assert stochastic.shape == (5, 2)
    assert torch.all(deterministic >= -1.0)
    assert torch.all(deterministic <= 1.0)
    assert torch.all(stochastic >= -1.0)
    assert torch.all(stochastic <= 1.0)


def test_dreamer_actor_log_prob_is_finite_and_shape_compatible():
    actor = DreamerActor(
        action_dim=3,
        input_dim=10,
        hidden_dims=[16, 16],
        action_low=[-1.0, -1.0, -1.0],
        action_high=[1.0, 1.0, 1.0],
    )
    latent = torch.randn(4, 10)

    action, log_prob = actor.act(latent, return_log_prob=True)

    assert action.shape == (4, 3)
    assert log_prob.shape == (4, 1)
    assert torch.isfinite(log_prob).all()


def test_dreamer_actor_supports_sequence_latents():
    actor = DreamerActor(
        action_dim=2,
        latent_dim=8,
        hidden_dim=16,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )
    latent = torch.randn(3, 5, 8)

    outputs = actor(latent)
    actions = actor.act(latent)

    assert outputs["mean"].shape == (3, 5, 2)
    assert outputs["std"].shape == (3, 5, 2)
    assert actions.shape == (3, 5, 2)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_dreamer_critic_returns_values_for_flat_and_sequence_latents():
    critic = DreamerCritic(input_dim=9, hidden_dims=[16, 16])

    assert critic(torch.randn(4, 9)).shape == (4, 1)
    assert critic(torch.randn(4, 6, 9)).shape == (4, 6, 1)


def test_dreamer_critic_backward_produces_finite_gradients():
    critic = DreamerCritic(latent_dim=7, hidden_dim=16)
    values = critic(torch.randn(5, 7))
    loss = values.pow(2).mean()

    loss.backward()

    grads = [param.grad for param in critic.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_dreamer_controller_yaml_configs_instantiate():
    actor_cfg = OmegaConf.load(CONFIG_DIR / "controller" / "dreamer_actor.yaml")
    actor_cfg.action_dim = 2
    actor_cfg.latent_dim = 11
    actor_cfg.action_low = [-1.0, -1.0]
    actor_cfg.action_high = [1.0, 1.0]
    actor_cfg.device = "cpu"
    actor = instantiate(actor_cfg)

    critic_cfg = OmegaConf.load(CONFIG_DIR / "controller" / "dreamer_critic.yaml")
    critic_cfg.latent_dim = 11
    critic_cfg.device = "cpu"
    critic = instantiate(critic_cfg)

    assert actor.act(torch.randn(2, 11)).shape == (2, 2)
    assert critic(torch.randn(2, 11)).shape == (2, 1)
