#!/usr/bin/env python3
"""
Test PPO Paradigm System

Tests the new architecture where PPO paradigm inherits from ModelFreeParadigm
and is registered as an algorithm, providing full trainer compatibility.
"""

import sys
import torch
import numpy as np
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.registry import auto_import_modules, get_algorithm
from src.paradigms.factory import ComponentFactory

class MockActionSpace:
    def __init__(self, shape, discrete=False):
        self.shape = shape
        self.discrete = discrete
        self.n = shape[0] if discrete else None

class MockObservationSpace:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture(scope="function")
def ppo():
    """Construct a PPO paradigm instance with modular components."""
    auto_import_modules()

    obs_space = MockObservationSpace((4,))
    action_space = MockActionSpace((2,), discrete=False)

    encoder = ComponentFactory.create_component('encoder', 'mlp', {
        'input_dim': obs_space.shape,
        'hidden_dims': [32, 32],
        'activation': 'tanh',
        'device': 'cpu'
    })

    representation_learner = ComponentFactory.create_component('representation_learner', 'identity', {
        'representation_dim': getattr(encoder, 'output_dim', None),
        'device': 'cpu'
    })

    policy_head = ComponentFactory.create_component('policy_head', 'gaussian_mlp', {
        'representation_dim': getattr(encoder, 'output_dim', None),
        'action_dim': int(np.prod(action_space.shape)),
        'device': 'cpu'
    })

    value_function = ComponentFactory.create_component('value_function', 'critic_mlp', {
        'representation_dim': getattr(encoder, 'output_dim', None),
        'hidden_dims': [32, 32],
        'device': 'cpu'
    })

    config = {
        'components': {
            'encoder': encoder,
            'representation_learner': representation_learner,
            'policy_head': policy_head,
            'value_function': value_function,
        },
        'observation_space': MockObservationSpace((4,)),
        'action_space': MockActionSpace((2,), discrete=False),
        'device': 'cpu',
        'lr': 3e-4,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 2,
        'minibatch_size': 32,
        'normalize_advantages': True,
        'clip_value_loss': True,
        'max_grad_norm': 0.5
    }

    ppo_class = get_algorithm('ppo')
    instance = ppo_class(config)
    return instance


def test_ppo_paradigm_registration():
    auto_import_modules()
    ppo_class = get_algorithm('ppo')
    assert ppo_class.__name__ == 'PPOParadigm'


def test_trainer_compatibility(ppo):
    required_methods = ['get_action_and_value', 'act', 'update', 'train', 'eval']
    for method in required_methods:
        assert hasattr(ppo, method), f"Missing method: {method}"

    required_attrs = ['networks', 'device', 'step']
    for attr in required_attrs:
        assert hasattr(ppo, attr), f"Missing attribute: {attr}"

    assert 'critic' in ppo.networks


def test_forward_pass(ppo):
    batch_size = 3
    obs = torch.randn(batch_size, 4)
    actions, log_probs, values = ppo.get_action_and_value(obs)

    assert actions.shape == (batch_size, 2)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)

    det_actions = ppo.act(obs, deterministic=True)
    assert det_actions.shape == (batch_size, 2)

    stoch_actions = ppo.act(obs, deterministic=False)
    assert stoch_actions.shape == (batch_size, 2)


def test_training_modes(ppo):
    ppo.train()
    assert ppo.encoder.training
    assert ppo.policy_head.training
    assert ppo.value_function.training

    ppo.eval()
    assert not ppo.encoder.training
    assert not ppo.policy_head.training
    assert not ppo.value_function.training


def test_update_method(ppo):
    batch_size = 64
    observations = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    batch = {
        'observations': observations,
        'actions': actions,
        'old_log_probs': old_log_probs,
        'advantages': advantages,
        'returns': returns,
    }

    metrics = ppo.update(batch)
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'total_loss' in metrics


def test_bootstrap_values(ppo):
    batch_size = 8
    obs = torch.randn(batch_size, 4)
    values = ppo.critic_network(obs)
    assert values.shape == (batch_size, 1)
