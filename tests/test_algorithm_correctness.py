import numpy as np
import torch

from src.algorithms.random import RandomAgent
from src.environments.base import SpaceSpec


def test_random_agent_discrete_action_shape():
    obs = torch.randn(4)
    action_space = SpaceSpec(shape=(2,), dtype=np.int64, discrete=True, n=2)
    obs_space = SpaceSpec(shape=(4,), dtype=np.float32)
    agent = RandomAgent({"action_space": action_space, "observation_space": obs_space, "device": "cpu"})
    a = agent.act(obs)
    assert isinstance(a.item(), int)


def test_random_agent_continuous_action_shape():
    obs = torch.randn(1, 4)
    # Provide bounds so RandomAgent can sample within [low, high]
    action_space = SpaceSpec(shape=(3,), dtype=np.float32, discrete=False, low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]))
    obs_space = SpaceSpec(shape=(4,), dtype=np.float32)
    agent = RandomAgent({"action_space": action_space, "observation_space": obs_space, "device": "cpu"})
    a = agent.act(obs)
    assert a.shape[-1] == 3
