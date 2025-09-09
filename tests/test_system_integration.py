import numpy as np
import torch

from src.buffers.trajectory import TrajectoryBuffer
from src.algorithms.random import RandomAgent
from src.environments.base import SpaceSpec


def test_buffer_algorithm_integration():
    buf = TrajectoryBuffer({"capacity": 128, "batch_size": 8})
    # Build a tiny trajectory
    obs = np.zeros(4, dtype=np.float32)
    for t in range(6):
        next_obs = np.ones(4, dtype=np.float32) * (t + 1)
        buf.add(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=next_obs,
            done=(t == 5),
            value=0.0,
            log_prob=0.0,
        )
        obs = next_obs
    assert len(buf) > 0
    batch = buf.sample_all()

    act_space = SpaceSpec(shape=(2,), dtype=np.int64, discrete=True, n=2)
    ob_space = SpaceSpec(shape=(4,), dtype=np.float32)
    agent = RandomAgent({"action_space": act_space, "observation_space": ob_space, "device": "cpu"})
    metrics = agent.update(batch)
    assert "random_step" in metrics
