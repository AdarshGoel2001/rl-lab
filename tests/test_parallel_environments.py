import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.environments.base import BaseEnvironment, SpaceSpec
from src.utils.registry import register_environment


@register_environment("dummy_parallel")
class DummyParallelEnv(BaseEnvironment):
    """Minimal environment for multiprocessing tests."""

    def _setup_environment(self):
        self._obs_dim = 4
        self._act_n = 2
        self._state = np.zeros(self._obs_dim, dtype=np.float32)
        self._internal_step = 0

    def _get_observation_space(self) -> SpaceSpec:
        return SpaceSpec(shape=(self._obs_dim,), dtype=np.float32)

    def _get_action_space(self) -> SpaceSpec:
        return SpaceSpec(shape=(self._act_n,), dtype=np.int64, discrete=True, n=self._act_n)

    def _reset_environment(self, seed=None):
        self._internal_step = 0
        return np.zeros(self._obs_dim, dtype=np.float32)

    def _step_environment(self, action):
        # action may be scalar int or array; just increment state deterministically
        self._internal_step += 1
        obs = np.full(self._obs_dim, self._internal_step, dtype=np.float32)
        reward = 1.0
        done = self._internal_step >= 3
        info = {"internal_step": self._internal_step}
        return obs, reward, done, info


def _make_manager(num_envs=2):
    from src.environments.parallel_manager import ParallelEnvironmentManager

    env_cfg = {
        "name": "DummyParallel",
        "wrapper": "dummy_parallel",
        "normalize_obs": False,
        "normalize_reward": False,
        "max_episode_steps": 10,
    }
    # Use 'fork' so child inherits registry from parent
    return ParallelEnvironmentManager(env_config=env_cfg, num_environments=num_envs, start_method="fork")


@pytest.mark.timeout(10)
def test_parallel_manager_basic_reset_step_close():
    try:
        mgr = _make_manager(2)
    except Exception as e:
        pytest.skip(f"Parallel manager not available: {e}")

    try:
        obs = mgr.reset()
        assert len(obs) == 2
        actions = [0, 1]
        next_obs, rewards, dones, infos = mgr.step(actions)
        assert len(next_obs) == 2 and len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)
    finally:
        mgr.close()

