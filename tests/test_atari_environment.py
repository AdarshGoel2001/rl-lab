import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments.atari_wrapper import AtariEnvironment


BASE_CONFIG = {
    "game": "ALE/Pong-v5",
    "wrapper": "atari",
    "frame_skip": 4,
    "frame_stack": 4,
    "sticky_actions": 0.25,
    "noop_max": 30,
    "terminal_on_life_loss": False,
    "clip_rewards": True,
    "full_action_space": False,
}


def test_atari_environment_observation_range_and_dtype():
    config = dict(BASE_CONFIG)
    env = AtariEnvironment(config)

    obs = env.reset(seed=123)
    assert obs.dtype == np.float32
    assert obs.shape == (84, 84, 4)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0 + 1e-5

    action = 0
    next_obs, reward, done, info = env.step(action)
    assert next_obs.dtype == np.float32
    assert next_obs.shape == (84, 84, 4)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    env.close()


def test_vectorized_atari_environment_shapes():
    config = dict(BASE_CONFIG)
    config.update({
        "num_environments": 2,
        "parallel_backend": "sync",
    })
    env = AtariEnvironment(config)

    obs = env.reset(seed=321)
    assert obs.shape == (2, 84, 84, 4)
    assert obs.dtype == np.float32

    actions = np.zeros(2, dtype=np.int64)
    next_obs, rewards, dones, infos = env.step(actions)

    assert next_obs.shape == (2, 84, 84, 4)
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (2,)
    assert isinstance(dones, np.ndarray)
    assert dones.shape == (2,)
    assert isinstance(infos, list)
    assert len(infos) == 2
    assert all(isinstance(item, dict) for item in infos)

    env.close()
