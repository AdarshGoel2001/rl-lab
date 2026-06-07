import sys
import types

import numpy as np

from src.environments.dmc_vectorized_wrapper import DMCVectorizedWrapper


class _FakeSpec:
    def __init__(self, shape, minimum=None, maximum=None):
        self.shape = shape
        self.minimum = np.asarray(minimum if minimum is not None else np.full(shape, -1.0), dtype=np.float32)
        self.maximum = np.asarray(maximum if maximum is not None else np.full(shape, 1.0), dtype=np.float32)


class _FakeStepType:
    def __init__(self, name):
        self.name = name


class _FakeTimestep:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.discount = 0.0 if done else 1.0
        self.step_type = _FakeStepType("LAST" if done else "MID")
        self._done = done

    def last(self):
        return self._done


class _FakeDMCEnv:
    def __init__(self, env_id):
        self.env_id = env_id
        self.steps = 0
        self.closed = False

    def observation_spec(self):
        return {
            "position": _FakeSpec((3,)),
            "velocity": _FakeSpec((2,)),
        }

    def action_spec(self):
        return _FakeSpec((1,), minimum=[-1.0], maximum=[1.0])

    def reset(self):
        self.steps = 0
        return _FakeTimestep(
            {
                "position": np.full(3, self.env_id, dtype=np.float32),
                "velocity": np.zeros(2, dtype=np.float32),
            },
            reward=0.0,
            done=False,
        )

    def step(self, action):
        del action
        self.steps += 1
        done = self.steps >= 2
        return _FakeTimestep(
            {
                "position": np.full(3, self.env_id, dtype=np.float32),
                "velocity": np.full(2, self.steps, dtype=np.float32),
            },
            reward=float(self.env_id + 1),
            done=done,
        )

    def close(self):
        self.closed = True


def test_dmc_vectorized_wrapper_batches_steps_and_autoresets_completed_envs(monkeypatch):
    loaded = []

    def fake_load(domain_name, task_name, task_kwargs=None):
        del domain_name, task_name, task_kwargs
        env = _FakeDMCEnv(len(loaded))
        loaded.append(env)
        return env

    suite_module = types.SimpleNamespace(load=fake_load)
    dm_control_module = types.SimpleNamespace(suite=suite_module)
    monkeypatch.setitem(sys.modules, "dm_control", dm_control_module)
    monkeypatch.setitem(sys.modules, "dm_control.suite", suite_module)

    env = DMCVectorizedWrapper(
        name="cartpole_swingup",
        num_envs=3,
        frame_skip=1,
        max_episode_steps=10,
    )

    obs = env.reset(seed=123)
    assert obs.shape == (3, 5)
    assert env.is_vectorized is True
    assert env.num_envs == 3
    assert env.observation_space.shape == (3, 5)
    assert env.action_space.shape == (3, 1)

    next_obs, reward, done, infos = env.step(np.zeros((3, 1), dtype=np.float32))
    assert next_obs.shape == (3, 5)
    assert reward.tolist() == [1.0, 2.0, 3.0]
    assert done.tolist() == [False, False, False]
    assert len(infos) == 3

    next_obs, reward, done, infos = env.step(np.zeros((3, 1), dtype=np.float32))
    assert next_obs.shape == (3, 5)
    assert reward.tolist() == [1.0, 2.0, 3.0]
    assert done.tolist() == [True, True, True]
    assert all(info["episode_count"] == 1 for info in infos)
    assert all(info["episode_length"] == 2 for info in infos)
    assert all(info["episode_return"] > 0.0 for info in infos)
    assert all(fake_env.steps == 0 for fake_env in loaded)

    env.close()
    assert all(fake_env.closed for fake_env in loaded)
