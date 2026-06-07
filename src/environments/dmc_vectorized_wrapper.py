"""Vectorized DeepMind Control Suite wrapper.

This wrapper keeps DMC stepping synchronous but exposes a batched environment
interface so workflows can collect from multiple DMC instances per collect step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.environments.base import BaseEnvironment, SpaceSpec

logger = logging.getLogger(__name__)


class DMCVectorizedWrapper(BaseEnvironment):
    """Synchronous vector wrapper for low-dimensional DMC environments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        config = dict(config or {})
        if kwargs:
            config.update(kwargs)

        self.env_name = config["name"]
        self.num_envs = int(config.get("num_envs", 4))
        self.from_pixels = bool(config.get("from_pixels", False))
        self.camera_id = config.get("camera_id", 0)
        self.frame_skip = int(config.get("frame_skip", 1))
        self.height = int(config.get("height", 84))
        self.width = int(config.get("width", 84))
        self.is_vectorized = True

        if "_" not in self.env_name:
            raise ValueError(f"DMC env name must be 'domain_task', got: {self.env_name}")
        self.domain, self.task = self.env_name.split("_", 1)

        self.envs: list[Any] = []
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int32)

        super().__init__(config)

    def _setup_environment(self):
        try:
            from dm_control import suite
        except ImportError as exc:
            raise ImportError("dm-control not installed. Run: pip install dm-control") from exc

        self.envs = []
        base_seed = self.config.get("seed")
        for env_id in range(self.num_envs):
            random_seed = None if base_seed is None else int(base_seed) + env_id
            env = suite.load(
                domain_name=self.domain,
                task_name=self.task,
                task_kwargs={"random": random_seed},
            )
            if self.from_pixels:
                from dm_control.suite.wrappers import pixels

                env = pixels.Wrapper(
                    env,
                    pixels_only=True,
                    render_kwargs={
                        "camera_id": self.camera_id,
                        "height": self.height,
                        "width": self.width,
                    },
                )
            self.envs.append(env)

        logger.info("Created vectorized DMC environment: %s (%s envs)", self.env_name, self.num_envs)

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            arrays = []
            for key in sorted(obs.keys()):
                arr = np.asarray(obs[key], dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                arrays.append(arr.flatten())
            return np.concatenate(arrays)
        arr = np.asarray(obs)
        if self.from_pixels and arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1)).astype(np.uint8)
        return arr.astype(np.float32)

    def _get_obs_dim(self) -> int:
        obs_spec = self.envs[0].observation_spec()
        if isinstance(obs_spec, dict):
            total = 0
            for spec in obs_spec.values():
                shape = spec.shape
                total += int(np.prod(shape)) if shape else 1
            return total
        return int(np.prod(obs_spec.shape))

    def _get_observation_space(self) -> SpaceSpec:
        if self.from_pixels:
            return SpaceSpec(
                shape=(self.num_envs, 3, self.height, self.width),
                dtype=np.uint8,
                low=0,
                high=255,
                discrete=False,
            )

        return SpaceSpec(
            shape=(self.num_envs, self._get_obs_dim()),
            dtype=np.float32,
            discrete=False,
        )

    def _get_action_space(self) -> SpaceSpec:
        action_spec = self.envs[0].action_spec()
        return SpaceSpec(
            shape=(self.num_envs,) + tuple(action_spec.shape),
            dtype=np.float32,
            low=action_spec.minimum,
            high=action_spec.maximum,
            discrete=False,
        )

    def _reset_one(self, env_id: int) -> np.ndarray:
        timestep = self.envs[env_id].reset()
        self._episode_returns[env_id] = 0.0
        self._episode_lengths[env_id] = 0
        return self._flatten_obs(timestep.observation)

    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        del seed
        observations = [self._reset_one(env_id) for env_id in range(self.num_envs)]
        return np.asarray(observations, dtype=np.float32)

    def _step_environment(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got batch size {actions.shape[0]}.")

        next_observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for env_id, env in enumerate(self.envs):
            action_spec = env.action_spec()
            action = np.asarray(actions[env_id], dtype=np.float32)
            action = np.clip(action, action_spec.minimum, action_spec.maximum)

            reward = 0.0
            timestep = None
            for _ in range(self.frame_skip):
                timestep = env.step(action)
                reward += float(timestep.reward or 0.0)
                if timestep.last():
                    break
            if timestep is None:
                raise RuntimeError("DMCVectorizedWrapper frame_skip must be at least 1.")

            self._episode_returns[env_id] += reward
            self._episode_lengths[env_id] += 1
            done = bool(timestep.last())
            if self.max_episode_steps and self._episode_lengths[env_id] >= int(self.max_episode_steps):
                done = True

            info = {
                "discount": timestep.discount,
                "step_type": timestep.step_type.name,
                "episode_step": int(self._episode_lengths[env_id]),
                "episode_return": float(self._episode_returns[env_id]),
            }

            obs = self._flatten_obs(timestep.observation)
            if done:
                self._episode_counts[env_id] += 1
                info.update(
                    {
                        "episode_count": int(self._episode_counts[env_id]),
                        "episode_length": int(self._episode_lengths[env_id]),
                        "terminal_observation": obs.copy(),
                    }
                )
                if self.max_episode_steps and self._episode_lengths[env_id] >= int(self.max_episode_steps):
                    info["timeout"] = True
                obs = self._reset_one(env_id)

            next_observations.append(obs)
            rewards[env_id] = reward
            dones[env_id] = done
            infos.append(info)

        return np.asarray(next_observations, dtype=np.float32), rewards, dones, infos

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        obs = self._reset_environment(seed)
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        return np.asarray(obs, dtype=np.float32)

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        obs, rewards, dones, infos = self._step_environment(actions)
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        if self.normalize_reward:
            rewards = np.asarray([self._normalize_reward(float(r)) for r in rewards], dtype=np.float32)

        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def render(self, mode: str = "rgb_array"):
        if mode == "rgb_array" and self.envs:
            return self.envs[0].physics.render(
                camera_id=self.camera_id,
                height=self.height,
                width=self.width,
            )
        return None

    def close(self):
        for env in getattr(self, "envs", []):
            if hasattr(env, "close"):
                env.close()
        logger.debug("Closed vectorized DMC environment: %s (%s envs)", self.env_name, self.num_envs)
