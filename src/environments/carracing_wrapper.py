"""CarRacing environment wrapper for World Models.

Preprocesses observations to match the 2018 paper:
- Converts 96x96 RGB to 64x64 grayscale
- Keeps uint8 format [0, 255] (workflow normalizes to [0, 1])
- Handles continuous actions [steering, gas, brake]
- Supports vectorized environments for parallel collection
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import gymnasium as gym

from .base import BaseEnvironment, SpaceSpec


class CarRacingWorldModelWrapper(BaseEnvironment):
    """CarRacing-v3 wrapper with grayscale 64x64 preprocessing and vectorization support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        defaults = {
            "name": "CarRacing-v3",
            "num_envs": 1,
            "max_episode_steps": 1000,
            "render_mode": None,  # No rendering for training
            "domain_randomize": False,  # Can enable for robustness
            "continuous": True,  # Use continuous actions
            "lap_complete_percent": 0.95,  # Episode ends at 95% track completion
            "parallel_backend": "sync",  # 'sync' or 'async' for vectorized envs
        }
        merged = dict(defaults)
        merged.update(config or {})
        if kwargs:
            merged.update(kwargs)

        # Initialize attributes before super().__init__() calls _setup_environment()
        self.env: Optional[gym.Env] = None
        self._last_info: Dict[str, Any] = {}

        # Vectorization setup
        self.num_environments = int(merged.get('num_environments', merged.get('num_envs', 1)))
        self.parallel_backend = merged.get('parallel_backend', 'sync')

        # Set flags for BaseEnvironment
        self.is_vectorized = self.num_environments > 1
        self.num_envs = self.num_environments if self.is_vectorized else 1

        super().__init__(merged, **kwargs)

        # Initialize per-environment episode tracking for vectorized envs
        if self.is_vectorized:
            self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # BaseEnvironment implementation
    # ------------------------------------------------------------------
    def _setup_environment(self) -> None:
        """Create the CarRacing-v3 environment (single or vectorized)."""
        env_kwargs = {
            "domain_randomize": self.config.get("domain_randomize", False),
            "continuous": self.config.get("continuous", True),
            "lap_complete_percent": self.config.get("lap_complete_percent", 0.95),
        }

        render_mode = self.config.get("render_mode")
        if render_mode is not None and not self.is_vectorized:
            env_kwargs["render_mode"] = render_mode

        def build_single_env():
            """Build a single CarRacing environment."""
            return gym.make(self.config["name"], **env_kwargs)

        if self.is_vectorized:
            try:
                from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
            except ImportError as e:
                raise ImportError("gymnasium installation required for vectorized environments") from e

            def make_env_fn(seed_offset: int = 0):
                def _init():
                    env = build_single_env()
                    if hasattr(env, 'seed'):
                        env.seed(self.config.get('seed', 0) + seed_offset)
                    return env
                return _init

            env_fns = [make_env_fn(i) for i in range(self.num_envs)]
            use_async = self.parallel_backend in ('async', 'auto')

            if use_async:
                self.env = AsyncVectorEnv(env_fns)
            else:
                self.env = SyncVectorEnv(env_fns)
        else:
            self.env = build_single_env()

    def _get_observation_space(self) -> SpaceSpec:
        """64x64 grayscale image (H, W, C) format."""
        return SpaceSpec(
            shape=(64, 64, 1),
            dtype=np.uint8,
            low=0,
            high=255
        )

    def _get_action_space(self) -> SpaceSpec:
        """Continuous actions: [steering, gas, brake]."""
        return SpaceSpec(
            shape=(3,),
            dtype=np.float32,
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )

    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return preprocessed observation."""
        if self.env is None:
            raise RuntimeError("Environment not setup. Call _setup_environment first.")

        if self.is_vectorized:
            if seed is not None:
                seeds = [seed + i for i in range(self.num_envs)]
                obs, _ = self.env.reset(seed=seeds)
            else:
                obs, _ = self.env.reset()

            # Reset episode statistics
            if hasattr(self, '_episode_returns'):
                self._episode_returns.fill(0.0)
                self._episode_lengths.fill(0)

            # Preprocess all observations
            processed_obs = np.stack([self._preprocess_observation(o) for o in obs], axis=0)
        else:
            obs, info = self.env.reset(seed=seed)
            self._last_info = info
            processed_obs = self._preprocess_observation(obs)

        return processed_obs

    def _step_environment(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step environment with the given action."""
        if self.env is None:
            raise RuntimeError("Environment not setup. Call _setup_environment first.")

        if self.is_vectorized:
            # Ensure action is float32 numpy array with shape (num_envs, 3)
            actions = np.asarray(action, dtype=np.float32)
            if actions.ndim == 1:
                # Single action provided, broadcast to all envs
                actions = np.tile(actions, (self.num_envs, 1))

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            # Preprocess all observations
            processed_obs = np.stack([self._preprocess_observation(o) for o in obs], axis=0)

            # Update episode statistics
            if hasattr(self, '_episode_returns'):
                self._episode_returns += rewards
                self._episode_lengths += 1

            # Return vectorized format
            return processed_obs, rewards, dones, infos
        else:
            # Ensure action is float32 numpy array
            action = np.asarray(action, dtype=np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # CarRacing: done when terminated OR truncated
            done = terminated or truncated

            # Preprocess observation
            processed_obs = self._preprocess_observation(obs)

            # Store info for debugging
            self._last_info = info

            return processed_obs, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert 96x96 RGB to 64x64 grayscale.

        Args:
            obs: RGB image with shape (96, 96, 3), dtype uint8

        Returns:
            Grayscale image with shape (64, 64, 1), dtype uint8
        """
        # Convert RGB to grayscale using standard weights
        # Gray = 0.299*R + 0.587*G + 0.114*B
        if obs.shape[-1] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs.squeeze(-1) if obs.ndim == 3 else obs

        # Resize to 64x64 using bilinear interpolation
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Add channel dimension (64, 64) → (64, 64, 1)
        processed = resized[:, :, np.newaxis]

        return processed.astype(np.uint8)

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment (only works for single env, not vectorized)."""
        if self.env is None:
            raise RuntimeError("Environment not setup.")

        if self.is_vectorized:
            raise NotImplementedError("Rendering not supported for vectorized environments")

        if mode == "rgb_array":
            return self.env.render()
        else:
            self.env.render()
            return None

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None


__all__ = ["CarRacingWorldModelWrapper"]
