"""DeepMind Control Suite Environment Wrapper.

Wraps DMC environments to match the RL-Lab BaseEnvironment interface.
DMC uses MuJoCo physics but provides a different task interface than Gymnasium.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from src.environments.base import BaseEnvironment, SpaceSpec

logger = logging.getLogger(__name__)


class DMCWrapper(BaseEnvironment):
    """Wrapper for DeepMind Control Suite environments.

    DMC environments are specified as 'domain_task' (e.g., 'walker_walk', 'cheetah_run').

    Example config:
        _target_: src.environments.dmc_wrapper.DMCWrapper
        name: walker_walk
        from_pixels: false
        frame_skip: 1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        config = dict(config or {})
        if kwargs:
            config.update(kwargs)

        self.env_name = config['name']
        self.from_pixels = config.get('from_pixels', False)
        self.camera_id = config.get('camera_id', 0)
        self.frame_skip = config.get('frame_skip', 1)
        self.height = config.get('height', 84)
        self.width = config.get('width', 84)

        # Parse domain_task format
        if '_' in self.env_name:
            parts = self.env_name.split('_', 1)
            self.domain = parts[0]
            self.task = parts[1]
        else:
            raise ValueError(f"DMC env name must be 'domain_task', got: {self.env_name}")

        super().__init__(config)

    def _setup_environment(self):
        """Create the DMC environment."""
        try:
            from dm_control import suite

            self.env = suite.load(
                domain_name=self.domain,
                task_name=self.task,
                task_kwargs={'random': self.config.get('seed')}
            )

            # Wrap for pixel observations if needed
            if self.from_pixels:
                from dm_control.suite.wrappers import pixels
                self.env = pixels.Wrapper(
                    self.env,
                    pixels_only=True,
                    render_kwargs={
                        'camera_id': self.camera_id,
                        'height': self.height,
                        'width': self.width,
                    }
                )

            logger.info(f"Created DMC environment: {self.domain}_{self.task}")

        except ImportError:
            raise ImportError("dm-control not installed. Run: pip install dm-control")

    def _flatten_obs(self, obs) -> np.ndarray:
        """Flatten DMC observation dict to single array."""
        if isinstance(obs, dict):
            arrays = []
            for key in sorted(obs.keys()):
                arr = np.asarray(obs[key], dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                arrays.append(arr.flatten())
            return np.concatenate(arrays)
        return np.asarray(obs, dtype=np.float32)

    def _get_obs_dim(self) -> int:
        """Calculate total observation dimension."""
        obs_spec = self.env.observation_spec()
        if isinstance(obs_spec, dict):
            total = 0
            for spec in obs_spec.values():
                shape = spec.shape
                total += int(np.prod(shape)) if shape else 1
            return total
        return int(np.prod(obs_spec.shape))

    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification."""
        if self.from_pixels:
            return SpaceSpec(
                shape=(3, self.height, self.width),
                dtype=np.uint8,
                low=0,
                high=255,
                discrete=False
            )
        else:
            obs_dim = self._get_obs_dim()
            return SpaceSpec(
                shape=(obs_dim,),
                dtype=np.float32,
                discrete=False
            )

    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification."""
        action_spec = self.env.action_spec()
        return SpaceSpec(
            shape=action_spec.shape,
            dtype=np.float32,
            low=action_spec.minimum,
            high=action_spec.maximum,
            discrete=False
        )

    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the DMC environment."""
        timestep = self.env.reset()
        obs = self._flatten_obs(timestep.observation)
        return obs

    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the DMC environment."""
        # Ensure action is proper shape and type
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 1:
            action = action.squeeze()

        # Clip action to valid range
        action_spec = self.env.action_spec()
        action = np.clip(action, action_spec.minimum, action_spec.maximum)

        # Step with optional frame skip
        reward = 0.0
        for _ in range(self.frame_skip):
            timestep = self.env.step(action)
            reward += timestep.reward or 0.0
            if timestep.last():
                break

        obs = self._flatten_obs(timestep.observation)
        done = timestep.last()

        info = {
            'discount': timestep.discount,
            'step_type': timestep.step_type.name,
        }

        return obs, float(reward), bool(done), info

    def render(self, mode: str = 'rgb_array'):
        """Render the environment."""
        if mode == 'rgb_array':
            return self.env.physics.render(
                camera_id=self.camera_id,
                height=self.height,
                width=self.width,
            )
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        logger.debug(f"Closed DMC environment: {self.env_name}")


# Common DMC environments and their dimensions for reference
DMC_ENVS = {
    # domain_task: (obs_dim, action_dim)
    'walker_walk': (24, 6),
    'walker_run': (24, 6),
    'walker_stand': (24, 6),
    'cheetah_run': (17, 6),
    'hopper_stand': (15, 4),
    'hopper_hop': (15, 4),
    'reacher_easy': (6, 2),
    'reacher_hard': (6, 2),
    'finger_spin': (9, 2),
    'finger_turn_easy': (12, 2),
    'finger_turn_hard': (12, 2),
    'cartpole_balance': (5, 1),
    'cartpole_swingup': (5, 1),
    'ball_in_cup_catch': (8, 2),
    'pendulum_swingup': (3, 1),
    'humanoid_stand': (67, 21),
    'humanoid_walk': (67, 21),
    'humanoid_run': (67, 21),
    'quadruped_walk': (78, 12),
    'quadruped_run': (78, 12),
}
