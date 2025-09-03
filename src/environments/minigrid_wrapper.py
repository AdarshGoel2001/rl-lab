"""
MiniGrid Environment Wrapper

Provides a unified interface for MiniGrid gridworld environments with support for:
- Dict observation spaces (image + direction)
- Observation processing (raw dict, one-hot encoding, CNN input)
- Success rate tracking for evaluation
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper

from .base import BaseEnvironment, SpaceSpec
from ..utils.registry import register_environment


@register_environment("minigrid")
class MiniGridWrapper(BaseEnvironment):
    """
    Wrapper for MiniGrid environments with configurable observation processing.
    
    Supports multiple observation modes:
    - 'dict': Raw Dict observation (image + direction + mission)
    - 'image': Just the 7x7x3 image tensor
    - 'one_hot': Flattened one-hot encoded observation
    - 'flat': Concatenated image + direction vector
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MiniGrid environment wrapper.
        
        Args:
            config: Configuration dictionary with keys:
                - name: MiniGrid environment name (e.g., 'DoorKey-5x5-v0')
                - obs_mode: Observation processing mode ('dict', 'image', 'one_hot', 'flat')
                - max_steps: Maximum steps per episode (overrides env default)
                - render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        self.obs_mode = config.get('obs_mode', 'image')
        self.render_mode = config.get('render_mode', None)
        
        # Success tracking for evaluation
        self._episode_success_count = 0
        self._episode_total_count = 0
        
        super().__init__(config)
    
    def _setup_environment(self):
        """Setup the MiniGrid environment with appropriate wrappers"""
        env_name = self.name
        
        # Create base environment
        self.env = gym.make(env_name, render_mode=self.render_mode)
        
        # Apply observation wrappers based on obs_mode
        if self.obs_mode == 'image':
            # Use ImgObsWrapper to get just the image
            self.env = ImgObsWrapper(self.env)
        elif self.obs_mode == 'one_hot':
            # Use FlatObsWrapper for one-hot encoding
            self.env = FlatObsWrapper(self.env)
        elif self.obs_mode == 'flat':
            # Custom wrapper for image + direction concatenation
            self.env = ImageDirectionFlatWrapper(self.env)
        # For 'dict' mode, use raw environment
        
        # Set max steps if specified
        if self.max_episode_steps:
            self.env._max_episode_steps = self.max_episode_steps
    
    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification based on obs_mode"""
        if self.obs_mode == 'dict':
            # Dict space - for now return the image space and handle dict separately
            image_shape = self.env.observation_space['image'].shape
            return SpaceSpec(
                shape=image_shape,
                dtype=np.uint8,
                low=0,
                high=255,
                discrete=False
            )
        elif self.obs_mode == 'image':
            # Image only: (7, 7, 3)
            obs_space = self.env.observation_space
            return SpaceSpec(
                shape=obs_space.shape,
                dtype=obs_space.dtype,
                low=obs_space.low,
                high=obs_space.high,
                discrete=False
            )
        elif self.obs_mode in ['one_hot', 'flat']:
            # Flattened observation
            obs_space = self.env.observation_space
            return SpaceSpec(
                shape=obs_space.shape,
                dtype=obs_space.dtype,
                low=obs_space.low,
                high=obs_space.high,
                discrete=False
            )
        else:
            raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")
    
    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification"""
        action_space = self.env.action_space
        return SpaceSpec(
            shape=(1,) if hasattr(action_space, 'n') else action_space.shape,
            dtype=np.int64 if hasattr(action_space, 'n') else action_space.dtype,
            discrete=hasattr(action_space, 'n'),
            n=action_space.n if hasattr(action_space, 'n') else None
        )
    
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the MiniGrid environment"""
        obs, info = self.env.reset(seed=seed)
        
        # Process observation based on mode
        if self.obs_mode == 'dict':
            # Return image component for now (can be extended for full dict support)
            return obs['image'].astype(np.float32)
        else:
            return obs.astype(np.float32)
    
    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the MiniGrid environment"""
        # Convert action to integer if needed
        if hasattr(action, 'ndim'):
            if action.ndim > 0:
                action = int(action[0])
            else:
                action = int(action)
        else:
            action = int(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Track success for evaluation
        if done:
            self._episode_total_count += 1
            if terminated and reward > 0:  # MiniGrid gives positive reward for success
                self._episode_success_count += 1
                info['success'] = True
            else:
                info['success'] = False
        
        # Process observation based on mode
        if self.obs_mode == 'dict':
            # Return image component for now
            obs = obs['image'].astype(np.float32)
        else:
            obs = obs.astype(np.float32)
        
        return obs, reward, done, info
    
    def get_metrics(self) -> Dict[str, float]:
        """Get environment metrics including success rate"""
        metrics = super().get_metrics()
        
        # Add success rate metrics
        if self._episode_total_count > 0:
            success_rate = self._episode_success_count / self._episode_total_count
            metrics.update({
                'env/success_rate': success_rate,
                'env/success_count': float(self._episode_success_count),
                'env/total_episodes': float(self._episode_total_count)
            })
        
        return metrics
    
    def reset_success_tracking(self):
        """Reset success tracking counters"""
        self._episode_success_count = 0
        self._episode_total_count = 0
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'env'):
            self.env.close()


class ImageDirectionFlatWrapper(gym.ObservationWrapper):
    """
    Wrapper that flattens the image observation and concatenates with direction.
    Converts Dict(image, direction, mission) -> flattened vector.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get original spaces
        orig_obs_space = env.observation_space
        image_shape = orig_obs_space['image'].shape
        
        # Calculate flattened size: image + direction (4 possible directions)
        flat_size = np.prod(image_shape) + 4  # 4 for one-hot encoded direction
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(flat_size,), dtype=np.float32
        )
    
    def observation(self, obs):
        """Flatten image and concatenate with one-hot encoded direction"""
        image = obs['image'].flatten()
        
        # One-hot encode direction (0-3 -> 4-dim vector)
        direction_one_hot = np.zeros(4)
        direction_one_hot[obs['direction']] = 1
        
        # Concatenate
        return np.concatenate([image, direction_one_hot]).astype(np.float32)


# Helper function to create MiniGrid environment
def make_minigrid_env(env_name: str, obs_mode: str = 'image', **kwargs) -> MiniGridWrapper:
    """
    Convenience function to create a MiniGrid environment.
    
    Args:
        env_name: MiniGrid environment name
        obs_mode: Observation processing mode
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured MiniGridWrapper instance
    """
    config = {
        'name': env_name,
        'obs_mode': obs_mode,
        **kwargs
    }
    return MiniGridWrapper(config)