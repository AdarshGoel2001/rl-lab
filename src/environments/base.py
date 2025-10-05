"""
Base Environment Interface

This module provides a unified interface for all environments, whether they're
from OpenAI Gym, DeepMind Control Suite, Atari, or custom environments.

Key benefits:
- Consistent interface across all environment types
- Automatic observation/action space handling
- Built-in normalization and preprocessing
- Easy environment swapping via config
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpaceSpec:
    """Specification for observation or action spaces"""
    shape: Tuple[int, ...]
    dtype: np.dtype
    low: Optional[Union[float, np.ndarray]] = None
    high: Optional[Union[float, np.ndarray]] = None
    discrete: bool = False
    n: Optional[int] = None  # For discrete spaces


class BaseEnvironment(ABC):
    """
    Abstract base class for all environment wrappers.
    
    This provides a consistent interface for interacting with any type of
    environment. All environment wrappers should inherit from this class
    and implement the abstract methods.
    
    Attributes:
        config: Environment configuration dictionary
        _observation_space: Specification of observation space
        _action_space: Specification of action space
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize environment wrapper.
        
        Args:
            config: Dictionary containing environment configuration including:
                - name: Environment name/identifier
                - normalize_obs: Whether to normalize observations
                - normalize_reward: Whether to normalize rewards
                - max_episode_steps: Maximum steps per episode
                - observation_transforms: List of transform configurations
                - Any environment-specific parameters
        """
        self.config = config
        self.name = config.get('name', 'unknown')
        self.normalize_obs = config.get('normalize_obs', False)
        self.normalize_reward = config.get('normalize_reward', False)
        self.max_episode_steps = config.get('max_episode_steps', None)
        
        # Environment type flags
        self.is_vectorized = getattr(self, 'is_vectorized', False)
        self.num_envs = getattr(self, 'num_envs', 1)
        
        self._current_step = 0
        self._episode_return = 0.0
        self._episode_count = 0
        
        # Normalization statistics (updated online if normalization enabled)
        self._obs_mean = None
        self._obs_var = None
        self._reward_mean = 0.0
        self._reward_var = 1.0
        
        # Initialize transform pipeline
        self.transform_pipeline = None
        self._setup_transform_pipeline()
        
        # Initialize environment-specific components
        self._setup_environment()
        
        # Cache space specifications
        self._observation_space = self._get_observation_space()
        self._action_space = self._get_action_space()
    
    def _setup_transform_pipeline(self):
        """Setup the observation transform pipeline"""
        try:
            from src.environments.transforms import create_transform_pipeline, expand_preset_configs
            
            # Get transform configurations
            transform_configs = self.config.get('observation_transforms', [])
            
            if transform_configs:
                # Expand any preset configurations
                expanded_configs = expand_preset_configs(transform_configs)
                
                # Create transform pipeline
                self.transform_pipeline = create_transform_pipeline(expanded_configs)
                
                if self.transform_pipeline:
                    logger.debug(f"Created transform pipeline with {len(self.transform_pipeline)} transforms: {self.transform_pipeline}")
                else:
                    logger.debug("No transform pipeline created (empty configuration)")
            else:
                logger.debug("No observation transforms specified")
                
        except ImportError as e:
            logger.warning(f"Could not import transforms module: {e}")
            self.transform_pipeline = None
        except Exception as e:
            logger.error(f"Failed to setup transform pipeline: {e}")
            self.transform_pipeline = None
    
    def _apply_transforms(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply transform pipeline to observation.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Transformed observation
        """
        if self.transform_pipeline is not None:
            try:
                return self.transform_pipeline.apply(obs)
            except Exception as e:
                logger.error(f"Transform pipeline failed: {e}")
                # Return original observation if transforms fail
                return obs
        return obs
    
    def _reset_transform_states(self):
        """Reset state of all stateful transforms"""
        if self.transform_pipeline is not None:
            self.transform_pipeline.reset_states()

    @abstractmethod
    def _setup_environment(self):
        """
        Setup the underlying environment.
        
        This should create the actual environment instance (gym.Env, dm_env.Environment, etc.)
        and handle any environment-specific initialization.
        """
        pass
    
    @abstractmethod
    def _get_observation_space(self) -> SpaceSpec:
        """
        Get observation space specification.
        
        Returns:
            SpaceSpec describing the observation space
        """
        pass
    
    @abstractmethod
    def _get_action_space(self) -> SpaceSpec:
        """
        Get action space specification.
        
        Returns:
            SpaceSpec describing the action space
        """
        pass
    
    @abstractmethod
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the underlying environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation as numpy array
        """
        pass
    
    @abstractmethod
    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Step the underlying environment.
        
        Args:
            action: Action to take as numpy array
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Reset environment and return initial observation.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation as PyTorch tensor
        """
        self._current_step = 0
        self._episode_return = 0.0
        
        # Reset transform states before resetting environment
        self._reset_transform_states()
        
        # Reset underlying environment
        obs = self._reset_environment(seed)
        
        # Apply transforms first, then normalization
        obs = self._apply_transforms(obs)
        
        # Apply normalization if enabled
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        
        return torch.tensor(obs, dtype=torch.float32)
    
    def step(self, action: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Execute action and return next observation, reward, done, info.
        
        Args:
            action: Action to execute (tensor or numpy array)
            
        Returns:
            Tuple of (next_observation, reward, done, info) where:
            - next_observation: PyTorch tensor
            - reward: Float reward value  
            - done: Boolean indicating if episode is finished
            - info: Dictionary with additional information
        """
        self._current_step += 1
        
        # Convert action to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Step underlying environment
        obs, reward, done, info = self._step_environment(action)

        # Apply transforms first, then normalization
        obs = self._apply_transforms(obs)

        if self.is_vectorized:
            # Handle vectorized outputs (arrays/lists)
            if self.normalize_obs:
                obs = self._normalize_observation(obs)

            if self.normalize_reward:
                reward = np.asarray(reward, dtype=np.float32)
                reward = np.array([self._normalize_reward(r) for r in reward], dtype=np.float32)
            else:
                reward = np.asarray(reward, dtype=np.float32)

            done = np.asarray(done)

            if not isinstance(info, list):
                if isinstance(info, dict):
                    info = [dict(info) for _ in range(self.num_envs)]
                else:
                    info = [{} for _ in range(self.num_envs)]

            return torch.tensor(obs, dtype=torch.float32), reward, done, info

        # Scalar environment handling
        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        if self.normalize_reward:
            reward = self._normalize_reward(float(reward))

        # Update episode statistics
        self._episode_return += reward

        # Check for episode timeout
        if self.max_episode_steps and self._current_step >= self.max_episode_steps:
            done = True
            if isinstance(info, dict):
                info['timeout'] = True
            else:
                info = {'timeout': True}

        if not isinstance(info, dict):
            info = {}

        # Add episode statistics to info
        info.update({
            'episode_step': self._current_step,
            'episode_return': self._episode_return
        })

        # Reset episode counter when done
        if done:
            self._episode_count += 1
            info['episode_count'] = self._episode_count

        return torch.tensor(obs, dtype=torch.float32), float(reward), bool(done), info
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation normalization"""
        if self._obs_mean is None:
            self._obs_mean = np.zeros_like(obs)
            self._obs_var = np.ones_like(obs)
            return obs
        
        # Update running statistics (simple exponential moving average)
        alpha = 0.01
        self._obs_mean = (1 - alpha) * self._obs_mean + alpha * obs
        self._obs_var = (1 - alpha) * self._obs_var + alpha * (obs - self._obs_mean) ** 2
        
        # Normalize observation
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
    
    def _normalize_reward(self, reward: float) -> float:
        """Apply reward normalization"""
        # Update running statistics
        alpha = 0.01
        self._reward_mean = (1 - alpha) * self._reward_mean + alpha * reward
        self._reward_var = (1 - alpha) * self._reward_var + alpha * (reward - self._reward_mean) ** 2
        
        # Normalize reward
        return (reward - self._reward_mean) / (np.sqrt(self._reward_var) + 1e-8)
    
    @property
    def observation_space(self) -> SpaceSpec:
        """Get observation space specification"""
        return self._observation_space
    
    @property
    def action_space(self) -> SpaceSpec:
        """Get action space specification"""
        return self._action_space
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get environment metrics for logging.
        
        Returns:
            Dictionary of environment metrics
        """
        metrics = {
            'env/episode_count': float(self._episode_count),
            'env/current_step': float(self._current_step),
            'env/episode_return': self._episode_return,
        }
        
        if self.normalize_obs and self._obs_mean is not None:
            metrics.update({
                'env/obs_mean': float(np.mean(self._obs_mean)),
                'env/obs_std': float(np.mean(np.sqrt(self._obs_var))),
            })
        
        if self.normalize_reward:
            metrics.update({
                'env/reward_mean': self._reward_mean,
                'env/reward_std': float(np.sqrt(self._reward_var)),
            })
        
        return metrics
    
    def close(self):
        """Close the environment and clean up resources"""
        pass
    
    def render(self, mode: str = 'human'):
        """Render the environment (override in subclasses if supported)"""
        pass
