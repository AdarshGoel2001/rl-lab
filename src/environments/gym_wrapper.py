"""
OpenAI Gym Environment Wrapper

This module provides a unified wrapper for OpenAI Gym environments that
integrates seamlessly with the RL framework. It handles observation/action
space conversion, normalization, and provides consistent interface.

Key features:
- Automatic space conversion to framework format
- Optional observation and reward normalization
- Episode length limiting
- Consistent tensor-based interface
- Error handling and logging
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, Union

from src.environments.base import BaseEnvironment, SpaceSpec
from src.utils.registry import register_environment

logger = logging.getLogger(__name__)


@register_environment("gym")
class GymWrapper(BaseEnvironment):
    """
    Wrapper for OpenAI Gym environments.
    
    Provides a consistent interface for all Gym environments while handling
    space conversions, normalization, and tensor management.
    
    Attributes:
        env: The underlying Gym environment
        env_name: Name of the Gym environment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gym environment wrapper.
        
        Args:
            config: Environment configuration containing:
                - name: Gym environment name (e.g., 'CartPole-v1')
                - normalize_obs: Whether to normalize observations
                - normalize_reward: Whether to normalize rewards
                - max_episode_steps: Maximum steps per episode
                - render_mode: Rendering mode ('human', 'rgb_array', None)
                - Additional environment-specific kwargs
        """
        self.env_name = config['name']
        self.render_mode = config.get('render_mode', None)
        self.env_kwargs = config.get('env_kwargs', {})
        
        super().__init__(config)
    
    def _setup_environment(self):
        """Setup the Gym environment"""
        try:
            import gymnasium as gym
            
            # Create environment
            if self.render_mode is not None:
                self.env = gym.make(self.env_name, render_mode=self.render_mode, **self.env_kwargs)
            else:
                self.env = gym.make(self.env_name, **self.env_kwargs)
            
            logger.info(f"Created Gym environment: {self.env_name}")
            
        except ImportError as e:
            logger.error("Gymnasium not installed. Install with: pip install gymnasium")
            raise e
        except Exception as e:
            logger.error(f"Failed to create Gym environment {self.env_name}: {e}")
            raise e
    
    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification from Gym environment"""
        obs_space = self.env.observation_space
        
        if hasattr(obs_space, 'shape'):
            # Box or similar space
            shape = obs_space.shape
            dtype = obs_space.dtype if hasattr(obs_space, 'dtype') else np.float32
            low = obs_space.low if hasattr(obs_space, 'low') else None
            high = obs_space.high if hasattr(obs_space, 'high') else None
            
            return SpaceSpec(
                shape=shape,
                dtype=dtype,
                low=low,
                high=high,
                discrete=False
            )
        
        elif hasattr(obs_space, 'n'):
            # Discrete space
            return SpaceSpec(
                shape=(obs_space.n,),
                dtype=np.int64,
                discrete=True,
                n=obs_space.n
            )
        
        else:
            # Fallback for unknown space types
            logger.warning(f"Unknown observation space type: {type(obs_space)}")
            return SpaceSpec(
                shape=(1,),
                dtype=np.float32,
                discrete=False
            )
    
    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification from Gym environment"""
        action_space = self.env.action_space
        
        if hasattr(action_space, 'n'):
            # Discrete space (check this first since Discrete spaces also have 'shape')
            return SpaceSpec(
                shape=(action_space.n,),
                dtype=np.int64,
                discrete=True,
                n=action_space.n
            )
        
        elif hasattr(action_space, 'shape'):
            # Box space (continuous actions)
            shape = action_space.shape
            dtype = action_space.dtype if hasattr(action_space, 'dtype') else np.float32
            low = action_space.low if hasattr(action_space, 'low') else None
            high = action_space.high if hasattr(action_space, 'high') else None
            
            return SpaceSpec(
                shape=shape,
                dtype=dtype,
                low=low,
                high=high,
                discrete=False
            )
        
        else:
            # Fallback
            logger.warning(f"Unknown action space type: {type(action_space)}")
            return SpaceSpec(
                shape=(1,),
                dtype=np.float32,
                discrete=False
            )
    
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the Gym environment"""
        try:
            # New Gymnasium API returns (observation, info)
            result = self.env.reset(seed=seed)
            if isinstance(result, tuple):
                observation, info = result
            else:
                # Fallback for older Gym versions
                observation = result
            
            return np.array(observation, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            raise e
    
    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the Gym environment"""
        try:
            # Handle action format conversion
            if self.action_space.discrete:
                # For discrete actions, convert from tensor/array to integer
                if isinstance(action, np.ndarray):
                    if action.shape == () or action.shape == (1,):
                        # Handle numpy scalars properly - convert to Python int
                        action = int(action.item() if hasattr(action, 'item') else action)
                    else:
                        # Multi-dimensional array, take argmax
                        action = int(np.argmax(action))
                else:
                    # Handle scalars (including torch tensors converted to numpy)
                    try:
                        action = int(action.item() if hasattr(action, 'item') else action)
                    except (ValueError, TypeError):
                        # Fallback for unusual action formats
                        action = int(float(action))
            else:
                # For continuous actions, ensure proper shape and type
                action = np.array(action, dtype=np.float32)
                if action.shape != self.action_space.shape:
                    action = action.reshape(self.action_space.shape)
            
            # Step environment (handle both old and new Gym API)
            result = self.env.step(action)
            
            if len(result) == 5:
                # New Gymnasium API: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                info['terminated'] = terminated
                info['truncated'] = truncated
            elif len(result) == 4:
                # Old Gym API: (obs, reward, done, info)  
                obs, reward, done, info = result
            else:
                raise ValueError(f"Unexpected step result length: {len(result)}")
            
            return (
                np.array(obs, dtype=np.float32),
                float(reward),
                bool(done),
                info
            )
        
        except Exception as e:
            logger.error(f"Error stepping environment: {e}")
            raise e
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        try:
            if hasattr(self.env, 'render'):
                return self.env.render()
            else:
                logger.warning("Environment does not support rendering")
                return None
        except Exception as e:
            logger.warning(f"Error rendering environment: {e}")
            return None
    
    def close(self):
        """Close the environment"""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
            logger.debug(f"Closed Gym environment: {self.env_name}")
        except Exception as e:
            logger.warning(f"Error closing environment: {e}")
    
    def seed(self, seed: int):
        """Set environment seed for reproducibility"""
        try:
            if hasattr(self.env, 'seed'):
                return self.env.seed(seed)
            else:
                # For newer Gymnasium, seed is passed to reset()
                logger.info(f"Seed will be applied on next reset: {seed}")
                return [seed]
        except Exception as e:
            logger.warning(f"Error setting seed: {e}")
            return None
    
    def get_wrapper_info(self) -> Dict[str, Any]:
        """Get information about the wrapped environment"""
        info = {
            'env_name': self.env_name,
            'wrapper': 'gym',
            'observation_space': {
                'shape': self.observation_space.shape,
                'dtype': str(self.observation_space.dtype),
                'discrete': self.observation_space.discrete,
            },
            'action_space': {
                'shape': self.action_space.shape, 
                'dtype': str(self.action_space.dtype),
                'discrete': self.action_space.discrete,
            }
        }
        
        # Add bounds information if available
        if self.observation_space.low is not None:
            info['observation_space']['low'] = self.observation_space.low.tolist() if hasattr(self.observation_space.low, 'tolist') else self.observation_space.low
        if self.observation_space.high is not None:
            info['observation_space']['high'] = self.observation_space.high.tolist() if hasattr(self.observation_space.high, 'tolist') else self.observation_space.high
        
        if self.action_space.low is not None:
            info['action_space']['low'] = self.action_space.low.tolist() if hasattr(self.action_space.low, 'tolist') else self.action_space.low
        if self.action_space.high is not None:
            info['action_space']['high'] = self.action_space.high.tolist() if hasattr(self.action_space.high, 'tolist') else self.action_space.high
        
        if self.observation_space.discrete and self.observation_space.n is not None:
            info['observation_space']['n'] = self.observation_space.n
        if self.action_space.discrete and self.action_space.n is not None:
            info['action_space']['n'] = self.action_space.n
        
        return info