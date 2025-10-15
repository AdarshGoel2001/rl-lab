"""
Vectorized Gym Environment Wrapper

This module provides vectorized environment support using Gymnasium's vector environments.
It automatically chooses between SyncVectorEnv and AsyncVectorEnv based on hardware
and performance characteristics, enabling efficient parallel environment execution.

Key features:
- Automatic vectorization strategy selection (sync vs async)
- Hardware-aware optimization
- Batched observations, actions, and rewards
- Seamless integration with existing training loops
- Memory-efficient parallel execution
"""

import numpy as np
import logging
import psutil 
from typing import Dict, Any, Tuple, Optional, Union, Callable, List
import torch

from src.environments.base import BaseEnvironment, SpaceSpec
from src.utils.registry import register_environment

logger = logging.getLogger(__name__)


@register_environment("vectorized_gym")
class VectorizedGymWrapper(BaseEnvironment):
    """
    Vectorized wrapper for OpenAI Gym environments using Gymnasium's vector environments.
    
    This wrapper creates multiple parallel environment instances and provides
    batched operations for efficient training. It automatically selects the
    optimal vectorization strategy based on system resources.
    
    Attributes:
        vec_env: The underlying vectorized environment
        num_envs: Number of parallel environments
        env_name: Name of the base Gym environment
        vectorization_type: Type of vectorization used ('sync' or 'async')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vectorized Gym environment wrapper.
        
        Args:
            config: Environment configuration containing:
                - name: Gym environment name (e.g., 'CartPole-v1')
                - num_envs: Number of parallel environments (default: 8)
                - vectorization: 'sync', 'async', or 'auto' (default: 'auto')
                - normalize_obs: Whether to normalize observations
                - normalize_reward: Whether to normalize rewards
                - max_episode_steps: Maximum steps per episode
                - render_mode: Rendering mode (None for vectorized envs)
                - env_kwargs: Additional environment-specific kwargs
        """
        self.env_name = config['name']
        self.num_envs = config.get('num_envs', 8)
        self.vectorization = config.get('vectorization', 'auto')
        self.env_kwargs = config.get('env_kwargs', {})
        
        # Set vectorized environment flags
        self.is_vectorized = True
        
        # Disable rendering for vectorized environments (not supported)
        if config.get('render_mode') is not None:
            logger.warning("Rendering not supported for vectorized environments. Disabling render_mode.")
            config = config.copy()
            config['render_mode'] = None
        
        # Store configuration before calling super().__init__
        self._vectorized_config = config
        
        super().__init__(config)
    
    def _setup_environment(self):
        """Setup the vectorized Gym environment with per-environment wrappers"""
        try:
            import gymnasium as gym
            from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
            
            # Determine vectorization strategy
            use_async = self._should_use_async()
            self.vectorization_type = 'async' if use_async else 'sync'
            
            # Create per-environment factory functions  
            env_id = self.env_name
            env_kwargs = dict(self.env_kwargs)
            def make_single_env():
                def _make():
                    # Create raw gym environment (no transforms here)
                    return gym.make(env_id, **env_kwargs)
                return _make
            
            # Create environment factory functions, one per environment
            env_fns = []
            for i in range(self.num_envs):
                env_fns.append(make_single_env())
            
            # Create vectorized environment
            if use_async:
                self.vec_env = AsyncVectorEnv(env_fns)
                logger.info(f"Created AsyncVectorEnv with {self.num_envs} {self.env_name} environments")
            else:
                self.vec_env = SyncVectorEnv(env_fns)
                logger.info(f"Created SyncVectorEnv with {self.num_envs} {self.env_name} environments")
            
            # Create per-environment transform pipelines for isolated state
            self._transform_pipelines = []
            if self.config.get('observation_transforms'):
                try:
                    from src.environments.transforms import create_transform_pipeline, expand_preset_configs
                    
                    # Expand any preset configurations
                    transform_configs = self.config.get('observation_transforms', [])
                    expanded_configs = expand_preset_configs(transform_configs)
                    
                    # Create isolated transform pipeline for each environment
                    for i in range(self.num_envs):
                        pipeline = create_transform_pipeline(expanded_configs)
                        self._transform_pipelines.append(pipeline)
                        
                    logger.info(f"Created {len(self._transform_pipelines)} isolated transform pipelines")
                    
                except Exception as e:
                    logger.error(f"Failed to setup transform pipelines: {e}")
                    self._transform_pipelines = [None] * self.num_envs
            else:
                self._transform_pipelines = [None] * self.num_envs
            
            # Initialize episode tracking for each environment
            self._episode_returns = np.zeros(self.num_envs)
            self._episode_lengths = np.zeros(self.num_envs, dtype=int)
            self._episode_counts = np.zeros(self.num_envs, dtype=int)
            
        except ImportError as e:
            logger.error("Gymnasium not installed or vector environments not available. Install with: pip install gymnasium")
            raise e
        except Exception as e:
            logger.error(f"Failed to create vectorized Gym environment {self.env_name}: {e}")
            raise e
    
    def _should_use_async(self) -> bool:
        """
        Determine whether to use async or sync vectorization based on system resources.
        
        Returns:
            True if async vectorization should be used, False for sync
        """
        if self.vectorization == 'sync':
            return False
        elif self.vectorization == 'async':
            return True
        elif self.vectorization == 'auto':
            # Auto-detection logic based on research findings
            available_memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            # Use async if:
            # 1. High memory (16GB+) for handling subprocess overhead
            # 2. Sufficient CPU cores (4+) for parallel processing
            # 3. Not too many environments (avoid excessive subprocess overhead)
            if available_memory_gb >= 16 and cpu_count >= 4 and self.num_envs <= 16:
                logger.info(f"Auto-selected async vectorization (RAM: {available_memory_gb:.1f}GB, CPUs: {cpu_count})")
                return True
            else:
                logger.info(f"Auto-selected sync vectorization (RAM: {available_memory_gb:.1f}GB, CPUs: {cpu_count})")
                return False
        else:
            logger.warning(f"Unknown vectorization type: {self.vectorization}. Using sync.")
            return False
    
    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification from vectorized environment"""
        # Get single environment observation space
        single_obs_space = self.vec_env.single_observation_space
        
        if hasattr(single_obs_space, 'shape'):
            # Box or similar space - add batch dimension
            shape = (self.num_envs,) + single_obs_space.shape
            dtype = single_obs_space.dtype if hasattr(single_obs_space, 'dtype') else np.float32
            low = single_obs_space.low if hasattr(single_obs_space, 'low') else None
            high = single_obs_space.high if hasattr(single_obs_space, 'high') else None
            
            return SpaceSpec(
                shape=shape,
                dtype=dtype,
                low=low,
                high=high,
                discrete=False
            )
        
        elif hasattr(single_obs_space, 'n'):
            # Discrete space - add batch dimension
            return SpaceSpec(
                shape=(self.num_envs, single_obs_space.n),
                dtype=np.int64,
                discrete=True,
                n=single_obs_space.n
            )
        
        else:
            # Fallback
            logger.warning(f"Unknown observation space type: {type(single_obs_space)}")
            return SpaceSpec(
                shape=(self.num_envs, 1),
                dtype=np.float32,
                discrete=False
            )
    
    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification from vectorized environment"""
        # Get single environment action space
        single_action_space = self.vec_env.single_action_space
        
        if hasattr(single_action_space, 'n'):
            # Discrete space - add batch dimension
            return SpaceSpec(
                shape=(self.num_envs, single_action_space.n),
                dtype=np.int64,
                discrete=True,
                n=single_action_space.n
            )
        
        elif hasattr(single_action_space, 'shape'):
            # Box space - add batch dimension
            shape = (self.num_envs,) + single_action_space.shape
            dtype = single_action_space.dtype if hasattr(single_action_space, 'dtype') else np.float32
            low = single_action_space.low if hasattr(single_action_space, 'low') else None
            high = single_action_space.high if hasattr(single_action_space, 'high') else None
            
            return SpaceSpec(
                shape=shape,
                dtype=dtype,
                low=low,
                high=high,
                discrete=False
            )
        
        else:
            # Fallback
            logger.warning(f"Unknown action space type: {type(single_action_space)}")
            return SpaceSpec(
                shape=(self.num_envs, 1),
                dtype=np.float32,
                discrete=False
            )
    
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset all vectorized environments"""
        try:
            # Reset episode tracking
            self._episode_returns.fill(0.0)
            self._episode_lengths.fill(0)
            
            # Reset transform states for all environments
            for pipeline in self._transform_pipelines:
                if pipeline is not None:
                    pipeline.reset_states()
            
            # Reset vectorized environment
            if seed is not None:
                # Create seeds for each environment
                seeds = [seed + i for i in range(self.num_envs)]
                observations, infos = self.vec_env.reset(seed=seeds)
            else:
                observations, infos = self.vec_env.reset()
            
            # Apply per-environment transforms
            observations = self._apply_per_env_transforms(observations)
            
            return np.array(observations, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error resetting vectorized environment: {e}")
            raise e
    
    def _apply_per_env_transforms(self, observations: np.ndarray) -> np.ndarray:
        """
        Apply transforms to each environment's observation independently.
        
        Args:
            observations: Batch of observations (num_envs, ...)
            
        Returns:
            Transformed observations batch
        """
        if not self._transform_pipelines or all(p is None for p in self._transform_pipelines):
            return observations
        
        transformed_obs = []
        for i, (obs, pipeline) in enumerate(zip(observations, self._transform_pipelines)):
            if pipeline is not None:
                try:
                    transformed_obs.append(pipeline.apply(obs))
                except Exception as e:
                    logger.error(f"Transform failed for env {i}: {e}")
                    transformed_obs.append(obs)  # Fallback to original
            else:
                transformed_obs.append(obs)
        
        return np.array(transformed_obs)
    
    def _step_environment(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all vectorized environments.
        
        Args:
            actions: Batch of actions with shape (num_envs, action_dim)
            
        Returns:
            Tuple of (observations, rewards, dones, infos) where each is batched
        """
        try:
            # Handle action format conversion for vectorized environments
            if self.action_space.discrete:
                # For discrete actions, ensure proper shape
                if actions.ndim == 2 and actions.shape[1] > 1:
                    # Convert from one-hot or logits to action indices
                    actions = np.argmax(actions, axis=1)
                elif actions.ndim == 1:
                    # Already in correct format
                    pass
                else:
                    # Handle other cases
                    actions = actions.reshape(-1)
                
                # Ensure we have exactly num_envs actions
                if len(actions) != self.num_envs:
                    raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
                
                actions = actions.astype(np.int64)
            else:
                # For continuous actions, ensure proper shape
                actions = np.array(actions, dtype=np.float32)
                if actions.shape[0] != self.num_envs:
                    raise ValueError(f"Expected batch size {self.num_envs}, got {actions.shape[0]}")
            
            # Step vectorized environment
            observations, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
            
            # Apply per-environment transforms
            observations = self._apply_per_env_transforms(observations)
            
            # Handle done flags (terminated OR truncated)
            dones = terminateds | truncateds
            
            # Normalize infos to list-of-dicts format
            # Gymnasium vectorized envs can return dict-of-arrays or list-of-dicts
            infos = self._normalize_infos(infos)
            
            # Update episode tracking
            self._episode_returns += rewards
            self._episode_lengths += 1
            
            # Handle episode completion
            for i, done in enumerate(dones):
                if done:
                    self._episode_counts[i] += 1
                    
                    # Add episode info (infos is guaranteed to be proper list-of-dicts now)
                    infos[i]['episode_return'] = float(self._episode_returns[i])
                    infos[i]['episode_length'] = int(self._episode_lengths[i])
                    infos[i]['episode_count'] = int(self._episode_counts[i])
                    
                    # Reset tracking for this environment
                    self._episode_returns[i] = 0.0
                    self._episode_lengths[i] = 0
            
            return (
                np.array(observations, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool),
                infos
            )
        
        except Exception as e:
            logger.error(f"Error stepping vectorized environment: {e}")
            raise e
    
    def _normalize_infos(self, infos) -> List[Dict[str, Any]]:
        """
        Normalize infos to list-of-dicts format.
        
        Gymnasium vectorized environments can return infos as:
        1. List of dicts: [{'key': val}, {'key': val}, ...]  
        2. Dict of arrays: {'key': [val1, val2, ...], '_final_info': [...]}
        3. None or malformed data
        
        Args:
            infos: Raw infos from vectorized environment
            
        Returns:
            List of dictionaries, one per environment
        """
        if infos is None:
            return [{} for _ in range(self.num_envs)]
        
        if isinstance(infos, list):
            # Case 1: Already list-of-dicts, ensure correct length and types
            info_list = []
            for i in range(self.num_envs):
                if i < len(infos) and isinstance(infos[i], dict):
                    info_list.append(infos[i])
                else:
                    info_list.append({})
            return info_list
            
        elif isinstance(infos, dict):
            # Case 2: Dict-of-arrays format from Gymnasium
            info_list = [{} for _ in range(self.num_envs)]
            
            # Distribute array values to individual environment dicts
            for key, values in infos.items():
                if key == '_final_info':
                    # Special case: final_info contains episode completion data
                    if isinstance(values, list) and len(values) >= self.num_envs:
                        for i in range(self.num_envs):
                            if values[i] is not None and isinstance(values[i], dict):
                                info_list[i].update(values[i])
                elif hasattr(values, '__len__') and len(values) >= self.num_envs:
                    # Regular array: distribute values[i] to info_list[i][key]
                    for i in range(self.num_envs):
                        try:
                            info_list[i][key] = values[i]
                        except (IndexError, TypeError):
                            # Skip if value can't be assigned
                            pass
                            
            return info_list
            
        else:
            # Case 3: Unknown format, return empty dicts
            logger.warning(f"Unknown infos format: {type(infos)}, using empty dicts")
            return [{} for _ in range(self.num_envs)]
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset all environments and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observations as numpy array with shape (num_envs, obs_dim)
        """
        # Reset underlying environments
        obs = self._reset_environment(seed)
        
        # Apply normalization if enabled
        if self.normalize_obs:
            obs = self._normalize_observation_batch(obs)
        
        return np.asarray(obs, dtype=np.float32)
    
    def step(self, actions: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Execute batched actions and return results.
        
        Args:
            actions: Batch of actions with shape (num_envs, action_dim)
            
        Returns:
            Tuple of (observations, rewards, dones, infos) where:
            - observations: numpy array (num_envs, obs_dim)
            - rewards: numpy array (num_envs,)
            - dones: numpy array (num_envs,)
            - infos: List of info dictionaries, one per environment
        """
        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # Step underlying environments
        obs, rewards, dones, infos = self._step_environment(actions)
        
        # Apply observation normalization
        if self.normalize_obs:
            obs = self._normalize_observation_batch(obs)
        
        # Apply reward normalization
        if self.normalize_reward:
            rewards = self._normalize_reward_batch(rewards)
        
        obs_out = np.asarray(obs, dtype=np.float32)
        rewards_out = np.asarray(rewards, dtype=np.float32)
        dones_out = np.asarray(dones, dtype=bool)
        return obs_out, rewards_out, dones_out, infos
    
    def _normalize_observation_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Apply observation normalization to a batch of observations"""
        if self._obs_mean is None:
            # Initialize with first observation
            self._obs_mean = np.mean(obs_batch, axis=0)
            self._obs_var = np.var(obs_batch, axis=0)
            return obs_batch
        
        # Update running statistics with batch
        alpha = 0.01
        batch_mean = np.mean(obs_batch, axis=0)
        batch_var = np.var(obs_batch, axis=0)
        
        self._obs_mean = (1 - alpha) * self._obs_mean + alpha * batch_mean
        self._obs_var = (1 - alpha) * self._obs_var + alpha * batch_var
        
        # Normalize batch
        return (obs_batch - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
    
    def _normalize_reward_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Apply reward normalization to a batch of rewards"""
        # Update running statistics with batch
        alpha = 0.01
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        self._reward_mean = (1 - alpha) * self._reward_mean + alpha * batch_mean
        self._reward_var = (1 - alpha) * self._reward_var + alpha * batch_var
        
        # Normalize batch
        return (rewards - self._reward_mean) / (np.sqrt(self._reward_var) + 1e-8)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get environment metrics for logging.
        
        Returns:
            Dictionary of aggregated environment metrics across all parallel environments
        """
        metrics = {
            'env/num_envs': float(self.num_envs),
            'env/vectorization_type': 1.0 if self.vectorization_type == 'async' else 0.0,
            'env/total_episode_count': float(np.sum(self._episode_counts)),
            'env/mean_episode_length': float(np.mean(self._episode_lengths[self._episode_lengths > 0])) if np.any(self._episode_lengths > 0) else 0.0,
            'env/mean_episode_return': float(np.mean(self._episode_returns)),
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
        """Close all vectorized environments"""
        try:
            if hasattr(self.vec_env, 'close'):
                self.vec_env.close()
            logger.debug(f"Closed vectorized Gym environment: {self.env_name} ({self.num_envs} envs)")
        except Exception as e:
            logger.warning(f"Error closing vectorized environment: {e}")
    
    def render(self, mode: str = 'human'):
        """Rendering not supported for vectorized environments"""
        logger.warning("Rendering not supported for vectorized environments")
        return None
    
    def seed(self, seed: int):
        """Set environment seed for reproducibility"""
        try:
            # For vectorized environments, seed on next reset
            logger.info(f"Seed {seed} will be applied on next reset for all {self.num_envs} environments")
            # We could store the seed and use it in reset, but for now just log
            # The trainer will call reset(seed=seed) which handles seeding properly
            return [seed] * self.num_envs
        except Exception as e:
            logger.warning(f"Error setting seed for vectorized environment: {e}")
            return None
    
    def get_wrapper_info(self) -> Dict[str, Any]:
        """Get information about the vectorized environment wrapper"""
        info = {
            'env_name': self.env_name,
            'wrapper': 'vectorized_gym',
            'num_envs': self.num_envs,
            'vectorization_type': self.vectorization_type,
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
        
        # Add single environment space info
        single_obs_shape = self.observation_space.shape[1:] if len(self.observation_space.shape) > 1 else (1,)
        single_action_shape = self.action_space.shape[1:] if len(self.action_space.shape) > 1 else (1,)
        
        info['single_env'] = {
            'observation_shape': single_obs_shape,
            'action_shape': single_action_shape,
        }
        
        # Add discrete space info
        if self.observation_space.discrete and self.observation_space.n is not None:
            info['observation_space']['n'] = self.observation_space.n
        if self.action_space.discrete and self.action_space.n is not None:
            info['action_space']['n'] = self.action_space.n
        
        return info
