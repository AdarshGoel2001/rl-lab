"""
Trajectory Buffer Implementation

This module implements trajectory buffers for on-policy reinforcement learning
algorithms like PPO, A2C, and REINFORCE. It stores complete episodes and
provides methods for computing returns, advantages, and sampling.

Key features:
- Stores complete trajectories with episode boundaries
- Computes discounted returns and GAE advantages  
- Supports both episodic and truncated trajectory handling
- Efficient batching and sampling for training
- Automatic capacity management
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import logging

from src.buffers.base import BaseBuffer, Experience
from src.utils.registry import register_buffer

logger = logging.getLogger(__name__)


@register_buffer("trajectory")
class TrajectoryBuffer(BaseBuffer):
    """
    Buffer for storing complete trajectories for on-policy RL algorithms.
    
    This buffer is designed for algorithms that need complete episode
    information like PPO, A2C, and REINFORCE. It stores trajectories
    and provides methods for computing returns and advantages.
    
    Attributes:
        trajectories: List of stored complete trajectories
        current_trajectory: Currently being recorded trajectory
        gamma: Discount factor for return computation
        gae_lambda: GAE lambda parameter for advantage computation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory buffer.
        
        Args:
            config: Buffer configuration containing:
                - capacity: Maximum number of transitions to store
                - gamma: Discount factor (default: 0.99)
                - gae_lambda: GAE lambda parameter (default: 0.95) 
                - normalize_advantages: Whether to normalize advantages
                - compute_returns: Whether to compute returns automatically
        """
        # Set trajectory-specific defaults
        config.setdefault('gamma', 0.99)
        config.setdefault('gae_lambda', 0.95) 
        config.setdefault('normalize_advantages', True)
        config.setdefault('compute_returns', True)
        
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.normalize_advantages = config['normalize_advantages']
        self.compute_returns = config['compute_returns']
        
        super().__init__(config)
    
    def _setup_storage(self):
        """Initialize trajectory storage"""
        self.trajectories: List[Dict[str, List]] = []
        self.current_trajectory: Dict[str, List] = defaultdict(list)
        self._trajectory_complete = True
    
    def add_step(self, observation: np.ndarray, action: np.ndarray, 
                 reward: float, next_observation: np.ndarray, done: bool,
                 value: Optional[float] = None, log_prob: Optional[float] = None,
                 **kwargs):
        """
        Add a single step to the current trajectory.
        
        Args:
            observation: Current observation
            action: Action taken  
            reward: Reward received
            next_observation: Next observation
            done: Episode termination flag
            value: Value estimate (for advantage computation)
            log_prob: Log probability of action (for policy gradient, stored as old_log_probs)
            **kwargs: Additional data to store with the step
        """
        # Start new trajectory if needed
        if self._trajectory_complete:
            self.current_trajectory = defaultdict(list)
            self._trajectory_complete = False
        
        # Store step data
        self.current_trajectory['observations'].append(observation)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['next_observations'].append(next_observation)
        self.current_trajectory['dones'].append(done)
        
        # Store optional data
        if value is not None:
            self.current_trajectory['values'].append(value)
        if log_prob is not None:
            self.current_trajectory['log_probs'].append(log_prob)
        
        # Store additional keyword arguments
        for key, value in kwargs.items():
            self.current_trajectory[key].append(value)
        
        # Complete trajectory if episode is done
        if done:
            self._complete_trajectory()
    
    def add_trajectory(self, trajectory: Dict[str, List]):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            trajectory: Dictionary containing trajectory data with keys:
                - observations: List of observations
                - actions: List of actions
                - rewards: List of rewards  
                - dones: List of done flags
                - old_values: List of value estimates from policy that collected data (optional)
                - old_log_probs: List of log probabilities from policy that collected data (optional)
        """
        if len(trajectory.get('observations', [])) == 0:
            logger.warning("Attempted to add empty trajectory")
            return
        
        # Compute returns and advantages if requested
        if self.compute_returns:
            trajectory = self._compute_trajectory_statistics(trajectory)
        
        # Add trajectory to buffer
        self.trajectories.append(trajectory)
        
        # Calculate correct size for vectorized trajectories
        observations = trajectory['observations']
        if isinstance(observations, np.ndarray) and observations.ndim >= 2:
            # Vectorized case: (T, B, ...) -> total experiences = T * B
            trajectory_size = observations.shape[0] * observations.shape[1]
        else:
            # Single environment case: (T, ...) -> total experiences = T
            trajectory_size = len(observations)
        
        self._size += trajectory_size
        
        # Remove old trajectories if over capacity
        self._enforce_capacity()
        
        logger.debug(f"Added trajectory with {trajectory_size} individual experiences")
    
    def _complete_trajectory(self):
        """Complete the current trajectory and add it to buffer"""
        if len(self.current_trajectory['observations']) == 0:
            return
        
        # Convert lists to numpy arrays
        trajectory = {}
        for key, values in self.current_trajectory.items():
            trajectory[key] = np.array(values)
        
        # Compute returns and advantages if requested
        if self.compute_returns:
            trajectory = self._compute_trajectory_statistics(trajectory)
        
        # Add to buffer
        self.add_trajectory(trajectory)
        
        # Mark trajectory as complete
        self._trajectory_complete = True
        self.current_trajectory = defaultdict(list)
    
    def _compute_trajectory_statistics(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute returns and advantages for a trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            Trajectory with added returns and advantages
        """
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        
        # Compute discounted returns
        returns = self._compute_returns(rewards, dones)
        trajectory['returns'] = returns
        
        # Compute GAE advantages if old_values are provided
        if 'old_values' in trajectory:
            values = trajectory['old_values']
            bootstrap_value = trajectory.get('bootstrap_value', 0.0)
            advantages = self._compute_gae_advantages(rewards, values, dones, bootstrap_value)
            trajectory['advantages'] = advantages
        
        return trajectory
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns for a trajectory.
        
        Args:
            rewards: Array of rewards - shape (T, B) where T=timesteps, B=batch_size/num_envs
            dones: Array of done flags - shape (T, B) 
            
        Returns:
            Array of discounted returns with same shape as rewards (T, B)
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # Handle vectorized environments case: (T, B) shape
        if rewards.ndim == 2:
            num_steps, num_envs = rewards.shape
            running_returns = np.zeros(num_envs, dtype=np.float32)
            
            # Compute returns backward through trajectory for each environment
            for t in reversed(range(num_steps)):
                # Reset running return for environments that are done
                running_returns[dones[t]] = 0.0
                # Update running returns for all environments
                running_returns = rewards[t] + self.gamma * running_returns
                returns[t] = running_returns.copy()
                
        # Handle single environment case: (T,) shape (legacy support)
        elif rewards.ndim == 1:
            running_return = 0.0
            
            # Compute returns backward through trajectory
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0.0
                running_return = rewards[t] + self.gamma * running_return
                returns[t] = running_return
                
        else:
            raise ValueError(f"Rewards array must be 1D or 2D, got shape {rewards.shape}")
        
        return returns
    
    def _compute_gae_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                               dones: np.ndarray, bootstrap_value: float = 0.0) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE) advantages.
        
        Args:
            rewards: Array of rewards - shape (T, B) or (T,)
            values: Array of value estimates - shape (T, B) or (T,)
            dones: Array of done flags - shape (T, B) or (T,)
            bootstrap_value: Value estimate for final state (for truncated episodes)
            
        Returns:
            Array of GAE advantages with same shape as rewards
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Handle vectorized environments case: (T, B) shape
        if rewards.ndim == 2:
            num_steps, num_envs = rewards.shape
            running_advantages = np.zeros(num_envs, dtype=np.float32)
            
            # Handle bootstrap values for vectorized case
            if isinstance(bootstrap_value, (int, float)):
                bootstrap_values = np.full(num_envs, bootstrap_value, dtype=np.float32)
            else:
                bootstrap_values = np.array(bootstrap_value, dtype=np.float32)
                if bootstrap_values.shape != (num_envs,):
                    bootstrap_values = np.full(num_envs, 0.0, dtype=np.float32)
            
            # Compute advantages backward through trajectory for each environment
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    # Use bootstrap values for final step if episodes are truncated
                    next_values = bootstrap_values
                else:
                    next_values = values[t + 1]
                
                # Reset next value and running advantage for terminated environments
                next_values = next_values.copy()
                next_values[dones[t]] = 0.0
                running_advantages[dones[t]] = 0.0
                
                # TD error for all environments
                deltas = rewards[t] + self.gamma * next_values - values[t]
                
                # GAE advantage for all environments  
                running_advantages = deltas + self.gamma * self.gae_lambda * running_advantages
                advantages[t] = running_advantages.copy()
                
        # Handle single environment case: (T,) shape (legacy support)
        elif rewards.ndim == 1:
            running_advantage = 0.0
            
            # Compute advantages backward through trajectory
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    # CRITICAL FIX: Use bootstrap value for final step if episode is truncated
                    next_value = bootstrap_value
                else:
                    next_value = values[t + 1]
                
                if dones[t]:
                    # Only zero out if actually terminated (not truncated)
                    next_value = 0.0
                    running_advantage = 0.0
                
                # TD error
                delta = rewards[t] + self.gamma * next_value - values[t]
                
                # GAE advantage
                running_advantage = delta + self.gamma * self.gae_lambda * running_advantage
                advantages[t] = running_advantage
                
        else:
            raise ValueError(f"Rewards array must be 1D or 2D, got shape {rewards.shape}")
        
        return advantages
    
    def _enforce_capacity(self):
        """Remove old trajectories to stay within capacity
        
        For trajectory buffers, we need to ensure at least one full rollout
        (batch_size experiences) is retained to enable training.
        """
        # Use the larger of capacity or batch_size as the actual limit
        # This prevents evicting a just-collected rollout needed for training
        effective_capacity = max(self.capacity, self.batch_size)
        
        while self._size > effective_capacity and len(self.trajectories) > 0:
            removed_trajectory = self.trajectories.pop(0)
            
            # Calculate correct size for removed trajectory
            observations = removed_trajectory['observations']
            if isinstance(observations, np.ndarray) and observations.ndim >= 2:
                # Vectorized case: (T, B, ...) -> total experiences = T * B
                removed_size = observations.shape[0] * observations.shape[1]
            else:
                # Single environment case: (T, ...) -> total experiences = T
                removed_size = len(observations)
            
            self._size -= removed_size
            logger.debug(f"DEBUGGING: Removed trajectory of size {removed_size}, buffer size now {self._size}")
    
    def add(self, **kwargs):
        """Add experience(s) to buffer - supports both step and trajectory addition"""
        if 'trajectory' in kwargs:
            self.add_trajectory(kwargs['trajectory'])
        else:
            # Extract step data with defaults
            observation = kwargs['observation']
            action = kwargs['action'] 
            reward = kwargs['reward']
            next_observation = kwargs['next_observation']
            done = kwargs['done']
            value = kwargs.get('value', None)
            log_prob = kwargs.get('log_prob', None)
            
            # Remove known keys and pass rest as additional data
            additional_data = {k: v for k, v in kwargs.items() 
                             if k not in ['observation', 'action', 'reward', 
                                        'next_observation', 'done', 'value', 'log_prob']}
            
            self.add_step(observation, action, reward, next_observation, done,
                         value, log_prob, **additional_data)
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from all trajectories.
        
        Args:
            batch_size: Number of experiences to sample (uses config default if None)
            
        Returns:
            Dictionary containing batched experience tensors
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} experiences but {batch_size} requested")
        
        # Flatten all trajectories into individual steps
        all_data = defaultdict(list)
        
        for trajectory in self.trajectories:
            for key, values in trajectory.items():
                # Skip bootstrap_value as it's trajectory-level metadata, not step data
                if key == 'bootstrap_value':
                    continue
                    
                if isinstance(values, np.ndarray):
                    # Handle vectorized data properly
                    if values.ndim >= 2:
                        # Vectorized data with shape (T, B, ...) where T=time steps, B=batch/num_envs
                        # We need to iterate through time and batch dimensions to get individual experiences
                        T, B = values.shape[0], values.shape[1]
                        for t in range(T):
                            for b in range(B):
                                # values[t, b] gives us one complete experience
                                # For observations: shape (84, 84, 4)
                                # For actions: scalar
                                # For rewards: scalar
                                all_data[key].append(values[t, b])
                    elif values.ndim == 1:
                        # Single environment case (T,) or special cases
                        if key == 'bootstrap_values':
                            # Skip bootstrap values - they're trajectory-level metadata
                            continue
                        else:
                            # Single env case: (T,) -> extend as list
                            all_data[key].extend(values.tolist())
                    else:
                        # Scalar or other cases
                        all_data[key].append(values)
                else:
                    all_data[key].append(values)
        
        # Sample random indices
        total_steps = len(all_data['observations']) if 'observations' in all_data else 0
        if total_steps == 0:
            raise ValueError("No observations found in buffer")
        
        indices = np.random.choice(total_steps, size=min(batch_size, total_steps), replace=False)
        
        # Create batch dictionary
        batch = {}
        for key, values in all_data.items():
            if len(values) > 0:
                # Check if all keys have same length
                if len(values) != total_steps:
                    # Skip keys with wrong length to avoid crashes
                    # (This is expected for keys like bootstrap_values which are trajectory-level)
                    continue
                sampled_values = [values[i] for i in indices]
                
                # Convert to numpy array with proper handling for different data shapes
                if len(sampled_values) > 0:
                    first_item = sampled_values[0]
                    if isinstance(first_item, np.ndarray) and first_item.ndim > 0:
                        # For multi-dimensional data (like observations), stack along new batch dimension
                        batch[key] = self.to_tensor(np.stack(sampled_values, axis=0))
                    else:
                        # For scalar data (like actions, rewards), convert to array normally
                        batch[key] = self.to_tensor(np.array(sampled_values))
                else:
                    # Empty case - should not happen but handle gracefully
                    batch[key] = self.to_tensor(np.array(sampled_values))
        
        # Normalize advantages if present and requested
        if 'advantages' in batch and self.normalize_advantages:
            batch['advantages'] = self._normalize_advantages(batch['advantages'])
        
        return batch
    
    def sample_all(self) -> Dict[str, torch.Tensor]:
        """
        Sample all experiences in buffer (useful for on-policy algorithms).
        
        Returns:
            Dictionary containing all experiences as tensors
        """
        return self.sample(batch_size=self._size)
    
    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Normalize advantages to zero mean and unit variance"""
        if len(advantages) <= 1:
            return advantages
        
        mean_adv = advantages.mean()
        std_adv = advantages.std()
        
        if std_adv > 1e-8:
            return (advantages - mean_adv) / std_adv
        else:
            return advantages - mean_adv
    
    def clear(self):
        """Clear all trajectories and current trajectory"""
        self.trajectories = []
        self.current_trajectory = defaultdict(list)
        self._size = 0
        self._position = 0
        self._trajectory_complete = True
        
        logger.debug("Trajectory buffer cleared")
    
    def finish_trajectory(self, final_value: float = 0.0):
        """
        Manually finish the current trajectory (useful for truncated episodes).
        
        Args:
            final_value: Value estimate for the final state (for bootstrap)
        """
        if not self._trajectory_complete and len(self.current_trajectory['observations']) > 0:
            # Add final value for bootstrap if provided
            if final_value != 0.0 and 'values' in self.current_trajectory:
                self.current_trajectory['values'].append(final_value)
            
            self._complete_trajectory()
    
    def get_trajectory_lengths(self) -> List[int]:
        """Get lengths of all stored trajectories"""
        return [len(traj['observations']) for traj in self.trajectories]
    
    def get_trajectory_returns(self) -> List[float]:
        """Get total returns for all stored trajectories"""
        returns = []
        for traj in self.trajectories:
            if 'rewards' in traj:
                returns.append(float(np.sum(traj['rewards'])))
        return returns
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete buffer state for checkpointing.

        Returns:
            Dictionary containing all buffer state including trajectories
            and configuration parameters.
        """
        return {
            'trajectories': self.trajectories,
            'current_trajectory': dict(self.current_trajectory),  # Convert defaultdict to dict
            'trajectory_complete': self._trajectory_complete,
            'size': self._size,
            'position': self._position,
            'capacity': self.capacity,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'normalize_advantages': self.normalize_advantages,
            'compute_returns': self.compute_returns,
        }

    def set_state(self, state: Dict[str, Any]):
        """
        Restore buffer state from checkpoint.

        Args:
            state: Dictionary containing buffer state from get_state()
        """
        self.trajectories = state.get('trajectories', [])
        self.current_trajectory = defaultdict(list, state.get('current_trajectory', {}))
        self._trajectory_complete = state.get('trajectory_complete', True)
        self._position = state.get('position', 0)

        # Restore configuration if present
        if 'gamma' in state:
            self.gamma = state['gamma']
        if 'gae_lambda' in state:
            self.gae_lambda = state['gae_lambda']

        # Recalculate size using correct vectorized logic
        self._size = 0
        for traj in self.trajectories:
            observations = traj.get('observations', [])
            if isinstance(observations, np.ndarray) and observations.ndim >= 2:
                # Vectorized case: (T, B, ...) -> total experiences = T * B
                self._size += observations.shape[0] * observations.shape[1]
            else:
                # Single environment case: (T, ...) -> total experiences = T
                self._size += len(observations)

        # Add current trajectory size
        current_obs = self.current_trajectory.get('observations', [])
        if len(current_obs) > 0:
            if isinstance(current_obs, list):
                self._size += len(current_obs)
            elif isinstance(current_obs, np.ndarray) and current_obs.ndim >= 2:
                self._size += current_obs.shape[0] * current_obs.shape[1]
            else:
                self._size += len(current_obs)

    # Legacy methods for backwards compatibility
    def save_checkpoint(self) -> Dict[str, Any]:
        """Legacy method - use get_state() instead"""
        return self.get_state()

    def load_checkpoint(self, state: Dict[str, Any]):
        """Legacy method - use set_state() instead"""
        self.set_state(state)
    
    def ready(self) -> bool:
        """
        Check if buffer has enough experiences for PPO training.
        
        For PPO, we want to train when we've collected a full rollout.
        With vectorized environments, this means capacity steps per environment.
        
        Returns:
            True if buffer has collected enough experiences for training
        """
        # For vectorized environments: capacity is per-env, so total = capacity * num_envs
        # But we use batch_size which should equal capacity * num_envs
        is_ready = self._size >= self.batch_size
        
        # DEBUGGING: Log buffer state to diagnose ready() issue - REMOVE LATER
        logger.debug(f"DEBUGGING: Buffer ready check: size={self._size}, batch_size={self.batch_size}, ready={is_ready}")
        logger.debug(f"DEBUGGING: Buffer trajectories: {len(self.trajectories)}, current_trajectory_complete: {self._trajectory_complete}")
        if len(self.trajectories) > 0:
            traj_shapes = []
            for i, traj in enumerate(self.trajectories):
                obs_shape = traj['observations'].shape if isinstance(traj['observations'], np.ndarray) else len(traj['observations'])
                traj_shapes.append(f"traj_{i}: {obs_shape}")
            logger.debug(f"DEBUGGING: Trajectory shapes: {', '.join(traj_shapes)}")
        
        return is_ready