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
            log_prob: Log probability of action (for policy gradient)
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
                - values: List of value estimates (optional)
                - log_probs: List of log probabilities (optional)
        """
        if len(trajectory.get('observations', [])) == 0:
            logger.warning("Attempted to add empty trajectory")
            return
        
        # Add trajectory to buffer
        self.trajectories.append(trajectory)
        self._size += len(trajectory['observations'])
        
        # Remove old trajectories if over capacity
        self._enforce_capacity()
        
        logger.debug(f"Added trajectory with {len(trajectory['observations'])} steps")
    
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
        
        # Compute GAE advantages if values are provided
        if 'values' in trajectory:
            values = trajectory['values']
            advantages = self._compute_gae_advantages(rewards, values, dones)
            trajectory['advantages'] = advantages
        
        return trajectory
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Compute discounted returns for a trajectory.
        
        Args:
            rewards: Array of rewards
            dones: Array of done flags
            
        Returns:
            Array of discounted returns
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        # Compute returns backward through trajectory
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _compute_gae_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                               dones: np.ndarray) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE) advantages.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            
        Returns:
            Array of GAE advantages
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        running_advantage = 0.0
        
        # Compute advantages backward through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0.0
                running_advantage = 0.0
            
            # TD error
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE advantage
            running_advantage = delta + self.gamma * self.gae_lambda * running_advantage
            advantages[t] = running_advantage
        
        return advantages
    
    def _enforce_capacity(self):
        """Remove old trajectories to stay within capacity"""
        while self._size > self.capacity and len(self.trajectories) > 0:
            removed_trajectory = self.trajectories.pop(0)
            self._size -= len(removed_trajectory['observations'])
    
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
                if isinstance(values, np.ndarray) and values.ndim > 0:
                    all_data[key].extend(values.tolist())
                else:
                    all_data[key].append(values)
        
        # Sample random indices
        total_steps = len(all_data['observations'])
        indices = np.random.choice(total_steps, size=min(batch_size, total_steps), replace=False)
        
        # Create batch dictionary
        batch = {}
        for key, values in all_data.items():
            if len(values) > 0:
                sampled_values = [values[i] for i in indices]
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
    
    def _save_buffer_state(self) -> Dict[str, Any]:
        """Save trajectory buffer specific state"""
        return {
            'trajectories': self.trajectories,
            'current_trajectory': dict(self.current_trajectory),  # Convert defaultdict to dict
            'trajectory_complete': self._trajectory_complete,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda
        }
    
    def _load_buffer_state(self, state: Dict[str, Any]):
        """Load trajectory buffer specific state"""
        self.trajectories = state.get('trajectories', [])
        self.current_trajectory = defaultdict(list, state.get('current_trajectory', {}))
        self._trajectory_complete = state.get('trajectory_complete', True)
        
        # Recalculate size
        self._size = sum(len(traj.get('observations', [])) for traj in self.trajectories)
        self._size += len(self.current_trajectory.get('observations', []))