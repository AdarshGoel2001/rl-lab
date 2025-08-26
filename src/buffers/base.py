"""
Base Buffer Interface

This module provides the abstract base class for experience replay buffers
and trajectory buffers used in RL algorithms. It supports both on-policy
(trajectory) and off-policy (replay) buffer types.

Key benefits:
- Consistent interface for different buffer types
- Automatic batching and sampling
- Built-in capacity management  
- Easy buffer swapping via configuration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from collections import deque, namedtuple


# Named tuple for storing individual experiences
Experience = namedtuple('Experience', [
    'observation', 'action', 'reward', 'next_observation', 'done'
])


class BaseBuffer(ABC):
    """
    Abstract base class for all experience buffers.
    
    This provides a consistent interface for storing and sampling experiences
    for RL algorithms. Supports both on-policy buffers (store full trajectories)
    and off-policy buffers (store individual transitions).
    
    Attributes:
        config: Buffer configuration dictionary
        capacity: Maximum number of experiences to store
        device: PyTorch device for tensor operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize buffer with configuration.
        
        Args:
            config: Dictionary containing buffer configuration including:
                - capacity: Maximum buffer size
                - batch_size: Batch size for sampling  
                - device: PyTorch device
                - Buffer-specific parameters
        """
        self.config = config
        self.capacity = config.get('capacity', 100000)
        self.batch_size = config.get('batch_size', 64)
        self.device = torch.device(config.get('device', 'cpu'))
        
        self._size = 0
        self._position = 0
        
        # Initialize buffer storage
        self._setup_storage()
    
    @abstractmethod
    def _setup_storage(self):
        """
        Initialize buffer storage structures.
        
        This should create the data structures needed to store experiences.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def add(self, **kwargs):
        """
        Add experience(s) to buffer.
        
        Args:
            **kwargs: Experience data (observations, actions, rewards, etc.)
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample (uses config default if None)
            
        Returns:
            Dictionary containing batched experience tensors
        """
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all experiences from buffer"""
        pass
    
    def __len__(self) -> int:
        """Get current number of experiences in buffer"""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return self._size >= self.capacity
    
    def ready(self) -> bool:
        """
        Check if buffer has enough experiences for sampling.
        
        By default, ready when we have at least one batch worth of data.
        Can be overridden by subclasses for different criteria.
        
        Returns:
            True if buffer is ready for sampling
        """
        return self._size >= self.batch_size
    
    def to_tensor(self, data: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor on correct device.
        
        Args:
            data: Numpy array to convert
            dtype: Target tensor dtype (auto-detect if None)
            
        Returns:
            PyTorch tensor on correct device
        """
        if dtype is None:
            # Auto-detect appropriate dtype
            if data.dtype == np.float64:
                dtype = torch.float32
            elif data.dtype == np.int64:
                dtype = torch.long
            else:
                dtype = torch.tensor(data).dtype
        
        return torch.tensor(data, dtype=dtype, device=self.device)
    
    def save_checkpoint(self) -> Dict[str, Any]:
        """
        Save buffer state for resuming training.
        
        Returns:
            Dictionary containing buffer state
        """
        return {
            'size': self._size,
            'position': self._position,
            'config': self.config,
            'buffer_state': self._save_buffer_state()
        }
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load buffer state from checkpoint.
        
        Args:
            checkpoint: Dictionary returned by save_checkpoint()
        """
        self._size = checkpoint.get('size', 0)
        self._position = checkpoint.get('position', 0)
        self._load_buffer_state(checkpoint.get('buffer_state', {}))
    
    @abstractmethod
    def _save_buffer_state(self) -> Dict[str, Any]:
        """
        Save buffer-specific state.
        
        Returns:
            Dictionary of buffer-specific state
        """
        pass
    
    @abstractmethod
    def _load_buffer_state(self, state: Dict[str, Any]):
        """
        Load buffer-specific state.
        
        Args:
            state: Dictionary returned by _save_buffer_state()
        """
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get buffer metrics for logging.
        
        Returns:
            Dictionary of buffer metrics
        """
        return {
            'buffer/size': float(self._size),
            'buffer/capacity': float(self.capacity),
            'buffer/utilization': float(self._size) / float(self.capacity),
            'buffer/position': float(self._position),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed buffer statistics (override in subclasses for more details).
        
        Returns:
            Dictionary of buffer statistics
        """
        return {
            'size': self._size,
            'capacity': self.capacity,
            'utilization': self._size / self.capacity if self.capacity > 0 else 0.0,
            'ready': self.ready(),
            'empty': self.is_empty(),
            'full': self.is_full(),
        }


class TrajectoryBuffer(BaseBuffer):
    """
    Buffer for storing complete trajectories (on-policy algorithms).
    
    This buffer stores complete episodes/trajectories and provides methods
    for computing returns and advantages. Used by algorithms like PPO, A2C.
    """
    
    def _setup_storage(self):
        """Initialize trajectory storage"""
        self.trajectories: List[List[Experience]] = []
        self.current_trajectory: List[Experience] = []
    
    def add_step(self, observation: np.ndarray, action: np.ndarray, 
                 reward: float, next_observation: np.ndarray, done: bool):
        """
        Add a single step to current trajectory.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Episode termination flag
        """
        experience = Experience(
            observation=observation,
            action=action, 
            reward=reward,
            next_observation=next_observation,
            done=done
        )
        
        self.current_trajectory.append(experience)
        
        # If episode is done, add trajectory to buffer
        if done:
            self.add_trajectory(self.current_trajectory)
            self.current_trajectory = []
    
    def add_trajectory(self, trajectory: List[Experience]):
        """
        Add complete trajectory to buffer.
        
        Args:
            trajectory: List of Experience tuples forming a complete trajectory
        """
        if len(trajectory) == 0:
            return
        
        self.trajectories.append(trajectory)
        self._size += len(trajectory)
        
        # Remove oldest trajectories if over capacity
        while self._size > self.capacity and len(self.trajectories) > 0:
            removed = self.trajectories.pop(0)
            self._size -= len(removed)
    
    def add(self, **kwargs):
        """Add step or trajectory based on provided arguments"""
        if 'trajectory' in kwargs:
            self.add_trajectory(kwargs['trajectory'])
        else:
            self.add_step(
                kwargs['observation'],
                kwargs['action'],
                kwargs['reward'], 
                kwargs['next_observation'],
                kwargs['done']
            )
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Sample batch of experiences from all trajectories.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of batched tensors
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} experiences but {batch_size} requested")
        
        # Flatten all trajectories
        all_experiences = []
        for trajectory in self.trajectories:
            all_experiences.extend(trajectory)
        
        # Sample experiences
        indices = np.random.choice(len(all_experiences), size=batch_size, replace=False)
        sampled_experiences = [all_experiences[i] for i in indices]
        
        # Convert to tensors
        batch = {
            'observations': self.to_tensor(np.array([exp.observation for exp in sampled_experiences])),
            'actions': self.to_tensor(np.array([exp.action for exp in sampled_experiences])),
            'rewards': self.to_tensor(np.array([exp.reward for exp in sampled_experiences])),
            'next_observations': self.to_tensor(np.array([exp.next_observation for exp in sampled_experiences])),
            'dones': self.to_tensor(np.array([exp.done for exp in sampled_experiences]), dtype=torch.bool),
        }
        
        return batch
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories = []
        self.current_trajectory = []
        self._size = 0
        self._position = 0
    
    def _save_buffer_state(self) -> Dict[str, Any]:
        """Save trajectory buffer state"""
        return {
            'trajectories': self.trajectories,
            'current_trajectory': self.current_trajectory
        }
    
    def _load_buffer_state(self, state: Dict[str, Any]):
        """Load trajectory buffer state"""
        self.trajectories = state.get('trajectories', [])
        self.current_trajectory = state.get('current_trajectory', [])