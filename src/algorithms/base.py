"""
Base Algorithm Interface

This module defines the abstract base class that all RL algorithms must implement.
It provides a consistent interface for action selection, learning updates, and 
state management (checkpoints).

Key benefits for researchers:
- Consistent API across all algorithms  
- Automatic checkpoint save/load functionality
- Built-in metric tracking
- Easy algorithm swapping via config changes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn


class BaseAlgorithm(ABC):
    """
    Abstract base class for all RL algorithms.
    
    This class defines the interface that all algorithms must implement
    to work with the training infrastructure. It handles common functionality
    like checkpoint management while leaving algorithm-specific logic abstract.
    
    Attributes:
        config: Algorithm configuration dictionary
        networks: Dictionary mapping network names to nn.Module instances
        optimizers: Dictionary mapping optimizer names to torch.optim instances
        step: Current training step count
        device: PyTorch device (cuda/cpu)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize algorithm with configuration.
        
        Args:
            config: Dictionary containing algorithm hyperparameters and settings
        """
        self.config = config
        self.networks: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.step = 0
        self.device = torch.device(config.get('device', 'cpu'))
        self._setup_networks_and_optimizers()
    
    @abstractmethod
    def _setup_networks_and_optimizers(self):
        """
        Create networks and optimizers for this algorithm.
        Must populate self.networks and self.optimizers dictionaries.
        
        Example:
            self.networks['actor'] = ActorNetwork(config)
            self.networks['critic'] = CriticNetwork(config)
            self.optimizers['actor'] = torch.optim.Adam(self.networks['actor'].parameters())
        """
        pass
    
    @abstractmethod
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action given observation.
        
        This is called during environment interaction to select actions.
        For continuous control, return action tensor directly.
        For discrete actions, return action indices.
        
        Args:
            observation: Current observation from environment
            deterministic: If True, select best action. If False, sample from policy.
            
        Returns:
            Action tensor to execute in environment
        """
        pass
    
    @abstractmethod 
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update algorithm with batch of experiences.
        
        This is where the learning happens. The batch contains experiences
        from the replay buffer (off-policy) or trajectory buffer (on-policy).
        
        Args:
            batch: Dictionary containing experience batch with keys like:
                - 'observations': Tensor of shape (batch_size, obs_dim)
                - 'actions': Tensor of shape (batch_size, action_dim) 
                - 'rewards': Tensor of shape (batch_size,)
                - 'next_observations': Tensor of shape (batch_size, obs_dim)
                - 'dones': Tensor of shape (batch_size,) 
                
        Returns:
            Dictionary of metrics (losses, etc.) for logging
        """
        pass
    
    def save_checkpoint(self) -> Dict[str, Any]:
        """
        Save complete algorithm state for resuming.
        
        This captures everything needed to resume training exactly where
        it left off, including network weights, optimizer states, and 
        algorithm-specific state.
        
        Returns:
            Dictionary containing all state needed for resuming
        """
        checkpoint = {
            'networks': {k: v.state_dict() for k, v in self.networks.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'step': self.step,
            'config': self.config,
            'algorithm_state': self._save_algorithm_state()
        }
        return checkpoint
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load algorithm state from checkpoint.
        
        Restores networks, optimizers, step count, and algorithm-specific state
        to resume training exactly where it left off.
        
        Args:
            checkpoint: Dictionary returned by save_checkpoint()
        """
        # Load network states
        for name, state_dict in checkpoint['networks'].items():
            if name in self.networks:
                self.networks[name].load_state_dict(state_dict)
            else:
                print(f"Warning: Network '{name}' in checkpoint but not in current algorithm")
        
        # Load optimizer states  
        for name, state_dict in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)
            else:
                print(f"Warning: Optimizer '{name}' in checkpoint but not in current algorithm")
        
        # Load step count and algorithm state
        self.step = checkpoint.get('step', 0)
        self._load_algorithm_state(checkpoint.get('algorithm_state', {}))
    
    def _save_algorithm_state(self) -> Dict[str, Any]:
        """
        Save algorithm-specific state (override in subclasses if needed).
        
        For example, PPO might save current policy version, DQN might save
        epsilon decay state, etc.
        
        Returns:
            Dictionary of algorithm-specific state
        """
        return {}
    
    def _load_algorithm_state(self, state: Dict[str, Any]):
        """
        Load algorithm-specific state (override in subclasses if needed).
        
        Args:
            state: Dictionary returned by _save_algorithm_state()
        """
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current training metrics for logging.
        
        Override in subclasses to provide algorithm-specific metrics.
        Called after each update() to collect metrics for TensorBoard/WandB.
        
        Returns:
            Dictionary of metric name -> value pairs
        """
        return {
            'algorithm/step': float(self.step)
        }
    
    def to(self, device: Union[str, torch.device]):
        """Move all networks to specified device"""
        self.device = torch.device(device)
        for network in self.networks.values():
            network.to(self.device)
    
    def train(self):
        """Set all networks to training mode"""
        for network in self.networks.values():
            network.train()
    
    def eval(self):
        """Set all networks to evaluation mode"""
        for network in self.networks.values():
            network.eval()
    
    def parameters(self):
        """Get all trainable parameters from all networks"""
        params = []
        for network in self.networks.values():
            params.extend(list(network.parameters()))
        return params
    
    def named_parameters(self):
        """Get all named parameters from all networks"""
        named_params = []
        for net_name, network in self.networks.items():
            for param_name, param in network.named_parameters():
                named_params.append((f"{net_name}.{param_name}", param))
        return named_params