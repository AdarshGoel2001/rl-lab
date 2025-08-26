"""
Random Agent Algorithm

A simple random agent for testing the framework infrastructure.
This agent selects random actions and doesn't actually learn,
but it demonstrates the complete algorithm interface.

Perfect for:
- Testing framework components
- Debugging environment integration
- Baseline comparisons
- System validation
"""

import torch
import numpy as np
from typing import Dict, Any

from src.algorithms.base import BaseAlgorithm
from src.utils.registry import register_algorithm


@register_algorithm("random")
class RandomAgent(BaseAlgorithm):
    """
    Random agent that selects actions uniformly at random.
    
    This is primarily useful for testing the framework and as a baseline.
    It implements the full algorithm interface but doesn't actually learn.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize random agent"""
        super().__init__(config)
        
        # Store action space information
        self.action_space = config.get('action_space')
        self.observation_space = config.get('observation_space')
        
        if self.action_space is None:
            raise ValueError("action_space must be provided in config")
    
    def _setup_networks_and_optimizers(self):
        """Random agent doesn't need any networks"""
        pass
    
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select a random action.
        
        Args:
            observation: Current observation (ignored for random agent)
            deterministic: Whether to act deterministically (ignored)
            
        Returns:
            Random action tensor
        """
        batch_size = observation.shape[0] if observation.dim() > 1 else 1
        
        if self.action_space.discrete:
            # Discrete action space - sample random integers
            actions = torch.randint(
                0, self.action_space.n, 
                size=(batch_size,), 
                device=self.device
            )
        else:
            # Continuous action space - sample from uniform distribution
            action_shape = (batch_size, self.action_space.shape[0])
            
            # Sample from action bounds if available
            if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                low = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
                high = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)
                actions = torch.rand(action_shape, device=self.device) * (high - low) + low
            else:
                # Default to [-1, 1] if no bounds specified
                actions = 2 * torch.rand(action_shape, device=self.device) - 1
        
        # Remove batch dimension if input was single observation
        if observation.dim() == 1:
            actions = actions.squeeze(0)
        
        return actions
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Random agent doesn't learn, so this is a no-op.
        
        Args:
            batch: Batch of experiences (ignored)
            
        Returns:
            Empty metrics dictionary
        """
        self.step += 1
        
        # Return some dummy metrics for logging
        return {
            'random_step': float(self.step),
            'batch_size': float(len(batch.get('observations', []))),
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics"""
        return {
            'algorithm/step': float(self.step),
            'algorithm/type': 0.0,  # 0 for random (just for logging)
        }
    
    def _save_algorithm_state(self) -> Dict[str, Any]:
        """Random agent has no state to save"""
        return {
            'action_space_info': {
                'discrete': self.action_space.discrete,
                'shape': self.action_space.shape,
                'n': getattr(self.action_space, 'n', None),
            }
        }
    
    def _load_algorithm_state(self, state: Dict[str, Any]):
        """Random agent has no state to load"""
        pass