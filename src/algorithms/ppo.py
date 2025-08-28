"""
Proximal Policy Optimization (PPO) Algorithm Implementation

This module implements PPO, a popular on-policy reinforcement learning algorithm
that uses clipped surrogate objectives for stable policy updates.

Key features to implement:
- Actor-critic architecture with shared or separate networks
- Clipped surrogate objective for policy updates  
- Value function loss with optional clipping
- Entropy bonus for exploration
- Multi-epoch training on collected trajectories
- Generalized Advantage Estimation (GAE)

Reference Paper: "Proximal Policy Optimization Algorithms" by Schulman et al. (2017)

HOMEWORK INSTRUCTIONS:
Fill in the TODO sections marked below. Each TODO has hints about what needs
to be implemented. Focus on:
1. Network architecture setup
2. Action sampling with log probabilities  
3. PPO loss computation (clipped + value + entropy)
4. Training loop with multiple epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.algorithms.base import BaseAlgorithm
from src.networks.mlp import ActorMLP, CriticMLP
from src.utils.registry import register_algorithm


@register_algorithm("ppo")
class PPOAlgorithm(BaseAlgorithm):
    """
    Proximal Policy Optimization (PPO) implementation.
    
    PPO is an on-policy algorithm that uses a clipped surrogate objective
    to prevent destructively large policy updates while maintaining 
    sample efficiency.
    
    Attributes:
        actor: Policy network that outputs action probabilities
        critic: Value network that estimates state values
        actor_optimizer: Optimizer for policy network
        critic_optimizer: Optimizer for value network
        clip_ratio: PPO clipping parameter (typically 0.2)
        value_coef: Weight for value function loss
        entropy_coef: Weight for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO algorithm.
        
        Args:
            config: Configuration dictionary containing hyperparameters
        """
        # Set PPO-specific defaults
        config.setdefault('lr', 3e-4)
        config.setdefault('clip_ratio', 0.2) 
        config.setdefault('value_coef', 0.5)
        config.setdefault('entropy_coef', 0.01)
        config.setdefault('max_grad_norm', 0.5)
        config.setdefault('ppo_epochs', 4)
        config.setdefault('minibatch_size', 64)
        config.setdefault('normalize_advantages', True)
        config.setdefault('clip_value_loss', True)
        
        # Store PPO hyperparameters with type conversion
        self.lr = float(config['lr'])
        self.clip_ratio = float(config['clip_ratio'])
        self.value_coef = float(config['value_coef']) 
        self.entropy_coef = float(config['entropy_coef'])
        self.max_grad_norm = float(config['max_grad_norm'])
        self.ppo_epochs = int(config['ppo_epochs'])
        self.minibatch_size = int(config['minibatch_size'])
        self.normalize_advantages = bool(config['normalize_advantages'])
        self.clip_value_loss = bool(config['clip_value_loss'])
        
        # Environment info (will be set by trainer)
        if 'observation_space' in config and 'action_space' in config:
            obs_space = config['observation_space']
            action_space = config['action_space']
            self.obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
            self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
            self.action_space_type = 'discrete' if hasattr(action_space, 'n') else 'continuous'
        else:
            self.obs_dim = config.get('obs_dim')
            self.action_dim = config.get('action_dim') 
            self.action_space_type = config.get('action_space_type', 'discrete')
        
        super().__init__(config)
    
    def _setup_networks_and_optimizers(self):
        """
        Create actor and critic networks with their optimizers.
        
        TODO: Implement this method
        Hints:
        - Create actor network using ActorMLP from networks.mlp
        - Create critic network using CriticMLP from networks.mlp  
        - Set input_dim to self.obs_dim
        - Set actor output_dim to self.action_dim
        - Set critic output_dim to 1 (value function outputs single value)
        - Create separate Adam optimizers for actor and critic
        - Store networks in self.networks dict with keys 'actor' and 'critic'
        - Store optimizers in self.optimizers dict with keys 'actor' and 'critic'
        """
        # Check if networks are already provided by trainer
        if 'networks' in self.config:
            # Networks are already created by trainer
            self.networks = self.config['networks']
        else:
            # Create actor network - 4 layer MLP with 64 neurons each
            actor_config = {
                'input_dim': self.obs_dim,
                'output_dim': self.action_dim,
                'hidden_dims': self.config['network']['actor']['hidden_dims'],  # From YAML
                'activation': self.config['network']['actor']['activation']      # From YAML
            }
            self.networks['actor'] = ActorMLP(actor_config)
            
            # Create critic network - 3 layer MLP with 32 neurons each  
            critic_config = {
                'input_dim': self.obs_dim,
                'output_dim': 1,
                'hidden_dims': self.config['network']['critic']['hidden_dims'],  # From YAML
                'activation': self.config['network']['critic']['activation']      # From YAML
            }
            self.networks['critic'] = CriticMLP(critic_config)
        
        # Create Adam optimizers for both networks
        self.optimizers['actor'] = torch.optim.Adam(self.networks['actor'].parameters(), lr=self.lr)
        self.optimizers['critic'] = torch.optim.Adam(self.networks['critic'].parameters(), lr=self.lr)
    
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action given observation using current policy.
        
        Args:
            observation: Current observation tensor
            deterministic: If True, select best action. If False, sample from distribution.
            
        Returns:
            Selected action tensor
            
        TODO: Implement this method
        Hints:
        - Use self.networks['actor'] to get action logits/parameters
        - For discrete actions: use Categorical distribution
        - For continuous actions: use Normal distribution (assume fixed std for simplicity)
        - If deterministic=True, return mode/argmax
        - If deterministic=False, sample from distribution
        - Don't forget to handle both training and evaluation modes
        """
        # TODO: Get action logits/parameters from actor network
        logits = self.networks['actor'](observation)
        
        # TODO: Create appropriate distribution (Categorical or Normal)
        if self.action_space_type == 'discrete':
            dist = Categorical(logits=logits)
        else:
            # For continuous actions, assume fixed std=1.0 for now
            dist = Normal(logits, torch.ones_like(logits))
        
        # TODO: Select action (deterministic vs stochastic)
        if deterministic:
            action = dist.mode if hasattr(dist, 'mode') else dist.mean
        else:
            action = dist.sample()
        
        return action
        
    
    def get_action_and_value(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value estimate for given observation.
        
        This is used during trajectory collection to get all needed information
        in a single forward pass.
        
        Args:
            observation: Current observation tensor
            
        Returns:
            Tuple of (action, log_prob, value)
            
        TODO: Implement this method
        Hints:
        - Similar to act() but also return log_prob and value
        - Use both actor and critic networks
        - Make sure to return log probability of the sampled action
        """
        # TODO: Get action logits and value
        action_logits = self.networks['actor'](observation)
        value = self.networks['critic'](observation)
        
        # TODO: Create distribution and sample action
        if self.action_space_type == 'discrete':
            dist = Categorical(logits=action_logits)
        else:
            dist = Normal(action_logits, torch.ones_like(action_logits))
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
        
    
    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Given observations and actions, compute log probabilities, values,
        and entropy under the current policy. Used during policy updates.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions to evaluate
            
        Returns:
            Tuple of (log_probs, values, entropy)
            
        TODO: Implement this method  
        Hints:
        - Get action logits from actor and values from critic
        - Create distribution from action logits
        - Compute log_prob of given actions under current distribution
        - Compute entropy of current distribution
        - Return log_probs, values, and entropy
        """
        # TODO: Forward pass through networks
        action_logits = self.networks['actor'](observations)
        values = self.networks['critic'](observations)
        
        # TODO: Create distribution
        if self.action_space_type == 'discrete':
            dist = Categorical(logits=action_logits)
        else:
            dist = Normal(action_logits, torch.ones_like(action_logits))
        
        # TODO: Evaluate actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
        
    
    def compute_ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO loss components.
        
        Args:
            batch: Batch of trajectory data containing:
                - observations: State observations
                - actions: Actions taken
                - old_log_probs: Log probabilities under old policy
                - advantages: GAE advantages
                - returns: Discounted returns
                - old_values: Value estimates under old policy (if value clipping enabled)
                
        Returns:
            Tuple of (policy_loss, value_loss, entropy_loss)
            
        TODO: Implement this method - This is the core of PPO!
        Hints:
        - Use evaluate_actions to get current log_probs, values, entropy
        - Compute probability ratio: exp(new_log_prob - old_log_prob)
        - Implement clipped surrogate objective: min(ratio * advantage, clipped_ratio * advantage)
        - Compute value loss: MSE between predicted values and returns
        - Optionally clip value loss if self.clip_value_loss is True
        - Entropy loss encourages exploration: -entropy.mean()
        - Don't forget to normalize advantages if self.normalize_advantages is True
        """
        # TODO: Extract data from batch
        observations = batch['observations'] 
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # TODO: Normalize advantages if requested
        if self.normalize_advantages:
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # TODO: Evaluate actions under current policy
        log_probs, values, entropy = self.evaluate_actions(observations, actions)
        
        # TODO: Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # TODO: Compute clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # TODO: Compute value loss
        if self.clip_value_loss and 'old_values' in batch:
        #     # Clipped value loss
            old_values = batch['old_values']
            values_clipped = old_values + torch.clamp(values - old_values, -self.clip_ratio, self.clip_ratio)
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)
        
        # TODO: Compute entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()
        
        return policy_loss, value_loss, entropy_loss
        
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update PPO policy using collected trajectory batch.
        
        Args:
            batch: Dictionary containing trajectory data from buffer
            
        Returns:
            Dictionary of training metrics
            
        TODO: Implement this method
        Hints:
        - Run multiple PPO epochs (self.ppo_epochs) on the same data
        - For each epoch, split data into minibatches of size self.minibatch_size
        - For each minibatch:
          1. Compute PPO losses using compute_ppo_loss
          2. Combine losses: total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss  
          3. Backpropagate and update both actor and critic
          4. Apply gradient clipping if max_grad_norm > 0
        - Track and return metrics like losses, gradient norms, etc.
        - Increment self.step
        """
        # TODO: Convert batch to appropriate device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # TODO: Get batch size and prepare for minibatching
        batch_size = batch['observations'].shape[0]
        indices = np.arange(batch_size)
        
        metrics = {
             'policy_loss': 0.0,
             'value_loss': 0.0, 
             'entropy_loss': 0.0,
             'total_loss': 0.0,
             'grad_norm': 0.0
         }
        
        # TODO: Multi-epoch training
        for epoch in range(self.ppo_epochs):
        #     # Shuffle data for each epoch
             np.random.shuffle(indices)
             
        #     # Process minibatches
             for start in range(0, batch_size, self.minibatch_size):
                 end = start + self.minibatch_size
                 mb_indices = indices[start:end]
        #         
        #         # Create minibatch
                 minibatch = {k: v[mb_indices] for k, v in batch.items()}
        #         
        #         # Compute losses
                 policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(minibatch)
                 total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        #         
        #         # Zero gradients
                 for optimizer in self.optimizers.values():
                     optimizer.zero_grad()
        #         
        #         # Backward pass
                 total_loss.backward()
        #         
        #         # Gradient clipping
                 if self.max_grad_norm > 0:
                     grad_norm = torch.nn.utils.clip_grad_norm_(
                         list(self.networks['actor'].parameters()) + list(self.networks['critic'].parameters()),
                         self.max_grad_norm
                     )
                 else:
                     grad_norm = 0.0
                 
        #         # Update parameters
                 for optimizer in self.optimizers.values():
                     optimizer.step()
        #         
        #         # Accumulate metrics
                 metrics['policy_loss'] += policy_loss.item()
                 metrics['value_loss'] += value_loss.item()
                 metrics['entropy_loss'] += entropy_loss.item() 
                 metrics['total_loss'] += total_loss.item()
                 metrics['grad_norm'] += grad_norm if isinstance(grad_norm, float) else grad_norm.item()
        
        # TODO: Average metrics and increment step
        num_updates = self.ppo_epochs * (batch_size // self.minibatch_size)
        for key in metrics:
             metrics[key] /= num_updates
         
        self.step += 1
         
        return metrics
        
    
    def get_metrics(self) -> Dict[str, float]:
        """Get additional PPO-specific metrics for logging"""
        base_metrics = super().get_metrics()
        
        # Add PPO-specific metrics
        ppo_metrics = {
            'ppo/clip_ratio': self.clip_ratio,
            'ppo/value_coef': self.value_coef,
            'ppo/entropy_coef': self.entropy_coef,
        }
        
        return {**base_metrics, **ppo_metrics}
    
    def _save_algorithm_state(self) -> Dict[str, Any]:
        """Save PPO-specific state"""
        return {
            'clip_ratio': self.clip_ratio,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
        }
    
    def _load_algorithm_state(self, state: Dict[str, Any]):
        """Load PPO-specific state"""
        self.clip_ratio = state.get('clip_ratio', self.clip_ratio)
        self.value_coef = state.get('value_coef', self.value_coef) 
        self.entropy_coef = state.get('entropy_coef', self.entropy_coef)