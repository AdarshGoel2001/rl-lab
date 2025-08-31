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
from src.utils.registry import register_algorithm, get_network


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
        config.setdefault('lr', 3e-4)  # Fallback if no separate rates specified
        config.setdefault('actor_lr', config.get('lr', 3e-4))
        config.setdefault('critic_lr', config.get('lr', 3e-4))
        config.setdefault('clip_ratio', 0.2) 
        config.setdefault('value_coef', 0.5)
        config.setdefault('entropy_coef', 0.01)
        config.setdefault('max_grad_norm', 0.5)
        config.setdefault('ppo_epochs', 4)
        config.setdefault('minibatch_size', 64)
        config.setdefault('normalize_advantages', True)
        config.setdefault('clip_value_loss', True)
        config.setdefault('log_std_min', -20)  # Minimum log std (exp(-20) ≈ 2e-9)
        config.setdefault('log_std_max', 2)    # Maximum log std (exp(2) ≈ 7.4)
        
        # Continuous control parameters
        config.setdefault('log_std_init', 0.0)  # Initial log std
        config.setdefault('action_bounds', None)  # [[low, high], ...] for each action dim
        config.setdefault('use_tanh_squashing', True)  # Use tanh to bound actions
        
        # Store PPO hyperparameters with type conversion
        self.actor_lr = float(config['actor_lr'])
        self.critic_lr = float(config['critic_lr'])
        self.clip_ratio = float(config['clip_ratio'])
        self.value_coef = float(config['value_coef']) 
        self.entropy_coef = float(config['entropy_coef'])
        self.max_grad_norm = float(config['max_grad_norm'])
        self.ppo_epochs = int(config['ppo_epochs'])
        self.minibatch_size = int(config['minibatch_size'])
        self.normalize_advantages = bool(config['normalize_advantages'])
        self.clip_value_loss = bool(config['clip_value_loss'])
        self.log_std_min = float(config['log_std_min'])  # Handles 2e-9, -20, etc.
        self.log_std_max = float(config['log_std_max'])
        
        # Add debugging for network outputs
        self.debug_network_outputs = config.get('debug_network_outputs', False)
        self._step_count = 0
        
        # Environment info (will be set by trainer)
        if 'observation_space' in config and 'action_space' in config:
            obs_space = config['observation_space']
            action_space = config['action_space']
            self.obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
            self.action_dim = action_space.n if (hasattr(action_space, 'n') and action_space.n is not None) else action_space.shape[0]
            self.action_space_type = 'discrete' if (hasattr(action_space, 'n') and action_space.n is not None) else 'continuous'
            # Action space detected successfully
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
            # Create actor network using registry system
            actor_config = {
                'input_dim': self.obs_dim,
                'output_dim': self.action_dim,
                'hidden_dims': self.config['network']['actor']['hidden_dims'],  # From YAML
                'activation': self.config['network']['actor']['activation'],     # From YAML
                # Pass continuous control parameters to network
                'log_std_init': self.config.get('log_std_init', 0.0),
                'action_bounds': self.config.get('action_bounds'),
                'use_tanh_squashing': self.config.get('use_tanh_squashing', True)
            }
            actor_type = self.config['network']['actor']['type']
            ActorClass = get_network(actor_type)
            self.networks['actor'] = ActorClass(actor_config)
            
            # Create critic network using registry system  
            critic_config = {
                'input_dim': self.obs_dim,
                'output_dim': 1,
                'hidden_dims': self.config['network']['critic']['hidden_dims'],  # From YAML
                'activation': self.config['network']['critic']['activation']      # From YAML
            }
            critic_type = self.config['network']['critic']['type']
            CriticClass = get_network(critic_type)
            self.networks['critic'] = CriticClass(critic_config)
        
        # Create Adam optimizers for both networks with separate learning rates
        self.optimizers['actor'] = torch.optim.Adam(self.networks['actor'].parameters(), lr=self.actor_lr)
        self.optimizers['critic'] = torch.optim.Adam(self.networks['critic'].parameters(), lr=self.critic_lr)
    
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action given observation using current policy.
        
        Args:
            observation: Current observation tensor
            deterministic: If True, select best action. If False, sample from distribution.
            
        Returns:
            Selected action tensor
        """
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                # Discrete actions - use standard actor output as logits
                logits = self.networks['actor'](observation)
                dist = Categorical(logits=logits)
            else:
                # Continuous actions - actor outputs mean and log_std
                actor_output = self.networks['actor'](observation)
                action_dim = self.action_dim
                
                mean = actor_output[:, :action_dim]
                log_std = actor_output[:, action_dim:]
                std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
                
                dist = Normal(mean, std)
            
            # Select action (deterministic vs stochastic)
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
        """
        # Get actor output and value
        actor_output = self.networks['actor'](observation)
        value = self.networks['critic'](observation)
        
        # Create distribution and sample action
        if self.action_space_type == 'discrete':
            dist = Categorical(logits=actor_output)
        else:
            # Continuous actions - actor outputs mean and log_std
            action_dim = self.action_dim
            mean = actor_output[:, :action_dim]
            log_std = actor_output[:, action_dim:]
            std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
            dist = Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Sum log_prob across action dimensions for continuous control
        if self.action_space_type == 'continuous':
            log_prob = log_prob.sum(axis=-1)
        
        # Debug logging during action collection
        if self.debug_network_outputs and self._step_count % 1000 == 0:
            self._step_count += 1
            _ = self._debug_network_outputs(observation, step_name="collect")
            # Debug metrics are computed but not used here (used in update method)
        else:
            self._step_count += 1
        
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
        """
        # Forward pass through networks
        actor_output = self.networks['actor'](observations)
        values = self.networks['critic'](observations)
        
        # Create distribution
        if self.action_space_type == 'discrete':
            dist = Categorical(logits=actor_output)
        else:
            # Continuous actions - actor outputs mean and log_std
            action_dim = self.action_dim
            mean = actor_output[:, :action_dim]
            log_std = actor_output[:, action_dim:]
            std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
            dist = Normal(mean, std)
        
        # Evaluate actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Sum log_prob and entropy across action dimensions for continuous control
        if self.action_space_type == 'continuous':
            log_probs = log_probs.sum(axis=-1)
            entropy = entropy.sum(axis=-1)
        
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
        for _ in range(self.ppo_epochs):
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
        
        # Add debug metrics if enabled
        if self.debug_network_outputs and batch_size > 0:
            debug_metrics = self._debug_network_outputs(
                batch['observations'][:min(64, batch_size)], # Sample for efficiency
                step_name="update"
            )
            
            # Add critical debugging for value learning issues
            returns = batch['returns'][:min(64, batch_size)]
            advantages = batch['advantages'][:min(64, batch_size)]
            old_values = batch.get('old_values', torch.zeros_like(returns))[:min(64, batch_size)]
            
            debug_metrics.update({
                'debug/returns_mean': returns.mean().item(),
                'debug/returns_std': returns.std().item(),
                'debug/returns_min': returns.min().item(),
                'debug/returns_max': returns.max().item(),
                'debug/advantages_mean': advantages.mean().item(),
                'debug/advantages_std': advantages.std().item(),
                'debug/advantages_min': advantages.min().item(),
                'debug/advantages_max': advantages.max().item(),
                'debug/old_values_mean': old_values.mean().item(),
                'debug/old_values_std': old_values.std().item(),
                'debug/value_error_mean': (returns - old_values).abs().mean().item(),
            })
            
            metrics.update(debug_metrics)
         
        return metrics
    
    def _debug_network_outputs(self, observations: torch.Tensor, step_name: str = "") -> Dict[str, float]:
        """
        Debug method to log network mean/std outputs and detect issues.
        
        Args:
            observations: Batch of observations to evaluate
            step_name: Prefix for logging metrics
            
        Returns:
            Dictionary of debugging metrics
        """
        if not self.debug_network_outputs:
            return {}
            
        with torch.no_grad():
            if self.action_space_type == 'continuous':
                actor_output = self.networks['actor'](observations)
                action_dim = self.action_dim
                
                # Split mean and log_std (PPO format)
                mean = actor_output[:, :action_dim]
                log_std = actor_output[:, action_dim:]
                std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
                
                # Compute action bounds checking
                action_clipped = (mean.abs() > 1.9).float().mean().item()  # Close to [-2,2] bounds
                
                metrics = {
                    f'debug/{step_name}_mean_avg': mean.mean().item(),
                    f'debug/{step_name}_mean_std': mean.std().item(),
                    f'debug/{step_name}_mean_min': mean.min().item(),
                    f'debug/{step_name}_mean_max': mean.max().item(),
                    f'debug/{step_name}_log_std_avg': log_std.mean().item(),
                    f'debug/{step_name}_std_avg': std.mean().item(),
                    f'debug/{step_name}_std_min': std.min().item(),
                    f'debug/{step_name}_std_max': std.max().item(),
                    f'debug/{step_name}_action_clipped_pct': action_clipped,
                    f'debug/{step_name}_network_output_shape': float(actor_output.shape[1]),
                }
                
                return metrics
            else:
                # For discrete actions, just log basic stats
                actor_output = self.networks['actor'](observations)
                return {
                    f'debug/{step_name}_logits_avg': actor_output.mean().item(),
                    f'debug/{step_name}_logits_std': actor_output.std().item(),
                }
        
    
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