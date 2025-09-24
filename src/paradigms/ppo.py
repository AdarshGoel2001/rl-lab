"""
PPO Paradigm

PPO implementation using the paradigm architecture.
Inherits generic forward pass from ModelFreeParadigm and implements
PPO-specific loss computation and multi-epoch update logic.
"""

from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .model_free import ModelFreeParadigm
from ..components.encoders.base import BaseEncoder
from ..components.representation_learners.base import BaseRepresentationLearner
from ..components.policy_heads.base import BasePolicyHead
from ..components.value_functions.base import BaseValueFunction
from ..utils.registry import register_algorithm


@register_algorithm("ppo")  # Register as algorithm for trainer compatibility
class PPOParadigm(ModelFreeParadigm):
    """
    PPO paradigm implementation.

    Inherits generic model-free forward pass and implements PPO-specific:
    - Clipped surrogate objective
    - Value loss (with optional clipping)
    - Multi-epoch minibatch training
    - Advantage normalization

    Preserves exact interface as original PPO algorithm for trainer compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO paradigm from config.

        This constructor matches the original PPO algorithm interface:
        - config contains: networks, observation_space, action_space, device
        - All PPO hyperparameters
        """
        # Extract components from config (same as original PPO)
        networks = config['networks']
        observation_space = config['observation_space']
        action_space = config['action_space']
        device = torch.device(config['device'])

        # Create components from existing networks (preserving all initialization)
        encoder, policy_head, value_function = self._create_components_from_networks(
            networks, observation_space, action_space, device, config
        )

        # Create identity representation learner (no representation learning for PPO)
        from ..components.representation_learners.identity import IdentityRepresentationLearner
        representation_learner = IdentityRepresentationLearner({'device': device})

        # Initialize parent with components
        super().__init__(encoder, representation_learner, policy_head, value_function, config)

        # PPO-specific hyperparameters - from original PPO
        self.actor_lr = float(config.get('actor_lr', config.get('lr', 3e-4)))
        self.critic_lr = float(config.get('critic_lr', config.get('lr', 3e-4)))
        self.clip_ratio = float(config.get('clip_ratio', 0.2))
        self.value_coef = float(config.get('value_coef', 0.5))
        self.entropy_coef = float(config.get('entropy_coef', 0.01))
        self.max_grad_norm = float(config.get('max_grad_norm', 0.5))
        self.ppo_epochs = int(config.get('ppo_epochs', 4))
        self.minibatch_size = int(config.get('minibatch_size', 64))
        self.normalize_advantages = bool(config.get('normalize_advantages', True))
        self.clip_value_loss = bool(config.get('clip_value_loss', True))

        # Set up optimizers - separate for actor and critic like original
        actor_params = list(self.encoder.parameters()) + list(self.policy_head.parameters())
        critic_params = list(self.encoder.parameters()) + list(self.value_function.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr)

        # For trainer compatibility
        self.optimizers = {
            'actor': self.actor_optimizer,
            'critic': self.critic_optimizer
        }

        # Store action space type for compatibility
        self.action_space_type = 'continuous' if not action_space.discrete else 'discrete'

        # Networks dict for trainer compatibility (bootstrap value computation)
        self.networks = {
            'actor': self.policy_head,
            'critic': self.value_function,
            'encoder': self.encoder
        }

    def _create_components_from_networks(self, networks, observation_space, action_space, device, config):
        """
        Create paradigm components from existing networks.
        This preserves all the network initialization logic from trainer.
        """
        # Determine encoder from actor network
        if 'actor' in networks:
            actor_network = networks['actor']
        else:
            # Single network case
            actor_network = networks[list(networks.keys())[0]]

        # Create encoder based on network type
        encoder = self._create_encoder_from_network(actor_network, observation_space, device)
        feature_dim = getattr(encoder, 'output_dim', 64)

        # Create policy head
        policy_head = self._create_policy_head_from_network(actor_network, action_space, feature_dim, device, config)

        # Create value function
        if 'critic' in networks:
            critic_network = networks['critic']
        else:
            critic_network = actor_network

        value_function = self._create_value_function_from_network(critic_network, feature_dim, device)

        return encoder, policy_head, value_function

    def _create_encoder_from_network(self, network, observation_space, device):
        """Create encoder from existing network architecture."""
        # Map network types to encoder types
        network_type = network.__class__.__name__.lower()

        if 'mlp' in network_type:
            encoder_config = {
                'input_dim': observation_space.shape,
                'hidden_dims': getattr(network, 'config', {}).get('hidden_dims', [64, 64]),
                'activation': getattr(network, 'config', {}).get('activation', 'relu'),
                'device': device
            }
            from ..components.encoders.simple_mlp import MLPEncoder
            return MLPEncoder(encoder_config)

        elif 'cnn' in network_type:
            # Handle CNN encoders
            if 'minigrid' in network_type:
                encoder_config = {
                    'input_dim': observation_space.shape,
                    'device': device
                }
                from ..components.encoders.cnn import MiniGridCNNEncoder
                return MiniGridCNNEncoder(encoder_config)
            elif 'nature' in network_type:
                encoder_config = {
                    'input_dim': observation_space.shape,
                    'device': device
                }
                from ..components.encoders.cnn import NatureCNNEncoder
                return NatureCNNEncoder(encoder_config)
            else:
                # Default CNN
                encoder_config = {
                    'input_dim': observation_space.shape,
                    'device': device
                }
                from ..components.encoders.cnn import IMPALACNNEncoder
                return IMPALACNNEncoder(encoder_config)
        else:
            # Default to MLP
            encoder_config = {
                'input_dim': observation_space.shape,
                'hidden_dims': [64, 64],
                'activation': 'relu',
                'device': device
            }
            from ..components.encoders.simple_mlp import MLPEncoder
            return MLPEncoder(encoder_config)

    def _create_policy_head_from_network(self, network, action_space, feature_dim, device, config):
        """Create policy head from network."""
        from ..components.policy_heads.gaussian_mlp import ContinuousActorPolicyHead

        policy_config = {
            'representation_dim': feature_dim,
            'action_dim': action_space.n if action_space.discrete else int(np.prod(action_space.shape)),
            'discrete_actions': action_space.discrete,
            'device': device,
            # Copy continuous control parameters from config
            'log_std_init': config.get('log_std_init', 0.0),
            'action_bounds': config.get('action_bounds', None),
            'use_tanh_squashing': config.get('use_tanh_squashing', True),
            'hidden_dims': getattr(network, 'config', {}).get('hidden_dims', [64, 64]),
            'activation': getattr(network, 'config', {}).get('activation', 'tanh')
        }

        return ContinuousActorPolicyHead(policy_config)

    def _create_value_function_from_network(self, network, feature_dim, device):
        """Create value function from network."""
        from ..components.value_functions.mlp_critic import CriticMLPValueFunction

        value_config = {
            'representation_dim': feature_dim,
            'device': device,
            'hidden_dims': getattr(network, 'config', {}).get('hidden_dims', [64, 64]),
            'activation': getattr(network, 'config', {}).get('activation', 'relu')
        }

        return CriticMLPValueFunction(value_config)

    def compute_ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO loss components - exact logic from original PPO.

        Args:
            batch: Batch of trajectory data

        Returns:
            Tuple of (policy_loss, value_loss, entropy_loss)
        """
        # Extract data from batch - from original
        observations = batch['observations']
        actions = batch['actions']
        if self.action_space_type == 'discrete':
            actions = actions.long()
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        # Normalize advantages - from original (robust implementation)
        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean

        # Evaluate actions under current policy
        log_probs, values, entropy = self.evaluate_actions(observations, actions)

        # Compute probability ratio - from original
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective - from original
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss - from original
        if self.clip_value_loss and 'old_values' in batch:
            old_values = batch['old_values']
            values_clipped = old_values + torch.clamp(values - old_values, -self.clip_ratio, self.clip_ratio)
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)

        # Entropy loss - from original
        entropy_loss = -entropy.mean()

        return policy_loss, value_loss, entropy_loss

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute losses (for paradigm interface compatibility).
        """
        policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(batch)
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO update with multi-epoch minibatch training.

        Exact logic from original PPO implementation.

        Args:
            batch: Experience batch from buffer

        Returns:
            Dictionary of training metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Prepare for minibatching
        batch_size = batch['observations'].shape[0]
        indices = np.arange(batch_size)

        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'grad_norm': 0.0
        }

        # Multi-epoch training - from original
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                minibatch = {k: v[mb_indices] for k, v in batch.items()}

                # Compute losses
                policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(minibatch)
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Zero gradients
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Backward pass
                total_loss.backward()

                # Gradient clipping - from original
                if self.max_grad_norm > 0:
                    all_params = (
                        list(self.encoder.parameters()) +
                        list(self.policy_head.parameters()) +
                        list(self.value_function.parameters())
                    )
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                else:
                    grad_norm = 0.0

                # Update parameters
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Accumulate metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += total_loss.item()
                metrics['grad_norm'] += grad_norm if isinstance(grad_norm, float) else grad_norm.item()

        # Average metrics
        num_updates = self.ppo_epochs * max(1, batch_size // self.minibatch_size)
        for key in metrics:
            metrics[key] /= num_updates

        self.step += 1
        return metrics

    def train(self):
        """Set all components to training mode for trainer compatibility."""
        self.encoder.train()
        self.representation_learner.train()
        self.policy_head.train()
        self.value_function.train()

    def eval(self):
        """Set all components to evaluation mode for trainer compatibility."""
        self.encoder.eval()
        self.representation_learner.eval()
        self.policy_head.eval()
        self.value_function.eval()