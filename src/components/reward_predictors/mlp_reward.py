"""
MLP Reward Predictor

A reward predictor using multi-layer perceptrons.
Predicts reward distribution as a Gaussian.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .base import BaseRewardPredictor
from ...utils.registry import register_reward_predictor


@register_reward_predictor("mlp")
class MLPRewardPredictor(BaseRewardPredictor):
    """
    MLP-based reward predictor.

    Learns to predict rewards as a Normal distribution with learned mean
    and either fixed or learned standard deviation.
    """

    def _build_predictor(self):
        """Build the MLP reward predictor."""
        if self.state_dim is None:
            raise ValueError("MLPRewardPredictor requires 'state_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)
        self.learn_std = self.config.get('learn_std', False)
        self.fixed_std = self.config.get('fixed_std', 1.0)
        self.min_std = self.config.get('min_std', 0.1)

        # Activation function mapping
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation_fn = activation_map[activation]

        # Build MLP for mean prediction
        layers = []
        current_dim = self.state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_fn)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Output layer for mean
        layers.append(nn.Linear(current_dim, 1))

        self.mean_network = nn.Sequential(*layers)

        # Optional learnable std
        if self.learn_std:
            # Build separate network for std
            std_layers = []
            current_dim = self.state_dim

            for hidden_dim in hidden_dims:
                std_layers.append(nn.Linear(current_dim, hidden_dim))
                std_layers.append(self.activation_fn)
                current_dim = hidden_dim

            std_layers.append(nn.Linear(current_dim, 1))
            self.std_network = nn.Sequential(*std_layers)
        else:
            self.std_network = None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Normal:
        """
        Predict reward distribution given state representation.

        Args:
            state: State representation tensor, shape (batch_size, state_dim)

        Returns:
            Normal distribution over rewards
        """
        # Predict mean
        reward_mean = self.mean_network(state).squeeze(-1)

        # Predict or use fixed std
        if self.learn_std:
            reward_log_std = self.std_network(state).squeeze(-1)
            # Apply softplus to ensure positivity and add minimum
            reward_std = F.softplus(reward_log_std) + self.min_std
        else:
            reward_std = torch.full_like(reward_mean, self.fixed_std)

        return Normal(reward_mean, reward_std)

    def reward_loss(self,
                    states: torch.Tensor,
                    rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reward prediction learning loss.

        Args:
            states: State representations, shape (batch_size, state_dim)
            rewards: True rewards, shape (batch_size,) or (batch_size, 1)

        Returns:
            Dictionary of loss components
        """
        predicted_dist = self.forward(states)

        # Ensure rewards are the right shape
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)

        # Negative log likelihood loss
        nll_loss = -predicted_dist.log_prob(rewards).mean()

        # Also compute MAE for monitoring
        mae_loss = (predicted_dist.mean - rewards).abs().mean()

        return {
            'reward_loss': nll_loss,
            'reward_nll': nll_loss,
            'reward_mae': mae_loss
        }

    def get_reward_info(self) -> Dict[str, Any]:
        """Get information about this reward predictor."""
        info = super().get_reward_info()
        info.update({
            'learn_std': self.learn_std,
            'fixed_std': self.fixed_std if not self.learn_std else None,
            'min_std': self.min_std,
            'hidden_dims': self.config.get('hidden_dims', [256, 256])
        })
        return info
