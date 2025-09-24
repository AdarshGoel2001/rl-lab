"""
Gaussian MLP Policy Head

Policy head based on the original ContinuousActorMLP design.
Preserves all functionality including action bounds and initialization.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from .base import BasePolicyHead
from ...utils.registry import register_policy_head


@register_policy_head("continuous_actor")
class ContinuousActorPolicyHead(BasePolicyHead):
    """
    Continuous actor policy head based on original ContinuousActorMLP.

    Outputs both mean and log_std for continuous action distributions.
    Supports action bounds through tanh squashing and scaling.
    """

    def _build_head(self):
        """Build policy head using original ContinuousActorMLP logic."""
        if self.representation_dim is None:
            raise ValueError("ContinuousActorPolicyHead requires 'representation_dim' in config")
        if self.action_dim is None:
            raise ValueError("ContinuousActorPolicyHead requires 'action_dim' in config")

        # Original defaults
        hidden_dims = self.config.get('hidden_dims', [64, 64])
        activation = self.config.get('activation', 'tanh')
        self.log_std_init = self.config.get('log_std_init', 0.0)
        self.action_bounds = self.config.get('action_bounds', None)
        self.use_tanh_squashing = self.config.get('use_tanh_squashing', True)

        # Shared feature layers - from original
        layers = []
        prev_dim = self.representation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
            activation_fn = self._get_activation_function(activation)
            layers.append(activation_fn)

            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Combined output layer: outputs 2 * action_dim (mean + log_std pairs) - from original
        self.output_layer = nn.Linear(prev_dim, 2 * self.action_dim)

        # Store action bounds for scaling - from original
        if self.action_bounds is not None:
            bounds = self.action_bounds
            if len(bounds) != self.action_dim:
                raise ValueError(f"action_bounds must have {self.action_dim} pairs, got {len(bounds)}")

            lows = [pair[0] for pair in bounds]
            highs = [pair[1] for pair in bounds]
            self.register_buffer('action_low', torch.tensor(lows, dtype=torch.float32))
            self.register_buffer('action_high', torch.tensor(highs, dtype=torch.float32))
        else:
            # Default to [-1, 1] for each dimension - from original
            self.register_buffer('action_low', torch.full((self.action_dim,), -1.0, dtype=torch.float32))
            self.register_buffer('action_high', torch.full((self.action_dim,), 1.0, dtype=torch.float32))

        # Initialize weights using original method
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using original ContinuousActorMLP method."""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        # Initialize output layer with special care - from original
        output_dim = self.action_dim

        # Initialize mean part (first half of weights) with small weights for better tanh gradients
        nn.init.orthogonal_(self.output_layer.weight[:output_dim], gain=0.01)
        nn.init.zeros_(self.output_layer.bias[:output_dim])

        # Initialize log_std part (second half of weights) with small weights
        nn.init.orthogonal_(self.output_layer.weight[output_dim:], gain=0.01)
        nn.init.constant_(self.output_layer.bias[output_dim:], self.log_std_init)

    def _get_activation_function(self, activation_name: str):
        """Get activation function."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def forward(self,
                representation: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Normal:
        """
        Forward pass using original ContinuousActorMLP logic.

        Args:
            representation: State representation
            context: Optional context (ignored)

        Returns:
            Normal distribution over actions
        """
        # Shared features
        features = self.shared_layers(representation)

        # Network outputs 2*action_dim values - from original
        network_output = self.output_layer(features)

        action_dim = self.action_dim

        # Split network output in half: first half for means, second half for log_stds - from original
        raw_mean = network_output[:, :action_dim]        # First half
        raw_log_std = network_output[:, action_dim:]     # Second half

        # Apply tanh squashing to mean if requested - from original
        if self.use_tanh_squashing:
            mean = torch.tanh(raw_mean)
            # Scale to action bounds using registered buffers
            mean = self.action_low + 0.5 * (self.action_high - self.action_low) * (mean + 1)
        else:
            mean = raw_mean

        log_std = raw_log_std
        std = torch.exp(log_std)

        # Scale std proportionally if using tanh squashing
        if self.use_tanh_squashing:
            std = std * 0.5 * (self.action_high - self.action_low)

        return Normal(mean, std)

    def sample_actions(self,
                      representation: torch.Tensor,
                      context: Optional[Dict[str, Any]] = None,
                      deterministic: bool = False) -> torch.Tensor:
        """Sample actions from policy."""
        action_dist = self.forward(representation, context)

        if deterministic:
            return action_dist.mean
        else:
            return action_dist.sample()

    def action_log_prob(self,
                       actions: torch.Tensor,
                       representation: torch.Tensor,
                       context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute log probability of actions."""
        action_dist = self.forward(representation, context)
        log_prob = action_dist.log_prob(actions)

        # Sum over action dimensions
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)

        return log_prob


@register_policy_head("gaussian_mlp")
class GaussianMLPPolicyHead(BasePolicyHead):
    """
    Simple Gaussian MLP policy head for basic continuous control.

    A simpler alternative to the full ContinuousActor when you don't
    need action bounds or tanh squashing.
    """

    def _build_head(self):
        """Build simple Gaussian policy head."""
        if self.representation_dim is None:
            raise ValueError("GaussianMLPPolicyHead requires 'representation_dim' in config")
        if self.action_dim is None:
            raise ValueError("GaussianMLPPolicyHead requires 'action_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')

        # Build shared layers
        layers = []
        current_dim = self.representation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self._get_activation_function(activation))
            current_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(current_dim, self.action_dim)
        self.log_std_head = nn.Linear(current_dim, self.action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def _get_activation_function(self, activation_name: str):
        """Get activation function."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def forward(self,
                representation: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Normal:
        """Forward pass."""
        features = self.shared_layers(representation)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return Normal(mean, std)