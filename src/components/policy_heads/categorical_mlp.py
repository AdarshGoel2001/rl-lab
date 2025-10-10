"""
Categorical MLP Policy Head

Policy head for discrete action spaces using categorical distributions.
Outputs raw logits for discrete action selection.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import BasePolicyHead
from ...utils.registry import register_policy_head


@register_policy_head("categorical_mlp")
class CategoricalMLPPolicyHead(BasePolicyHead):
    """
    MLP-based policy head for discrete action spaces.

    Outputs raw logits that are used to create a Categorical distribution.
    Suitable for environments with discrete action spaces (e.g., CartPole, Atari).
    """

    def _build_head(self):
        """Build the categorical policy head architecture."""
        hidden_dims = self.config.get('hidden_dims', [64])
        activation = self.config.get('activation', 'tanh')

        self.discrete_actions = True

        layers = []
        current_dim = self.representation_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())  # Default
            current_dim = hidden_dim

        # Output layer - raw logits for each action
        layers.append(nn.Linear(current_dim, self.action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, representation: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Categorical:
        """
        Forward pass through policy head.

        Args:
            representation: Representation tensor from encoder, shape (batch_size, representation_dim)
            context: Optional context (unused for basic categorical policy)

        Returns:
            Categorical distribution over actions
        """
        logits = self.network(representation)
        return Categorical(logits=logits)

    @property
    def supports_context(self) -> bool:
        """This policy head does not use context."""
        return False
