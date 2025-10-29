"""
Base Policy Head Interface

This module defines the abstract base class for policy heads.
Policy heads convert representations to action distributions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.distributions import Distribution


class BasePolicyHead(nn.Module, ABC):
    """
    Abstract base class for all policy heads.

    Policy heads take representations and convert them to action distributions.
    They handle the final stage of action generation in the modular architecture.
    """

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize policy head with configuration.

        Args:
            config: Dictionary containing hyperparameters and settings
        """
        super().__init__()
        merged_config: Dict[str, Any] = {}
        if config is not None:
            merged_config.update(config)
        if kwargs:
            merged_config.update(kwargs)
        self.config = merged_config
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.representation_dim = self.config.get('representation_dim')
        self.action_dim = self.config.get('action_dim')
        self.discrete_actions = self.config.get('discrete_actions', False)
        self._build_head()

    @abstractmethod
    def _build_head(self):
        """
        Build the policy head architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def forward(self,
                representation: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Distribution:
        """
        Convert representation to action distribution.

        Args:
            representation: Representation tensor, shape (batch_size, representation_dim)
            context: Optional context information (goals, etc.)

        Returns:
            Action distribution
        """
        pass

    def sample_actions(self,
                      representation: torch.Tensor,
                      context: Optional[Dict[str, Any]] = None,
                      deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions from policy distribution.

        Args:
            representation: Representation tensor
            context: Optional context information
            deterministic: If True, return mode/mean of distribution

        Returns:
            Action tensor
        """
        action_dist = self.forward(representation, context)

        if deterministic:
            if hasattr(action_dist, 'mode'):
                return action_dist.mode
            else:
                return action_dist.mean
        else:
            return action_dist.sample()

    def action_log_prob(self,
                       actions: torch.Tensor,
                       representation: torch.Tensor,
                       context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute log probability of actions under current policy.

        Args:
            actions: Action tensor
            representation: Representation tensor
            context: Optional context information

        Returns:
            Log probabilities tensor
        """
        action_dist = self.forward(representation, context)
        return action_dist.log_prob(actions)

    def entropy(self,
               representation: torch.Tensor,
               context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute policy entropy.

        Args:
            representation: Representation tensor
            context: Optional context information

        Returns:
            Entropy tensor
        """
        action_dist = self.forward(representation, context)
        return action_dist.entropy()

    @property
    def supports_context(self) -> bool:
        """
        Whether this policy head supports context (goals, etc.).

        Override in subclasses that use context.

        Returns:
            False by default
        """
        return False

    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get information about this policy head.

        Returns:
            Dictionary with policy head information
        """
        return {
            'representation_dim': self.representation_dim,
            'action_dim': self.action_dim,
            'discrete_actions': self.discrete_actions,
            'supports_context': self.supports_context,
            'head_type': self.__class__.__name__,
            'device': str(self.device)
        }
