"""
Base Value Function Interface

This module defines the abstract base class for value functions.
Value functions estimate state or state-action values for policy evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseValueFunction(nn.Module, ABC):
    """
    Abstract base class for all value functions.

    Value functions estimate the expected return from states or state-action pairs.
    They are used for policy evaluation and improvement in RL algorithms.
    """

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize value function with configuration.

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
        self.action_dim = self.config.get('action_dim', None)  # None for state-value functions
        self._build_value_function()

    @abstractmethod
    def _build_value_function(self):
        """
        Build the value function architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def forward(self,
                representation: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Estimate value given state representation and optionally action.

        Args:
            representation: State representation, shape (batch_size, representation_dim)
            action: Optional action tensor, shape (batch_size, action_dim)
                   Required for Q-functions, ignored for state-value functions

        Returns:
            Value estimates, shape (batch_size, 1) or (batch_size,)
        """
        pass

    def value_loss(self,
                  representations: torch.Tensor,
                  targets: torch.Tensor,
                  actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute value function learning loss.

        Args:
            representations: State representations
            targets: Target values
            actions: Optional actions (for Q-functions)

        Returns:
            Dictionary of loss components
        """
        values = self.forward(representations, actions)
        if values.dim() > 1:
            values = values.squeeze(-1)  # Remove last dimension if it's 1

        mse_loss = nn.functional.mse_loss(values, targets)

        return {
            'value_loss': mse_loss,
            'value_mse': mse_loss
        }

    @property
    def is_q_function(self) -> bool:
        """
        Whether this is a Q-function (state-action values) or V-function (state values).

        Returns:
            True if this function estimates Q-values, False for V-values
        """
        return self.action_dim is not None

    def get_value_info(self) -> Dict[str, Any]:
        """
        Get information about this value function.

        Returns:
            Dictionary with value function information
        """
        return {
            'representation_dim': self.representation_dim,
            'action_dim': self.action_dim,
            'is_q_function': self.is_q_function,
            'function_type': self.__class__.__name__,
            'device': str(self.device)
        }
