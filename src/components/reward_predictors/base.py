"""
Base Reward Predictor Interface

This module defines the abstract base class for reward predictors.
Reward predictors estimate expected rewards from state representations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.distributions import Distribution


class BaseRewardPredictor(nn.Module, ABC):
    """
    Abstract base class for all reward predictors.

    Reward predictors estimate expected rewards from state representations.
    They are essential components for model-based RL and world model paradigms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward predictor with configuration.

        Args:
            config: Dictionary containing hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.state_dim = config.get('state_dim')
        self._build_predictor()

    @abstractmethod
    def _build_predictor(self):
        """
        Build the reward predictor architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor) -> Distribution:
        """
        Predict reward distribution given state representation.

        Args:
            state: State representation tensor, shape (batch_size, state_dim)

        Returns:
            Distribution over rewards
        """
        pass

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

        return {
            'reward_loss': nll_loss,
            'reward_nll': nll_loss
        }

    def get_reward_info(self) -> Dict[str, Any]:
        """
        Get information about this reward predictor.

        Returns:
            Dictionary with reward predictor information
        """
        return {
            'state_dim': self.state_dim,
            'predictor_type': self.__class__.__name__,
            'device': str(self.device)
        }
