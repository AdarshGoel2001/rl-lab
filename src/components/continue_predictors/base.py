"""
Base Continue Predictor Interface

This module defines the abstract base class for continue (discount) predictors.
Continue predictors estimate whether an episode will continue from a given state.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class BaseContinuePredictor(nn.Module, ABC):
    """
    Abstract base class for all continue predictors.

    Continue predictors estimate the probability that an episode continues
    from a given state representation. They predict P(episode continues | state),
    which is equivalent to predicting (1 - done).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize continue predictor with configuration.

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
        Build the continue predictor architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor) -> Bernoulli:
        """
        Predict continue distribution given state representation.

        Args:
            state: State representation tensor, shape (batch_size, state_dim)

        Returns:
            Bernoulli distribution over continuation (1 = continue, 0 = terminate)
        """
        pass

    def continue_loss(self,
                     states: torch.Tensor,
                     continues: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute continue prediction learning loss.

        Args:
            states: State representations, shape (batch_size, state_dim)
            continues: True continuation flags, shape (batch_size,) or (batch_size, 1)
                      Should be 1 - done (i.e., 1 = episode continues, 0 = terminal)

        Returns:
            Dictionary of loss components
        """
        predicted_dist = self.forward(states)

        # Ensure continues are the right shape and type
        if continues.dim() > 1:
            continues = continues.squeeze(-1)
        continues = continues.float()

        # Binary cross-entropy loss (using logits for numerical stability)
        # Note: Bernoulli.log_prob already handles this, but we can also use F.binary_cross_entropy_with_logits
        bce_loss = -predicted_dist.log_prob(continues).mean()

        # Compute accuracy for monitoring
        predictions = (predicted_dist.probs > 0.5).float()
        accuracy = (predictions == continues).float().mean()

        return {
            'continue_loss': bce_loss,
            'continue_bce': bce_loss,
            'continue_accuracy': accuracy
        }

    def get_continue_info(self) -> Dict[str, Any]:
        """
        Get information about this continue predictor.

        Returns:
            Dictionary with continue predictor information
        """
        return {
            'state_dim': self.state_dim,
            'predictor_type': self.__class__.__name__,
            'device': str(self.device)
        }
