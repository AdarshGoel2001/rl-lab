"""
Base Representation Learner Interface

This module defines the abstract base class for representation learning components.
Representation learners take encoder features and learn structured representations
using techniques like VAEs, contrastive learning, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn


class BaseRepresentationLearner(nn.Module, ABC):
    """
    Abstract base class for all representation learners.

    Representation learners take encoder features and learn structured
    representations that capture useful properties of the data.

    Attributes:
        config: Representation learner configuration dictionary
        device: PyTorch device (cuda/cpu)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize representation learner with configuration.

        Args:
            config: Dictionary containing hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self._build_learner()

    @abstractmethod
    def _build_learner(self):
        """
        Build the representation learner architecture.

        This method should create all the necessary layers and components.
        It's called during __init__.
        """
        pass

    @abstractmethod
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features into structured representation.

        Args:
            features: Feature tensor from encoder, shape (batch_size, feature_dim)

        Returns:
            Representation tensor of shape (batch_size, representation_dim)
        """
        pass

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Decode representation back to feature space.

        Optional method for representation learners that support decoding
        (e.g., autoencoders, VAEs). Override in subclasses that need it.

        Args:
            representation: Representation tensor, shape (batch_size, representation_dim)

        Returns:
            Reconstructed features of shape (batch_size, feature_dim)
        """
        raise NotImplementedError("Decoding not implemented for this representation learner")

    def representation_loss(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute self-supervised learning objective.

        This is the main learning signal for the representation learner.
        Different learners will implement different objectives (reconstruction,
        contrastive, predictive, etc.).

        Args:
            features: Feature tensor from encoder
            **kwargs: Additional arguments (e.g., augmented views, future features)

        Returns:
            Dictionary of loss components
        """
        representation = self.encode(features)

        # Default: no representation loss (identity learner)
        return {
            'representation_loss': torch.tensor(0.0, device=features.device)
        }

    @property
    @abstractmethod
    def representation_dim(self) -> int:
        """
        Get the dimensionality of the learned representation.

        Returns:
            Integer representing the representation dimension
        """
        pass

    @property
    def supports_decoding(self) -> bool:
        """
        Whether this learner supports decoding representations back to features.

        Returns:
            True if decode() method is implemented, False otherwise
        """
        try:
            # Try calling decode with dummy input to see if it's implemented
            dummy_input = torch.zeros(1, self.representation_dim, device=self.device)
            self.decode(dummy_input)
            return True
        except (NotImplementedError, AttributeError):
            return False

    def get_representation_info(self) -> Dict[str, Any]:
        """
        Get information about the representations produced by this learner.

        Returns:
            Dictionary with representation information
        """
        return {
            'representation_dim': self.representation_dim,
            'supports_decoding': self.supports_decoding,
            'learner_type': self.__class__.__name__,
            'device': str(self.device)
        }