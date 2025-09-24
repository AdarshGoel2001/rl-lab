"""
Identity Representation Learner

A pass-through representation learner that doesn't modify the input features.
Useful when you want to skip representation learning or for simple environments.
"""

from typing import Dict, Any
import torch

from .base import BaseRepresentationLearner
from ...utils.registry import register_representation_learner


@register_representation_learner("identity")
class IdentityRepresentationLearner(BaseRepresentationLearner):
    """
    Identity representation learner that passes features through unchanged.

    This is useful when you want to skip representation learning or
    when using pre-trained features that don't need further processing.
    """

    def _build_learner(self):
        """Build the identity learner (no-op)."""
        # Get representation dimension from config or infer from input
        self._representation_dim = self.config.get('representation_dim')

        # If not specified, we'll set it during the first forward pass
        if self._representation_dim is None:
            self._representation_dim = None  # Will be set on first call

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pass features through unchanged.

        Args:
            features: Feature tensor from encoder, shape (batch_size, feature_dim)

        Returns:
            Same feature tensor unchanged
        """
        # Set representation dim on first call if not specified
        if self._representation_dim is None:
            self._representation_dim = features.shape[-1]

        return features

    def decode(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Pass representation through unchanged (identity decoding).

        Args:
            representation: Representation tensor

        Returns:
            Same representation tensor unchanged
        """
        return representation

    def representation_loss(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute representation learning loss (none for identity).

        Args:
            features: Feature tensor from encoder
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with zero loss
        """
        return {
            'representation_loss': torch.tensor(0.0, device=features.device, requires_grad=True)
        }

    @property
    def representation_dim(self) -> int:
        """
        Get the dimensionality of the learned representation.

        Returns:
            Same as input feature dimension
        """
        if self._representation_dim is None:
            raise ValueError("Representation dimension not set. Call encode() first.")
        return self._representation_dim