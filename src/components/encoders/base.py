"""
Base Encoder Interface

This module defines the abstract base class that all encoders must implement.
Encoders transform raw observations into feature representations that can be
used by downstream components like representation learners and policy heads.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Union
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoders.

    Encoders transform observations to feature representations. They handle
    the first stage of processing in the modular agent architecture.

    Attributes:
        config: Encoder configuration dictionary
        device: PyTorch device (cuda/cpu)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize encoder with configuration.

        Args:
            config: Dictionary containing encoder hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self._build_encoder()

    @abstractmethod
    def _build_encoder(self):
        """
        Build the encoder architecture.

        This method should create all the necessary layers and components
        for the encoder. It's called during __init__.
        """
        pass

    @abstractmethod
    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Transform observations to feature representations.

        Args:
            observations: Raw observations from environment. Can be:
                - Single tensor for simple observation spaces
                - Dictionary of tensors for complex observation spaces
                  (e.g., {'image': image_tensor, 'proprioception': proprio_tensor})

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Get the dimensionality of the encoder's output features.

        Returns:
            Integer representing the feature dimension
        """
        pass

    @property
    def supports_sequences(self) -> bool:
        """
        Whether this encoder can handle temporal sequences.

        Override in subclasses that support sequence input.

        Returns:
            False by default (most encoders process single timesteps)
        """
        return False

    @property
    def supports_multimodal(self) -> bool:
        """
        Whether this encoder can handle multiple input modalities.

        Override in subclasses that can fuse multiple input types.

        Returns:
            False by default (most encoders handle single modality)
        """
        return False

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the features produced by this encoder.

        Useful for debugging and component compatibility checking.

        Returns:
            Dictionary with feature information
        """
        return {
            'output_dim': self.output_dim,
            'supports_sequences': self.supports_sequences,
            'supports_multimodal': self.supports_multimodal,
            'encoder_type': self.__class__.__name__,
            'device': str(self.device)
        }

    def reset(self):
        """
        Reset encoder state (for stateful encoders like RNNs).

        Override in subclasses that maintain internal state.
        Default implementation does nothing.
        """
        pass