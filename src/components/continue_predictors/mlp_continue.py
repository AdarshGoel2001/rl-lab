"""
MLP Continue Predictor

A continue predictor using multi-layer perceptrons.
Predicts episode continuation probability as a Bernoulli distribution.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .base import BaseContinuePredictor
from ...utils.registry import register_continue_predictor


@register_continue_predictor("mlp")
class MLPContinuePredictor(BaseContinuePredictor):
    """
    MLP-based continue predictor.

    Learns to predict whether an episode will continue from a given state
    as a Bernoulli distribution (binary classification).
    """

    def _build_predictor(self):
        """Build the MLP continue predictor."""
        if self.state_dim is None:
            raise ValueError("MLPContinuePredictor requires 'state_dim' in config")

        hidden_dims = self.config.get('hidden_dims', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.0)

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

        # Build MLP
        layers = []
        current_dim = self.state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_fn)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Output layer (logit for binary classification)
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Bernoulli:
        """
        Predict continue distribution given state representation.

        Args:
            state: State representation tensor, shape (batch_size, state_dim)

        Returns:
            Bernoulli distribution over continuation
        """
        # Predict logits
        logits = self.network(state).squeeze(-1)

        # Return Bernoulli distribution
        return Bernoulli(logits=logits)

    def continue_loss(self,
                     states: torch.Tensor,
                     continues: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute continue prediction learning loss.

        Args:
            states: State representations, shape (batch_size, state_dim)
            continues: True continuation flags, shape (batch_size,) or (batch_size, 1)

        Returns:
            Dictionary of loss components
        """
        predicted_dist = self.forward(states)

        # Ensure continues are the right shape and type
        if continues.dim() > 1:
            continues = continues.squeeze(-1)
        continues = continues.float()

        # Binary cross-entropy loss
        bce_loss = -predicted_dist.log_prob(continues).mean()

        # Compute accuracy and other metrics for monitoring
        predictions = (predicted_dist.probs > 0.5).float()
        accuracy = (predictions == continues).float().mean()

        # Compute precision and recall for continues (class 1)
        true_positives = ((predictions == 1) & (continues == 1)).float().sum()
        predicted_positives = (predictions == 1).float().sum()
        actual_positives = (continues == 1).float().sum()

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        return {
            'continue_loss': bce_loss,
            'continue_bce': bce_loss,
            'continue_accuracy': accuracy,
            'continue_precision': precision,
            'continue_recall': recall
        }

    def get_continue_info(self) -> Dict[str, Any]:
        """Get information about this continue predictor."""
        info = super().get_continue_info()
        info.update({
            'hidden_dims': self.config.get('hidden_dims', [256, 256])
        })
        return info
