"""
CNN Encoders

CNN encoders reimplemented from existing CNN networks.
Preserves all functionality from the original designs.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np

from .base import BaseEncoder
from ...utils.registry import register_encoder


@register_encoder("minigrid_cnn")
class MiniGridCNNEncoder(BaseEncoder):
    """
    MiniGrid CNN encoder based on the original MiniGridCNN design.

    Preserves the exact architecture and logic from the original:
    - Conv1: 7x7x3 -> 5x5x16 (3x3 conv, stride=1, padding=0)
    - Conv2: 5x5x16 -> 3x3x32 (3x3 conv, stride=1, padding=0)
    - Conv3: 3x3x32 -> 1x1x64 (3x3 conv, stride=1, padding=0)
    - GlobalPool: 1x1x64 -> 64 + direction(4) = 68
    """

    def _build_encoder(self):
        """Build CNN using original MiniGridCNN logic."""
        # Original defaults
        self.use_direction = self.config.get('use_direction', True)
        self.channels = self.config.get('channels', [16, 32, 64])

        # Validate input dimensions - from original
        if self.config.get('input_dim') != (7, 7, 3):
            print(f"Warning: MiniGridCNN designed for (7,7,3) input, got {self.config.get('input_dim')}")

        # Get activation function
        activation_fn = self._get_activation_function(self.config.get('activation', 'relu'))

        # CNN layers for processing 7x7x3 images - exact original architecture
        self.conv_layers = nn.Sequential(
            # Layer 1: 7x7x3 -> 5x5x16
            nn.Conv2d(3, self.channels[0], kernel_size=3, stride=1, padding=0),
            activation_fn,

            # Layer 2: 5x5x16 -> 3x3x32
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=0),
            activation_fn,

            # Layer 3: 3x3x32 -> 1x1x64
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=0),
            activation_fn,
        )

        # Calculate output dimension - from original
        cnn_output_dim = self.channels[2]  # After global pooling

        # Add direction dimension if used
        if self.use_direction:
            self._output_dim = cnn_output_dim + 4  # +4 for one-hot direction
        else:
            self._output_dim = cnn_output_dim

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

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass using original logic exactly."""
        if isinstance(observations, dict):
            # Handle dict input (image + direction) - from original
            image = observations['image']  # (batch, 7, 7, 3)
            direction = observations['direction']  # (batch,) or (batch, 4)
        elif observations.dim() == 4:
            # Handle pure image input
            image = observations  # (batch, 7, 7, 3)
            direction = None
        else:
            # Handle flattened input with direction - from original
            if self.use_direction:
                image_flat = observations[:, :-4]  # All but last 4 elements
                direction = observations[:, -4:]   # Last 4 elements (one-hot)
                image = image_flat.view(-1, 7, 7, 3)
            else:
                image = observations.view(-1, 7, 7, 3)
                direction = None

        # Convert from HWC to CHW format for PyTorch - from original
        if image.dim() == 4 and image.size(-1) == 3:
            image = image.permute(0, 3, 1, 2)  # (batch, 3, 7, 7)

        # Pass through CNN layers
        conv_out = self.conv_layers(image)  # (batch, 64, 1, 1)

        # Global average pooling (flatten spatial dimensions) - from original
        conv_features = conv_out.mean(dim=(-2, -1))  # (batch, 64)

        # Concatenate with direction if available and used - from original
        if self.use_direction:
            if direction is not None:
                if direction.dim() == 1:
                    # Convert scalar direction to one-hot
                    direction_one_hot = torch.zeros(direction.size(0), 4, device=direction.device)
                    direction_one_hot.scatter_(1, direction.long().unsqueeze(1), 1)
                    direction = direction_one_hot
                features = torch.cat([conv_features, direction], dim=1)
            else:
                # If use_direction=True but no direction provided, pad with zeros
                batch_size = conv_features.size(0)
                zero_direction = torch.zeros(batch_size, 4, device=conv_features.device)
                features = torch.cat([conv_features, zero_direction], dim=1)
        else:
            features = conv_features

        return features

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self._output_dim


@register_encoder("nature_cnn")
class NatureCNNEncoder(BaseEncoder):
    """
    Nature CNN encoder based on the original NatureCNN design.

    Architecture from DQN Nature paper: Mnih et al. (2015)
    - Conv1: 84x84x4 -> 20x20x32 (8x8 conv, stride=4)
    - Conv2: 20x20x32 -> 9x9x64 (4x4 conv, stride=2)
    - Conv3: 9x9x64 -> 7x7x64 (3x3 conv, stride=1)
    - FC1: 7*7*64 -> 512
    """

    def _build_encoder(self):
        """Build Nature CNN using original architecture."""
        # Original defaults
        self.channels = self.config.get('channels', [32, 64, 64])
        self.hidden_dim = self.config.get('hidden_dim', 512)

        activation_fn = self._get_activation_function(self.config.get('activation', 'relu'))

        # Original Nature CNN architecture exactly
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, self.channels[0], kernel_size=8, stride=4),
            activation_fn,
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2),
            activation_fn,
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1),
            activation_fn,
        )

        self.flatten = nn.Flatten()

        # Calculate conv output size - from original
        conv_out_size = self._get_conv_output_size()

        # Feature extraction layer
        self.fc_layer = nn.Sequential(
            nn.Linear(conv_out_size, self.hidden_dim),
            activation_fn
        )

        self._output_dim = self.hidden_dim

    def _get_conv_output_size(self):
        """Calculate output size of conv layers - from original."""
        # For 84x84 input with the standard Nature CNN architecture
        return 7 * 7 * self.channels[2]

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

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass using original logic."""
        if isinstance(observations, dict):
            x = observations.get('image', observations.get('observation', list(observations.values())[0]))
        else:
            x = observations

        # Ensure input is in CHW format and contiguous - from original
        if x.dim() == 4 and x.size(-1) == 4:  # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2).contiguous()

        conv_out = self.conv_layers(x)
        conv_flat = self.flatten(conv_out)
        features = self.fc_layer(conv_flat)
        return features

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self._output_dim


@register_encoder("impala_cnn")
class IMPALACNNEncoder(BaseEncoder):
    """
    IMPALA CNN encoder based on the original IMPALACNN design.

    More efficient than Nature CNN with residual connections.
    """

    def _build_encoder(self):
        """Build IMPALA CNN using original architecture."""
        # Original defaults
        self.depths = self.config.get('depths', [16, 32, 32])
        self.use_batch_norm = self.config.get('use_batch_norm', False)

        activation_fn = self._get_activation_function(self.config.get('activation', 'relu'))

        # Initial conv layer - from original
        layers = [nn.Conv2d(4, self.depths[0], kernel_size=3, stride=1, padding=1)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.depths[0]))
        layers.append(activation_fn)

        # Residual blocks - from original
        in_channels = self.depths[0]
        for depth in self.depths[1:]:
            layers.append(self._make_residual_block(in_channels, depth, activation_fn))
            in_channels = depth

        # Global pooling - encoder doesn't need final linear layer
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ])

        self.network = nn.Sequential(*layers)
        self._output_dim = in_channels

    def _make_residual_block(self, in_channels: int, out_channels: int, activation_fn: nn.Module):
        """Create residual block using original logic."""
        layers = []

        # Downsample if changing channels - from original
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_fn)

        # Second conv - from original
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

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

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass using original logic."""
        if isinstance(observations, dict):
            x = observations.get('image', observations.get('observation', list(observations.values())[0]))
        else:
            x = observations

        # Handle format conversion - from original
        if x.dim() == 4 and x.size(-1) in [3, 4]:  # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2)

        return self.network(x)

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self._output_dim