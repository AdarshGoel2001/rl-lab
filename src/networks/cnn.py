"""
CNN Network Architectures

This module provides CNN architectures optimized for different RL environments:
- MiniGridCNN: Lightweight CNN for 7x7x3 MiniGrid observations
- NatureCNN: Standard CNN for Atari-style 84x84 observations 
- IMPALACNN: More efficient CNN with residual connections
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseNetwork
from ..utils.registry import register_network


@register_network("minigrid_cnn")
class MiniGridCNN(BaseNetwork):
    """
    Lightweight CNN designed specifically for MiniGrid 7x7x3 observations.
    
    Architecture:
    - Conv1: 7x7x3 -> 5x5x16 (3x3 conv, stride=1, padding=0) 
    - Conv2: 5x5x16 -> 3x3x32 (3x3 conv, stride=1, padding=0)
    - Conv3: 3x3x32 -> 1x1x64 (3x3 conv, stride=1, padding=0)
    - GlobalPool: 1x1x64 -> 64
    - FC: 64 + direction(4) -> output_dim
    
    This creates a compact feature representation that can be used for both
    actor and critic networks in PPO.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MiniGrid CNN.
        
        Args:
            config: Configuration dictionary with keys:
                - input_dim: Should be (7, 7, 3) for MiniGrid
                - output_dim: Output dimension for final layer
                - activation: Activation function ('relu', 'tanh', etc.)
                - use_direction: Whether to concatenate direction info (default: True)
                - channels: List of channel dimensions [16, 32, 64] (optional)
        """
        self.use_direction = config.get('use_direction', True)
        self.channels = config.get('channels', [16, 32, 64])
        
        # Validate input dimensions
        if config.get('input_dim') != (7, 7, 3):
            print(f"Warning: MiniGridCNN designed for (7,7,3) input, got {config.get('input_dim')}")
        
        super().__init__(config)
    
    def _build_network(self):
        """Build the CNN architecture"""
        activation_fn = self.get_activation_function(self.config.get('activation', 'relu'))
        
        # CNN layers for processing 7x7x3 images
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
        
        # Calculate CNN output dimension
        cnn_output_dim = self.channels[2]  # After global pooling
        
        # Add direction dimension if used
        if self.use_direction:
            final_input_dim = cnn_output_dim + 4  # +4 for one-hot direction
        else:
            final_input_dim = cnn_output_dim
        
        # Final fully connected layer
        self.fc = nn.Linear(final_input_dim, self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor - either:
                - (batch, 7, 7, 3) for image only
                - (batch, 7*7*3 + 4) for flattened image + direction
                - Dict with 'image' and 'direction' keys
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        if isinstance(x, dict):
            # Handle dict input (image + direction)
            image = x['image']  # (batch, 7, 7, 3)
            direction = x['direction']  # (batch,) or (batch, 4)
        elif x.dim() == 4:
            # Handle pure image input
            image = x  # (batch, 7, 7, 3)
            direction = None
        else:
            # Handle flattened input with direction
            if self.use_direction:
                image_flat = x[:, :-4]  # All but last 4 elements
                direction = x[:, -4:]   # Last 4 elements (one-hot)
                image = image_flat.view(-1, 7, 7, 3)
            else:
                image = x.view(-1, 7, 7, 3)
                direction = None
        
        # Convert from HWC to CHW format for PyTorch
        if image.dim() == 4 and image.size(-1) == 3:
            image = image.permute(0, 3, 1, 2)  # (batch, 3, 7, 7)
        
        # Pass through CNN layers
        conv_out = self.conv_layers(image)  # (batch, 64, 1, 1)
        
        # Global average pooling (flatten spatial dimensions)
        conv_features = conv_out.mean(dim=(-2, -1))  # (batch, 64)
        
        # Concatenate with direction if available and used
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
        
        # Final fully connected layer
        output = self.fc(features)
        
        return output
    
    def get_feature_dim(self) -> int:
        """Get the dimension of features before the final FC layer"""
        if self.use_direction:
            return self.channels[2] + 4
        else:
            return self.channels[2]


@register_network("nature_cnn")
class NatureCNN(BaseNetwork):
    """
    Nature CNN architecture for Atari environments (84x84 observations).
    
    Based on the DQN Nature paper: Mnih et al. (2015)
    - Conv1: 84x84x4 -> 20x20x32 (8x8 conv, stride=4)
    - Conv2: 20x20x32 -> 9x9x64 (4x4 conv, stride=2)
    - Conv3: 9x9x64 -> 7x7x64 (3x3 conv, stride=1)
    - FC1: 7*7*64 -> 512
    - FC2: 512 -> output_dim
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.channels = config.get('channels', [32, 64, 64])
        self.hidden_dim = config.get('hidden_dim', 512)
        super().__init__(config)
    
    def _build_network(self):
        """Build the Nature CNN architecture"""
        activation_fn = self.get_activation_function(self.config.get('activation', 'relu'))
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, self.channels[0], kernel_size=8, stride=4),
            activation_fn,
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=4, stride=2),
            activation_fn,
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1),
            activation_fn,
        )
        
        self.flatten = nn.Flatten()
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, self.hidden_dim),
            activation_fn,
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def _get_conv_output_size(self):
        """Calculate the output size of conv layers"""
        # For 84x84 input with the standard Nature CNN architecture
        return 7 * 7 * self.channels[2]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Nature CNN"""
        # Ensure input is in CHW format and contiguous
        if x.dim() == 4 and x.size(-1) == 4:  # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        
        conv_out = self.conv_layers(x)
        conv_flat = self.flatten(conv_out)
        output = self.fc_layers(conv_flat)
        return output


@register_network("impala_cnn")
class IMPALACNN(BaseNetwork):
    """
    IMPALA CNN architecture with residual connections.
    
    More efficient than Nature CNN for environments requiring faster training.
    Uses residual blocks and smaller convolutions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.depths = config.get('depths', [16, 32, 32])
        self.use_batch_norm = config.get('use_batch_norm', False)
        super().__init__(config)
    
    def _build_network(self):
        """Build IMPALA CNN with residual blocks"""
        activation_fn = self.get_activation_function(self.config.get('activation', 'relu'))
        
        # Initial conv layer
        layers = [nn.Conv2d(4, self.depths[0], kernel_size=3, stride=1, padding=1)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.depths[0]))
        layers.append(activation_fn)
        
        # Residual blocks
        in_channels = self.depths[0]
        for depth in self.depths[1:]:
            layers.append(self._make_residual_block(in_channels, depth, activation_fn))
            in_channels = depth
        
        # Global pooling and output
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
    
    def _make_residual_block(self, in_channels: int, out_channels: int, activation_fn: nn.Module):
        """Create a residual block"""
        layers = []
        
        # Downsample if changing channels
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_fn)
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through IMPALA CNN"""
        if x.dim() == 4 and x.size(-1) in [3, 4]:  # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2)
        
        return self.network(x)


# Convenience functions
def create_minigrid_cnn(output_dim: int, activation: str = 'relu', use_direction: bool = True) -> MiniGridCNN:
    """Create a MiniGrid CNN with standard configuration"""
    config = {
        'input_dim': (7, 7, 3),
        'output_dim': output_dim,
        'activation': activation,
        'use_direction': use_direction,
    }
    return MiniGridCNN(config)


def create_nature_cnn(output_dim: int, activation: str = 'relu', hidden_dim: int = 512) -> NatureCNN:
    """Create a Nature CNN with standard configuration"""
    config = {
        'input_dim': (84, 84, 4),
        'output_dim': output_dim,
        'activation': activation,
        'hidden_dim': hidden_dim,
    }
    return NatureCNN(config)