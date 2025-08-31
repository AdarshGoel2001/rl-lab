"""
Multi-Layer Perceptron (MLP) Network

This module implements a flexible MLP architecture that can be used as
actor networks, critic networks, or general function approximators in RL.

Key features:
- Configurable depth and width
- Multiple activation functions
- Batch normalization and dropout support  
- Automatic input/output dimension handling
- Works seamlessly with the registry system
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.networks.base import BaseNetwork
from src.utils.registry import register_network


@register_network("mlp")
class MLP(BaseNetwork):
    """
    Multi-Layer Perceptron network.
    
    A flexible MLP implementation that can serve as actor networks,
    critic networks, or general function approximators.
    
    Attributes:
        layers: Sequential container of network layers
        output_layer: Final output layer (separate for easier access)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP network.
        
        Args:
            config: Network configuration containing:
                - input_dim: Input dimension (int)
                - output_dim: Output dimension (int) 
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function name
                - output_activation: Output layer activation (default: linear)
                - batch_norm: Whether to use batch normalization
                - dropout: Dropout probability (0 = no dropout)
                - layer_norm: Whether to use layer normalization
        """
        # Set defaults
        config.setdefault('hidden_dims', [64, 64])
        config.setdefault('activation', 'relu')
        config.setdefault('output_activation', 'linear')
        config.setdefault('batch_norm', False)
        config.setdefault('dropout', 0.0)
        config.setdefault('layer_norm', False)
        
        super().__init__(config)
    
    def _build_network(self):
        """Build the MLP architecture"""
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for MLP")
        if self.output_dim is None:
            raise ValueError("output_dim must be specified for MLP")
        
        # Handle tuple input dimensions (flatten to single dimension)
        if isinstance(self.input_dim, (tuple, list)):
            input_size = int(np.prod(self.input_dim))
        else:
            input_size = int(self.input_dim)
        
        output_size = int(self.output_dim)
        hidden_dims = self.config['hidden_dims']
        
        layers = []
        
        # Input layer
        prev_dim = input_size
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if self.config['batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Layer normalization  
            if self.config['layer_norm']:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            activation = self.get_activation_function(self.config['activation'])
            layers.append(activation)
            
            # Dropout
            if self.config['dropout'] > 0:
                layers.append(nn.Dropout(self.config['dropout']))
            
            prev_dim = hidden_dim
        
        # Store hidden layers
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_size)
        
        # Output activation
        output_activation = self.config.get('output_activation', 'linear')
        if output_activation != 'linear':
            self.output_activation = self.get_activation_function(output_activation)
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, *input_shape)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        x = self.layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation
        x = self.output_activation(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of features before the output layer.
        
        Useful for shared feature extractors.
        
        Returns:
            Feature dimension
        """
        if len(self.config['hidden_dims']) > 0:
            return self.config['hidden_dims'][-1]
        else:
            return self.input_dim
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass up to the last hidden layer (before output).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor before output layer
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)


@register_network("actor_mlp")  
class ActorMLP(MLP):
    """
    MLP specifically designed for actor networks.
    
    Includes common modifications for policy networks like
    tanh output activation for continuous control.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set actor-specific defaults
        config.setdefault('output_activation', 'tanh')  # Common for continuous control
        config.setdefault('initialization', 'orthogonal')
        
        super().__init__(config)
    
    def _build_network(self):
        """Build actor network with policy-specific modifications"""
        super()._build_network()
        
        # Initialize output layer with smaller weights (common practice for policy networks)
        if hasattr(self.output_layer, 'weight'):
            nn.init.orthogonal_(self.output_layer.weight, gain=0.01)


@register_network("critic_mlp")
class CriticMLP(MLP):
    """
    MLP specifically designed for critic networks (value functions).
    
    Typically has linear output activation for value estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set critic-specific defaults
        config.setdefault('output_activation', 'linear')
        config.setdefault('output_dim', 1)  # Value functions typically output single value
        
        super().__init__(config)


@register_network("continuous_actor_mlp")
class ContinuousActorMLP(BaseNetwork):
    """
    MLP specifically designed for continuous action actor networks.
    
    Outputs both mean and log_std for continuous action distributions.
    Supports action bounds through tanh squashing and scaling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set continuous actor specific defaults
        config.setdefault('hidden_dims', [64, 64])
        config.setdefault('activation', 'tanh')
        config.setdefault('initialization', 'orthogonal')
        config.setdefault('log_std_init', 0.0)  # Initial log std (std=1.0)
        config.setdefault('action_bounds', None)  # [[low1,high1], [low2,high2], ...] for each motor
        config.setdefault('use_tanh_squashing', True)  # Use tanh to bound actions
        
        super().__init__(config)
    
    def _build_network(self):
        """Build continuous actor network with mean and log_std outputs"""
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("input_dim and output_dim must be specified for ContinuousActorMLP")
        
        # Handle input dimensions
        if isinstance(self.input_dim, (tuple, list)):
            input_size = int(np.prod(self.input_dim))
        else:
            input_size = int(self.input_dim)
        
        output_size = int(self.output_dim)
        hidden_dims = self.config['hidden_dims']
        
        # Shared feature layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            activation = self.get_activation_function(self.config['activation'])
            layers.append(activation)
            
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Combined output layer: outputs 2 * action_dim (mean + log_std pairs)
        # For 1D action: [mean, log_std]
        # For 2D action: [mean1, log_std1, mean2, log_std2]  
        # For 3D action: [mean1, log_std1, mean2, log_std2, mean3, log_std3]
        self.output_layer = nn.Linear(prev_dim, 2 * output_size)
        
        # Store initial log_std value for initialization
        self.log_std_init = self.config['log_std_init']
        
        # Store action bounds for scaling - convert list of pairs to low/high tensors
        self.use_tanh_squashing = self.config['use_tanh_squashing']
        if self.config['action_bounds'] is not None:
            bounds = self.config['action_bounds']
            # Convert [[low1,high1], [low2,high2], ...] to separate low/high tensors
            if len(bounds) != output_size:
                raise ValueError(f"action_bounds must have {output_size} pairs, got {len(bounds)}")
            
            lows = [pair[0] for pair in bounds]
            highs = [pair[1] for pair in bounds]
            self.register_buffer('action_low', torch.tensor(lows, dtype=torch.float32))
            self.register_buffer('action_high', torch.tensor(highs, dtype=torch.float32))
        else:
            # Default to [-1, 1] for each dimension
            self.register_buffer('action_low', torch.full((output_size,), -1.0, dtype=torch.float32))
            self.register_buffer('action_high', torch.full((output_size,), 1.0, dtype=torch.float32))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Initialize output layer with special care
        # Split initialization: mean part gets small weights, log_std part gets specific init
        output_dim = self.output_dim
        
        # Initialize mean part (first half of weights) with small weights for better tanh gradients
        nn.init.orthogonal_(self.output_layer.weight[:output_dim], gain=0.01)
        nn.init.zeros_(self.output_layer.bias[:output_dim])
        
        # Initialize log_std part (second half of weights) with small weights
        nn.init.orthogonal_(self.output_layer.weight[output_dim:], gain=0.01) 
        nn.init.constant_(self.output_layer.bias[output_dim:], self.log_std_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning both mean and log_std as network outputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with shape (batch_size, 2 * action_dim) where:
            First half is means, second half is log_stds (PPO format)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.shared_layers(x)
        
        # Network outputs 2*action_dim values
        network_output = self.output_layer(features)
        
        action_dim = self.output_dim
        
        # Split network output in half: first half for means, second half for log_stds
        raw_mean = network_output[:, :action_dim]        # First half
        raw_log_std = network_output[:, action_dim:]     # Second half
        
        # Apply tanh squashing to mean if requested
        if self.use_tanh_squashing:
            mean = torch.tanh(raw_mean)
            # Scale to action bounds using registered buffers
            mean = self.action_low + 0.5 * (self.action_high - self.action_low) * (mean + 1)
        else:
            mean = raw_mean
        
        log_std = raw_log_std
        
        # Return in PPO expected format: [mean1, mean2, ..., log_std1, log_std2, ...]
        return torch.cat([mean, log_std], dim=1)
    
    def get_action_distribution(self, x: torch.Tensor):
        """
        Get action distribution for the given input.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.distributions.Normal distribution
        """
        output = self.forward(x)
        action_dim = self.output_dim
        
        mean = output[:, :action_dim]
        log_std = output[:, action_dim:]
        std = torch.exp(log_std)
        
        return Normal(mean, std)


@register_network("dueling_mlp")
class DuelingMLP(BaseNetwork):
    """
    Dueling network architecture for Q-learning.
    
    Separates state value and advantage estimation as in 
    "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016).
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set dueling-specific defaults
        config.setdefault('hidden_dims', [64, 64])
        config.setdefault('activation', 'relu')
        
        super().__init__(config)
    
    def _build_network(self):
        """Build dueling architecture with shared features, separate value/advantage heads"""
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("input_dim and output_dim must be specified for DuelingMLP")
        
        # Handle input dimensions
        if isinstance(self.input_dim, (tuple, list)):
            input_size = int(np.prod(self.input_dim))
        else:
            input_size = int(self.input_dim)
        
        output_size = int(self.output_dim)
        hidden_dims = self.config['hidden_dims']
        
        # Shared feature extractor
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.get_activation_function(self.config['activation']))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value head (single output)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Advantage head (one output per action)
        self.advantage_head = nn.Linear(prev_dim, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values combining value and advantage estimates
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Shared features
        features = self.feature_extractor(x)
        
        # Value and advantage estimation
        value = self.value_head(features)  # Shape: (batch_size, 1)
        advantage = self.advantage_head(features)  # Shape: (batch_size, num_actions)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values