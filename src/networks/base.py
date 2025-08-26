"""
Base Network Interface

This module provides the abstract base class for all neural network architectures
used in RL algorithms. It ensures consistent interfaces for different network types
(MLPs, CNNs, RNNs, Transformers, etc.).

Key benefits:
- Consistent initialization and forward pass interface
- Built-in device management and parameter counting
- Easy network swapping via configuration
- Automatic weight initialization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base class for all neural networks in RL algorithms.
    
    This provides a consistent interface for creating and using different
    network architectures. All network implementations should inherit from
    this class.
    
    Attributes:
        config: Network configuration dictionary
        input_dim: Input dimension (automatically determined)
        output_dim: Output dimension
        device: PyTorch device for computation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base network.
        
        Args:
            config: Dictionary containing network configuration including:
                - input_dim: Input dimension (int or tuple)
                - output_dim: Output dimension (int)
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function name
                - initialization: Weight initialization scheme
                - device: PyTorch device
        """
        super().__init__()
        
        self.config = config
        self.input_dim = config.get('input_dim')
        self.output_dim = config.get('output_dim')
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Build the network architecture
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    @abstractmethod
    def _build_network(self):
        """
        Build the network architecture.
        
        This should create all layers and components of the network.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def _initialize_weights(self):
        """
        Initialize network weights.
        
        Uses the initialization scheme specified in config, with sensible
        defaults for different layer types.
        """
        init_scheme = self.config.get('initialization', 'xavier_uniform')
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_scheme == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_scheme == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_scheme == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif init_scheme == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif init_scheme == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                
                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Conv2d):
                if init_scheme in ['kaiming_uniform', 'kaiming_normal']:
                    if init_scheme == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    else:
                        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_activation_function(self, name: str) -> nn.Module:
        """
        Get activation function by name.
        
        Args:
            name: Name of activation function
            
        Returns:
            PyTorch activation module
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),  # SiLU is Swish
            'gelu': nn.GELU(),
            'softplus': nn.Softplus(),
            'linear': nn.Identity(),
            'none': nn.Identity(),
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}. "
                           f"Available: {list(activations.keys())}")
        
        return activations[name.lower()]
    
    def parameter_count(self) -> Dict[str, int]:
        """
        Get parameter count statistics.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate layer outputs for debugging/visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer names to their outputs
        """
        outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output.detach()
            return hook
        
        # Register hooks for all named modules
        handles = []
        for name, module in self.named_modules():
            if name:  # Skip empty name (root module)
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            self.forward(x)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return outputs
    
    def freeze_parameters(self, layer_names: Optional[list] = None):
        """
        Freeze parameters to prevent training.
        
        Args:
            layer_names: List of layer names to freeze. If None, freeze all.
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
    
    def unfreeze_parameters(self, layer_names: Optional[list] = None):
        """
        Unfreeze parameters to enable training.
        
        Args:
            layer_names: List of layer names to unfreeze. If None, unfreeze all.
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
    
    def summary(self) -> str:
        """
        Get a summary of the network architecture.
        
        Returns:
            String summary of the network
        """
        lines = [f"{self.__class__.__name__} Summary:"]
        lines.append("-" * 50)
        
        # Add layer information
        for name, module in self.named_modules():
            if name and len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                lines.append(f"{name:30s} {str(module):30s} {param_count:>10,} params")
        
        # Add total parameter count
        params = self.parameter_count()
        lines.append("-" * 50)
        lines.append(f"Total parameters: {params['total']:,}")
        lines.append(f"Trainable parameters: {params['trainable']:,}")
        
        return "\n".join(lines)
    
    def save_architecture(self, filepath: str):
        """
        Save network architecture to file for reproducibility.
        
        Args:
            filepath: Path to save architecture
        """
        import json
        
        arch_info = {
            'class': self.__class__.__name__,
            'config': self.config,
            'parameter_count': self.parameter_count(),
            'architecture': str(self)
        }
        
        with open(filepath, 'w') as f:
            json.dump(arch_info, f, indent=2)