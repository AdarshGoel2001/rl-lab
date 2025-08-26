"""
Configuration Loading and Validation System

This module handles loading, validating, and managing experiment configurations.
It supports YAML configs with environment variable substitution, config inheritance,
and automatic validation.

Key benefits for researchers:
- YAML-based configs are human readable and version controllable
- Environment variable substitution for sensitive data
- Config inheritance for shared settings
- Automatic validation prevents configuration errors
- Hash-based config tracking for reproducibility
"""

import os
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid"""
    pass


@dataclass
class ExperimentConfig:
    """Structured experiment configuration"""
    name: str
    seed: int = 42
    device: str = 'cpu'
    debug: bool = False
    
    def __post_init__(self):
        # Validate device
        if self.device not in ['cpu', 'cuda', 'auto']:
            logger.warning(f"Unusual device specified: {self.device}")


@dataclass 
class AlgorithmConfig:
    """Algorithm configuration"""
    name: str
    lr: float = 3e-4
    
    def __post_init__(self):
        if self.lr <= 0:
            raise ConfigError(f"Learning rate must be positive, got {self.lr}")


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    name: str
    wrapper: str = 'gym'
    normalize_obs: bool = False
    normalize_reward: bool = False
    max_episode_steps: Optional[int] = None


@dataclass
class NetworkConfig:
    """Network configuration"""
    type: str
    input_dim: Optional[Union[int, tuple]] = None
    output_dim: Optional[int] = None
    hidden_dims: list = None
    activation: str = 'relu'
    output_activation: str = 'linear'
    initialization: str = 'xavier_uniform'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


@dataclass
class BufferConfig:
    """Buffer configuration"""
    type: str = 'trajectory'
    capacity: int = 100000
    batch_size: int = 64
    
    def __post_init__(self):
        if self.capacity <= 0:
            raise ConfigError(f"Buffer capacity must be positive, got {self.capacity}")
        if self.batch_size <= 0:
            raise ConfigError(f"Batch size must be positive, got {self.batch_size}")


@dataclass
class TrainingConfig:
    """Training configuration"""
    total_timesteps: int = 1000000
    eval_frequency: int = 10000
    checkpoint_frequency: int = 50000
    num_eval_episodes: int = 10
    
    def __post_init__(self):
        if self.total_timesteps <= 0:
            raise ConfigError(f"Total timesteps must be positive, got {self.total_timesteps}")


@dataclass
class LoggingConfig:
    """Logging configuration"""
    terminal: bool = True
    tensorboard: bool = False
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_tags: list = None
    log_frequency: int = 1000
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []


@dataclass
class Config:
    """Main configuration container"""
    experiment: ExperimentConfig
    algorithm: AlgorithmConfig  
    environment: EnvironmentConfig
    network: Union[NetworkConfig, Dict[str, NetworkConfig]]
    buffer: BufferConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        try:
            # Handle networks - can be single network or multiple networks
            network_config = config_dict.get('network', {})
            if isinstance(network_config, dict) and any(
                k in network_config for k in ['actor', 'critic', 'encoder', 'decoder']
            ):
                # Multiple networks
                networks = {}
                for net_name, net_config in network_config.items():
                    networks[net_name] = NetworkConfig(**net_config)
                network = networks
            else:
                # Single network
                network = NetworkConfig(**network_config)
            
            return cls(
                experiment=ExperimentConfig(**config_dict.get('experiment', {})),
                algorithm=AlgorithmConfig(**config_dict.get('algorithm', {})),
                environment=EnvironmentConfig(**config_dict.get('environment', {})),
                network=network,
                buffer=BufferConfig(**config_dict.get('buffer', {})),
                training=TrainingConfig(**config_dict.get('training', {})),
                logging=LoggingConfig(**config_dict.get('logging', {}))
            )
        except TypeError as e:
            raise ConfigError(f"Invalid configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary"""
        config_dict = asdict(self)
        
        # Handle network conversion
        if isinstance(self.network, dict):
            config_dict['network'] = {k: asdict(v) for k, v in self.network.items()}
        else:
            config_dict['network'] = asdict(self.network)
            
        return config_dict
    
    def get_hash(self) -> str:
        """Get deterministic hash of configuration for reproducibility tracking"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def expand_environment_variables(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in config dictionary.
    
    Supports ${VAR} and ${VAR:default} syntax.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dictionary with environment variables expanded
    """
    import re
    
    def expand_value(value):
        if isinstance(value, str):
            # Pattern matches ${VAR} or ${VAR:default}  
            pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_value)
            
            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(v) for v in value]
        else:
            return value
    
    return expand_value(config_dict)


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable expansion.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file {config_path}: {e}")
    
    if config_dict is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")
    
    # Expand environment variables
    config_dict = expand_environment_variables(config_dict)
    
    # Handle config inheritance
    if 'base_config' in config_dict:
        base_config_path = config_path.parent / config_dict.pop('base_config')
        base_config = load_yaml_config(base_config_path)
        config_dict = merge_configs(base_config, config_dict)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config_dict


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(config_path: Union[str, Path], 
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load and validate complete configuration.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional dictionary of configuration overrides
        
    Returns:
        Validated Config object
    """
    # Load base config
    config_dict = load_yaml_config(config_path)
    
    # Apply overrides
    if overrides:
        config_dict = merge_configs(config_dict, overrides)
    
    # Create and validate config object
    try:
        config = Config.from_dict(config_dict)
        logger.info(f"Configuration loaded successfully (hash: {config.get_hash()})")
        return config
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}")


def save_config(config: Config, save_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {save_path}")


def validate_config_compatibility(config: Config) -> List[str]:
    """
    Validate that configuration components are compatible.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []
    
    # Check algorithm-buffer compatibility
    if config.algorithm.name in ['ppo', 'a2c'] and config.buffer.type != 'trajectory':
        warnings.append(f"Algorithm {config.algorithm.name} typically uses trajectory buffer, "
                       f"but {config.buffer.type} buffer configured")
    
    if config.algorithm.name in ['dqn', 'sac'] and config.buffer.type == 'trajectory':
        warnings.append(f"Algorithm {config.algorithm.name} typically uses replay buffer, "
                       f"but trajectory buffer configured")
    
    # Check device availability
    if config.experiment.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("CUDA device requested but not available")
        except ImportError:
            warnings.append("PyTorch not available for device validation")
    
    # Check wandb configuration
    if config.logging.wandb_enabled and not config.logging.wandb_project:
        warnings.append("WandB enabled but no project specified")
    
    return warnings


class ConfigManager:
    """
    Utility class for managing configurations during experiments.
    
    Handles config loading, validation, saving, and tracking.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[Config] = None
        
        if self.config_path:
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None, 
                   overrides: Optional[Dict[str, Any]] = None):
        """Load configuration from file"""
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ConfigError("No configuration path specified")
        
        self.config = load_config(self.config_path, overrides)
        
        # Validate compatibility
        warnings = validate_config_compatibility(self.config)
        for warning in warnings:
            logger.warning(f"Config validation warning: {warning}")
    
    def save_config(self, save_path: Union[str, Path]):
        """Save current configuration"""
        if not self.config:
            raise ConfigError("No configuration loaded")
        
        save_config(self.config, save_path)
    
    def get_config_hash(self) -> str:
        """Get hash of current configuration"""
        if not self.config:
            raise ConfigError("No configuration loaded")
        
        return self.config.get_hash()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if not self.config:
            raise ConfigError("No configuration loaded")
        
        config_dict = self.config.to_dict()
        updated_config_dict = merge_configs(config_dict, updates)
        self.config = Config.from_dict(updated_config_dict)