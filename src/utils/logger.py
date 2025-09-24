"""
Universal Logging System

This module provides a simple, plug-and-play logging system that works consistently
across all environments and algorithms. Key features:

- Accept any metrics dictionary, never block logging
- Automatic output to bash, TensorBoard, and W&B
- Consistent metric naming via synonym mapping
- Environment agnostic (single/vectorized)

Design principle: Write algorithm once, logging works everywhere automatically.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch

# Optional imports with fallbacks
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for logging backends"""
    terminal: bool = True
    tensorboard: bool = True
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = None
    wandb_notes: Optional[str] = None
    log_frequency: int = 1000  # Only used by trainer for deciding when to call logger
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []


class UniversalLogger:
    """
    Universal logging system that accepts any metrics and outputs to all channels.
    
    Key features:
    - Never fails or blocks logging
    - Consistent output to bash, TensorBoard, and W&B
    - Automatic metric name standardization
    - Environment agnostic
    """
    
    # Synonym mapping for consistent metric names (only core harmonization)
    SYNONYM_MAP = {
        # Returns/Rewards harmonization
        'reward_mean': 'return_mean',
        'mean_episode_return': 'return_mean', 
        'episode_return_mean': 'return_mean',
        'reward_std': 'return_std',
        'episode_return_std': 'return_std',
        'reward_min': 'return_min',
        'episode_return_min': 'return_min',
        'reward_max': 'return_max',
        'episode_return_max': 'return_max',
        
        # Episode lengths harmonization
        'episode_length_mean': 'length_mean',
        'mean_episode_length': 'length_mean',
        'episode_length_std': 'length_std',
        
        # Throughput harmonization
        'steps_per_second': 'sps',
        'throughput': 'sps',
    }
    
    # Short names for bash output (space-efficient) 
    BASH_NAMES = {
        'step': 'Step',
        'return_mean': 'Return',
        'length_mean': 'EpLen', 
        # Support both full and short loss names for bash display
        'policy_loss': 'PiLoss',
        'pi_loss': 'PiLoss',
        'value_loss': 'VfLoss', 
        'vf_loss': 'VfLoss',
        'entropy_loss': 'EntLoss',
        'ent_loss': 'EntLoss',
        'total_loss': 'Loss',
        'grad_norm': 'GradNorm',
        'sps': 'SPS',
    }
    
    def __init__(self, experiment_dir: Path, config: LoggingConfig, 
                 experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize universal logger.
        
        Args:
            experiment_dir: Directory for this experiment's outputs
            config: Logging configuration
            experiment_config: Full experiment configuration for metadata
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.experiment_config = experiment_config or {}
        
        # Create directories
        self.log_dir = self.experiment_dir / "logs"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.log_dir.mkdir(exist_ok=True)
        self.tensorboard_dir.mkdir(exist_ok=True)
        
        # Initialize backends
        self.tensorboard_writer = None
        self.wandb_run = None
        self._setup_backends()
        
        # Bash output file for Claude Code monitoring
        self.bash_log_file = self.log_dir / "bash_output.log"
        
        # Track start time for metrics
        self.start_time = time.time()
        
        logger.info(f"Universal logger initialized: {experiment_dir}")
    
    def _setup_backends(self):
        """Initialize logging backends"""
        
        # TensorBoard setup
        if self.config.tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(self.tensorboard_dir),
                    comment=self.experiment_config.get('experiment', {}).get('name', 'experiment')
                )
                logger.info(f"TensorBoard logging enabled: {self.tensorboard_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.tensorboard_writer = None
        
        # Weights & Biases setup
        if self.config.wandb_enabled and WANDB_AVAILABLE:
            try:
                exp_config = self.experiment_config.get('experiment', {})
                
                wandb_config = {
                    'project': self.config.wandb_project or 'rl-experiments',
                    'entity': self.config.wandb_entity,
                    'name': exp_config.get('name', 'unnamed-experiment'),
                    'tags': self.config.wandb_tags,
                    'notes': self.config.wandb_notes,
                    'dir': str(self.experiment_dir),
                    'config': self._flatten_config(self.experiment_config)
                }
                
                # Remove None values
                wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
                
                self.wandb_run = wandb.init(**wandb_config)
                logger.info(f"W&B logging enabled: {self.wandb_run.url}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.wandb_run = None
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested config dictionary for W&B"""
        flat_config = {}
        
        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, new_key))
            elif isinstance(value, (int, float, str, bool, type(None))):
                flat_config[new_key] = value
            else:
                flat_config[new_key] = str(value)
        
        return flat_config
    
    def sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Sanitize metrics to valid float values.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            Dictionary with sanitized float values
        """
        sanitized = {}
        
        for key, value in metrics.items():
            try:
                # Convert various types to float
                if isinstance(value, (int, float)):
                    float_val = float(value)
                elif isinstance(value, np.ndarray):
                    if value.size == 1:
                        float_val = float(value.item())
                    else:
                        float_val = float(value.mean())
                elif isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        float_val = float(value.item())
                    else:
                        float_val = float(value.mean().item())
                elif hasattr(value, '__len__') and len(value) == 1:
                    float_val = float(value[0])
                else:
                    float_val = float(value)
                
                # Filter non-finite values
                if np.isfinite(float_val):
                    sanitized[key] = float_val
                else:
                    logger.warning(f"Dropped non-finite metric {key}: {value}")
                    
            except (ValueError, TypeError, AttributeError):
                logger.warning(f"Could not convert metric {key} to float: {value}")
                continue
        
        return sanitized
    
    def standardize_names(self, metrics: Dict[str, float], prefix: Optional[str] = None) -> Dict[str, float]:
        """
        Standardize metric names using synonym mapping and prefixes.
        
        Args:
            metrics: Sanitized metrics dictionary
            prefix: Optional prefix (e.g., 'train', 'eval')
            
        Returns:
            Dictionary with standardized metric names
        """
        standardized = {}
        
        for key, value in metrics.items():
            # Preserve existing namespaces while applying synonyms
            if '/' in key:
                # Key has namespace: "ppo/clip_ratio" or "debug/actor_mean"
                namespace_parts = key.split('/')
                clean_key = namespace_parts[-1]  # Get the actual metric name
                namespace = '/'.join(namespace_parts[:-1])  # Get the namespace
                
                # Apply synonym mapping to just the metric name
                mapped_key = self.SYNONYM_MAP.get(clean_key, clean_key)
                
                # Rebuild: prefix/namespace/mapped_key or namespace/mapped_key
                if prefix and not key.startswith(prefix + '/'):
                    final_key = f"{prefix}/{namespace}/{mapped_key}"
                else:
                    final_key = f"{namespace}/{mapped_key}"
            else:
                # Key has no namespace: "policy_loss", "return_mean"
                clean_key = key
                
                # Apply synonym mapping
                mapped_key = self.SYNONYM_MAP.get(clean_key, clean_key)
                
                # Apply prefix if provided and not already prefixed
                if prefix and not key.startswith(prefix + '/'):
                    final_key = f"{prefix}/{mapped_key}"
                else:
                    final_key = mapped_key
            
            standardized[final_key] = value
        
        return standardized
    
    def format_bash_output(self, metrics: Dict[str, float], step: int) -> str:
        """
        Format metrics for bash/terminal output.
        
        Args:
            metrics: Standardized metrics dictionary
            step: Current step number
            
        Returns:
            Formatted string for bash output
        """
        # Priority metrics for bash display (most important first)
        # Support both full and abbreviated loss names
        priority_keys = [
            'step', 'return_mean', 'length_mean', 
            'policy_loss', 'pi_loss', 'value_loss', 'vf_loss', 'total_loss', 'sps'
        ]
        
        parts = [f"Step {step}"]
        
        # Add priority metrics first
        for key in priority_keys:
            if key == 'step':
                continue  # Already added
                
            # Look for key with any prefix
            value = None
            for metric_key, metric_value in metrics.items():
                clean_key = metric_key.split('/')[-1] if '/' in metric_key else metric_key
                if clean_key == key:
                    value = metric_value
                    break
            
            if value is not None:
                bash_name = self.BASH_NAMES.get(key, key)
                if isinstance(value, float):
                    if key == 'sps':
                        parts.append(f"{bash_name}: {value:.0f}")
                    elif abs(value) >= 10:
                        parts.append(f"{bash_name}: {value:.2f}")
                    else:
                        parts.append(f"{bash_name}: {value:.4f}")
                else:
                    parts.append(f"{bash_name}: {value}")
        
        return " | ".join(parts)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, 
                   prefix: Optional[str] = None, commit: bool = True):
        """
        Log metrics to all enabled backends.
        
        This is the main API - accepts any metrics dictionary and ensures
        output to bash, TensorBoard, and W&B with consistent naming.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Current training step
            prefix: Optional prefix for metric names (e.g., 'train/', 'eval/')
            commit: Whether to commit/flush the metrics immediately
        """
        print(f"LOGGER DEBUG: Received metrics with prefix '{prefix}': {metrics}", flush=True)
        if not metrics:
            return
        
        try:
            # Step 1: Sanitize metrics (convert to floats, filter non-finite)
            sanitized = self.sanitize_metrics(metrics)
            if not sanitized:
                return
            
            # Step 2: Standardize names (synonyms + prefixes)
            standardized = self.standardize_names(sanitized, prefix)
            print(f"LOGGER DEBUG: After standardization: {standardized}", flush=True)
            
            # Step 3: Always write to bash file (no frequency gating here)
            bash_output = self.format_bash_output(standardized, step)
            try:
                with open(self.bash_log_file, "a") as f:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {bash_output}\n")
            except Exception as e:
                logger.warning(f"Failed to write bash log: {e}")
            
            # Step 4: TensorBoard logging
            if self.tensorboard_writer is not None:
                try:
                    print(f"LOGGER DEBUG: Writing to TensorBoard: {list(standardized.keys())}", flush=True)
                    for name, value in standardized.items():
                        print(f"LOGGER DEBUG: TensorBoard add_scalar({name}, {value}, {step})", flush=True)
                        self.tensorboard_writer.add_scalar(name, value, step)
                    if commit:
                        self.tensorboard_writer.flush()
                except Exception as e:
                    logger.warning(f"TensorBoard logging failed: {e}")
                    print(f"LOGGER DEBUG: TensorBoard error: {e}", flush=True)
            
            # Step 5: W&B logging
            if self.wandb_run is not None:
                try:
                    # Convert slashes to dots for W&B
                    wandb_metrics = {k.replace('/', '.'): v for k, v in standardized.items()}
                    wandb_metrics['step'] = step
                    self.wandb_run.log(wandb_metrics, step=step, commit=commit)
                except Exception as e:
                    logger.warning(f"W&B logging failed: {e}")
            
        except Exception as e:
            logger.error(f"Logging failed completely: {e}")
            # Never let logging failures stop training
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """Log hyperparameters with optional final metrics"""
        
        # TensorBoard hyperparameter logging
        if self.tensorboard_writer is not None:
            try:
                tb_hparams = {}
                for k, v in hparams.items():
                    if isinstance(v, (int, float, str, bool)):
                        tb_hparams[k] = v
                    else:
                        tb_hparams[k] = str(v)
                
                self.tensorboard_writer.add_hparams(
                    tb_hparams, 
                    metrics or {},
                    run_name='hparams'
                )
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
        
        # W&B automatically logs config, but we can update it
        if self.wandb_run is not None:
            try:
                self.wandb_run.config.update(hparams)
            except Exception as e:
                logger.warning(f"Failed to update W&B config: {e}")
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """Log histogram data"""
        if self.tensorboard_writer is not None:
            try:
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                self.tensorboard_writer.add_histogram(name, values, step)
            except Exception as e:
                logger.warning(f"Histogram logging failed: {e}")
        
        if self.wandb_run is not None and WANDB_AVAILABLE:
            try:
                self.wandb_run.log({f"{name}_hist": wandb.Histogram(values)}, step=step)
            except Exception as e:
                logger.warning(f"W&B histogram logging failed: {e}")
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model gradients and parameters"""
        if self.wandb_run is not None:
            try:
                self.wandb_run.watch(model, log_freq=log_freq, log_graph=True)
            except Exception as e:
                logger.warning(f"Model watching failed: {e}")
    
    def finish(self):
        """Clean up logging backends"""
        # TensorBoard cleanup
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"TensorBoard cleanup failed: {e}")
        
        # W&B cleanup
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"W&B cleanup failed: {e}")


# Alias for backwards compatibility
ExperimentLogger = UniversalLogger


def create_logger(experiment_dir: Path, logging_config: Dict[str, Any], 
                 experiment_config: Optional[Dict[str, Any]] = None) -> UniversalLogger:
    """
    Create and configure universal logger.
    
    Args:
        experiment_dir: Experiment directory
        logging_config: Logging configuration
        experiment_config: Full experiment configuration
        
    Returns:
        Configured UniversalLogger instance
    """
    # Convert dict to LoggingConfig
    config = LoggingConfig(
        terminal=logging_config.get('terminal', True),
        tensorboard=logging_config.get('tensorboard', True),
        wandb_enabled=logging_config.get('wandb_enabled', False),
        wandb_project=logging_config.get('wandb', {}).get('project'),
        wandb_entity=logging_config.get('wandb', {}).get('entity'),
        wandb_tags=logging_config.get('wandb', {}).get('tags', []),
        wandb_notes=logging_config.get('wandb', {}).get('notes'),
        log_frequency=logging_config.get('log_frequency', 1000)
    )
    
    return UniversalLogger(experiment_dir, config, experiment_config)