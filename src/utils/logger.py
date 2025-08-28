"""
Advanced Logging and Monitoring System

This module provides unified logging to multiple backends including TensorBoard,
Weights & Biases (W&B), and terminal output. Designed for seamless experiment
tracking and visualization.

Key features:
- Multi-backend logging (TensorBoard, W&B, terminal)
- Automatic metric aggregation and smoothing
- Real-time visualization support
- Experiment comparison and analysis
- Automatic hyperparameter logging
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

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
    log_frequency: int = 1000
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []


class ExperimentLogger:
    """
    Multi-backend experiment logger for RL training.
    
    Handles logging to TensorBoard, Weights & Biases, and terminal output
    with automatic metric aggregation and visualization support.
    """
    
    def __init__(self, experiment_dir: Path, config: LoggingConfig, 
                 experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Directory for this experiment's outputs
            config: Logging configuration
            experiment_config: Full experiment configuration for metadata
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.experiment_config = experiment_config or {}
        
        # Create logging directories
        self.log_dir = self.experiment_dir / "logs"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.log_dir.mkdir(exist_ok=True)
        self.tensorboard_dir.mkdir(exist_ok=True)
        
        # Initialize backends
        self.tensorboard_writer = None
        self.wandb_run = None
        self._setup_backends()
        
        # Metrics tracking
        self.step_metrics = {}
        self.episode_metrics = []
        self.start_time = time.time()
        
        logger.info(f"Experiment logger initialized: {experiment_dir}")
    
    def _setup_backends(self):
        """Initialize logging backends based on configuration"""
        
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
                # Extract experiment metadata
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
                # Convert other types to string
                flat_config[new_key] = str(value)
        
        return flat_config
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, 
                   prefix: str = '', commit: bool = True):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Current training step
            prefix: Optional prefix for metric names (e.g., 'train/', 'eval/')
            commit: Whether to commit/flush the metrics immediately
        """
        # Add prefix to metric names
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
        
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Terminal logging
        if self.config.terminal and step % self.config.log_frequency == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in prefixed_metrics.items()])
            logger.info(f"Step {step} | {metrics_str}")
        
        # TensorBoard logging
        if self.tensorboard_writer is not None:
            for metric_name, value in prefixed_metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    self.tensorboard_writer.add_scalar(metric_name, value, step)
            
            if commit:
                self.tensorboard_writer.flush()
        
        # W&B logging
        if self.wandb_run is not None:
            wandb_metrics = {'step': step, **prefixed_metrics}
            # Filter out NaN/inf values
            wandb_metrics = {k: v for k, v in wandb_metrics.items() 
                           if not (isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)))}
            self.wandb_run.log(wandb_metrics, step=step, commit=commit)
        
        # Store for internal tracking
        self.step_metrics[step] = prefixed_metrics
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """Log hyperparameters with optional final metrics"""
        
        # TensorBoard hyperparameter logging
        if self.tensorboard_writer is not None:
            try:
                # Convert values to appropriate types for TensorBoard
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
    
    def log_episode(self, episode_data: Dict[str, Any], episode: int):
        """Log episode-specific data"""
        self.episode_metrics.append(episode_data)
        
        # Log episode metrics with episode number as step
        episode_metrics = {k: v for k, v in episode_data.items() if isinstance(v, (int, float))}
        self.log_metrics(episode_metrics, episode, prefix='episode', commit=True)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """Log histogram data"""
        if self.tensorboard_writer is not None:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self.tensorboard_writer.add_histogram(name, values, step)
        
        if self.wandb_run is not None:
            self.wandb_run.log({f"{name}_hist": wandb.Histogram(values), 'step': step}, step=step)
    
    def log_video(self, name: str, video: np.ndarray, step: int, fps: int = 30):
        """Log video data"""
        if self.tensorboard_writer is not None:
            # TensorBoard expects video in shape (N, T, C, H, W)
            if video.ndim == 4:  # (T, H, W, C)
                video = video.transpose(3, 0, 1, 2)[None]  # Add batch dim and reorder
            self.tensorboard_writer.add_video(name, video, step, fps=fps)
        
        if self.wandb_run is not None:
            self.wandb_run.log({name: wandb.Video(video, fps=fps, format="mp4")}, step=step)
    
    def save_model(self, model_state: Dict[str, Any], step: int, name: str = 'model'):
        """Save model checkpoint with logging"""
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}_step_{step}.pt"
        torch.save(model_state, checkpoint_path)
        
        # Log model artifact to W&B
        if self.wandb_run is not None:
            try:
                artifact = wandb.Artifact(f"{name}_step_{step}", type="model")
                artifact.add_file(str(checkpoint_path))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log model to W&B: {e}")
        
        return checkpoint_path
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model gradients and parameters"""
        if self.wandb_run is not None:
            self.wandb_run.watch(model, log_freq=log_freq, log_graph=True)
    
    def finish(self):
        """Clean up logging backends"""
        # TensorBoard cleanup
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        # W&B cleanup
        if self.wandb_run is not None:
            self.wandb_run.finish()
            logger.info("W&B run finished")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the experiment"""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_runtime': total_time,
            'total_steps': len(self.step_metrics),
            'total_episodes': len(self.episode_metrics),
            'steps_per_second': len(self.step_metrics) / total_time if total_time > 0 else 0
        }
        
        return summary


def create_logger(experiment_dir: Path, logging_config: Dict[str, Any], 
                 experiment_config: Optional[Dict[str, Any]] = None) -> ExperimentLogger:
    """
    Create and configure experiment logger.
    
    Args:
        experiment_dir: Experiment directory
        logging_config: Logging configuration
        experiment_config: Full experiment configuration
        
    Returns:
        Configured ExperimentLogger instance
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
    
    return ExperimentLogger(experiment_dir, config, experiment_config)
