"""
Core Training Orchestration

This module implements the main training loop that orchestrates all components
of the RL framework. It handles experiment setup, training loops, evaluation,
checkpointing, and logging.

Key features for daily research:
- Automatic component initialization from configs
- Seamless checkpoint resuming 
- Built-in evaluation and logging
- Error handling and recovery
- Experiment tracking and organization
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from src.utils.config import Config, ConfigManager, resolve_device
from src.utils.checkpoint import CheckpointManager  
from src.utils.logger import create_logger
from src.utils.registry import (
    get_algorithm, get_environment, get_network, get_buffer,
    auto_import_modules, RegistryMixin
)

logger = logging.getLogger(__name__)


class Trainer(RegistryMixin):
    """
    Main training orchestrator for RL experiments.
    
    Coordinates all components and handles the complete training lifecycle
    from initialization through completion. Designed for seamless daily
    research workflows with automatic resuming and robust error handling.
    
    Attributes:
        config: Complete experiment configuration
        experiment_dir: Directory for this experiment's outputs
        algorithm: RL algorithm instance
        environment: Environment wrapper instance  
        buffer: Experience buffer instance
        networks: Dictionary of network instances
        checkpoint_manager: Handles saving and loading training state
        step: Current global training step
        episode: Current episode number
        metrics: Current training metrics
    """
    
    def __init__(self, config: Config, experiment_dir: Optional[Path] = None):
        """
        Initialize trainer with experiment configuration.
        
        Args:
            config: Validated experiment configuration
            experiment_dir: Custom experiment directory (auto-generated if None)
        """
        self.config = config
        self.step = 0
        self.episode = 0
        self.metrics = {}
        self.start_time = None
        
        # Setup experiment directory
        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config.experiment.name}_{timestamp}"
            self.experiment_dir = Path("experiments") / experiment_name
        else:
            self.experiment_dir = Path(experiment_dir)
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging for this experiment
        self._setup_logging()
        
        # Auto-import modules to populate registries
        auto_import_modules()
        
        # Initialize components
        self._initialize_components()
        
        # Setup checkpoint management
        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
            auto_save_frequency=config.training.checkpoint_frequency,
            max_checkpoints=5
        )
        
        # Initialize experiment logger
        self.experiment_logger = create_logger(
            self.experiment_dir,
            config.logging.__dict__,
            config.to_dict()
        )
        
        # Try to resume from checkpoint
        self._attempt_resume()
        
        logger.info(f"Trainer initialized for experiment: {config.experiment.name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_logging(self):
        """Setup experiment-specific logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler for this experiment
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging setup complete. Log file: {log_file}")
    
    def _initialize_components(self):
        """Initialize all RL components from configuration"""
        try:
            # Initialize environment first (needed for space information)
            logger.info("Initializing environment...")
            env_class = get_environment(self.config.environment.wrapper)
            self.environment = env_class(self.config.environment.__dict__)
            
            # Get space information
            obs_space = self.environment.observation_space
            action_space = self.environment.action_space
            
            logger.info(f"Environment: {self.config.environment.name}")
            logger.info(f"Observation space: {obs_space.shape}")
            logger.info(f"Action space: {action_space.shape}, discrete: {action_space.discrete}")
            
            # Initialize networks with proper dimensions
            logger.info("Initializing networks...")
            self.networks = {}
            
            if isinstance(self.config.network, dict):
                # Multiple networks (e.g., actor-critic)
                for net_name, net_config in self.config.network.items():
                    net_config_dict = net_config.__dict__.copy()
                    
                    # Add device info to network config
                    net_config_dict['device'] = resolve_device(self.config.experiment.device)
                    
                    # Set dimensions if not specified
                    if net_config_dict.get('input_dim') is None:
                        net_config_dict['input_dim'] = obs_space.shape
                    
                    if net_config_dict.get('output_dim') is None:
                        if net_name == 'actor':
                            net_config_dict['output_dim'] = action_space.shape[0] if not action_space.discrete else action_space.n
                        elif net_name == 'critic':
                            net_config_dict['output_dim'] = 1
                    
                    # For continuous control actors, pass algorithm's continuous control parameters
                    # Networks will use what they need and ignore the rest
                    if net_name == 'actor' and not action_space.discrete:
                        algorithm_config = self.config.algorithm.__dict__
                        continuous_params = ['action_bounds', 'log_std_init', 'use_tanh_squashing']
                        for param in continuous_params:
                            if param in algorithm_config:
                                net_config_dict[param] = algorithm_config[param]
                    
                    net_class = get_network(net_config.type)
                    self.networks[net_name] = net_class(net_config_dict)
                    logger.info(f"Created {net_name} network: {net_config.type}")
            
            else:
                # Single network
                net_config = self.config.network.__dict__.copy()
                
                # Add device info to network config
                net_config['device'] = resolve_device(self.config.experiment.device)
                
                if net_config.get('input_dim') is None:
                    net_config['input_dim'] = obs_space.shape
                if net_config.get('output_dim') is None:
                    net_config['output_dim'] = action_space.shape[0] if not action_space.discrete else action_space.n
                
                net_class = get_network(self.config.network.type)
                self.networks['main'] = net_class(net_config)
                logger.info(f"Created main network: {self.config.network.type}")
            
            # Initialize algorithm with networks and spaces
            logger.info("Initializing algorithm...")
            algorithm_config = self.config.algorithm.__dict__.copy()
            algorithm_config['networks'] = self.networks
            algorithm_config['observation_space'] = obs_space
            algorithm_config['action_space'] = action_space
            algorithm_config['device'] = resolve_device(self.config.experiment.device)
            
            algorithm_class = get_algorithm(self.config.algorithm.name)
            self.algorithm = algorithm_class(algorithm_config)
            logger.info(f"Created algorithm: {self.config.algorithm.name}")
            
            # Initialize buffer
            logger.info("Initializing buffer...")
            buffer_config = self.config.buffer.__dict__.copy()
            buffer_config['device'] = resolve_device(self.config.experiment.device)
            
            buffer_class = get_buffer(self.config.buffer.type)
            self.buffer = buffer_class(buffer_config)
            logger.info(f"Created buffer: {self.config.buffer.type}")
            
            # Set random seeds for reproducibility
            self._set_seeds(self.config.experiment.seed)
            
            logger.info("All components initialized successfully")
            
            # Watch model parameters for W&B
            if hasattr(self, 'experiment_logger'):
                for name, network in self.networks.items():
                    self.experiment_logger.watch_model(network, log_freq=1000)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Set environment seed
        try:
            self.environment.seed(seed)
        except Exception as e:
            logger.warning(f"Could not set environment seed: {e}")
        
        logger.info(f"Random seeds set to {seed}")
    
    def _attempt_resume(self):
        """Attempt to resume training from latest checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint is not None:
            logger.info("Resuming from checkpoint...")
            
            trainer_state = {
                'algorithm': self.algorithm,
                'buffer': self.buffer,
                'environment': self.environment,
                'networks': self.networks,
                'metrics': self.metrics
            }
            
            trainer_state = self.checkpoint_manager.restore_training_state(
                trainer_state, checkpoint
            )
            
            self.step = trainer_state['step']
            self.metrics = trainer_state['metrics']
            
            logger.info(f"Resumed from step {self.step}")
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the main training loop.
        
        Returns:
            Dictionary containing final training results
        """
        logger.info("Starting training...")
        logger.info(f"Target steps: {self.config.training.total_timesteps}")
        logger.info(f"Evaluation frequency: {self.config.training.eval_frequency}")
        logger.info(f"Checkpoint frequency: {self.config.training.checkpoint_frequency}")
        
        self.start_time = time.time()
        
        try:
            while self.step < self.config.training.total_timesteps:
                # Training step
                self._training_step()
                
                # Evaluation
                if self.step % self.config.training.eval_frequency == 0:
                    eval_metrics = self._evaluation_step()
                    self.metrics.update(eval_metrics)
                
                # Checkpointing
                checkpoint_path = self.checkpoint_manager.auto_save(
                    self._get_trainer_state(), self.step
                )
                if checkpoint_path is not None:
                    logger.info(f"Auto-saved checkpoint at step {self.step}")
                
                # Logging
                if self.step % self.config.logging.log_frequency == 0:
                    self._log_metrics()
                
                # Check for early termination conditions
                if self._should_terminate():
                    logger.info("Early termination triggered")
                    break
            
            # Final evaluation and checkpoint
            logger.info("Training completed, running final evaluation...")
            final_eval_metrics = self._evaluation_step()
            self.metrics.update(final_eval_metrics)
            
            final_checkpoint = self.checkpoint_manager.save_checkpoint(
                self._get_trainer_state(), self.step, name="final"
            )
            logger.info(f"Final checkpoint saved: {final_checkpoint}")
            
            # Compute final results
            training_time = time.time() - self.start_time
            final_results = {
                'final_step': self.step,
                'training_time': training_time,
                'steps_per_second': self.step / training_time,
                **self.metrics
            }
            
            logger.info(f"Training completed successfully in {training_time:.1f}s")
            
            # Log final hyperparameters and results
            self.experiment_logger.log_hyperparameters(
                self._flatten_config_for_logging(self.config.to_dict()),
                final_results
            )
            
            # Cleanup logging
            self.experiment_logger.finish()
            
            return final_results
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save checkpoint before exiting
            self.checkpoint_manager.save_checkpoint(
                self._get_trainer_state(), self.step, name="interrupted"
            )
            # Cleanup logging
            if hasattr(self, 'experiment_logger'):
                self.experiment_logger.finish()
            raise
            
        except Exception as e:
            import traceback
            logger.error(f"Training failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Save emergency checkpoint
            try:
                self.checkpoint_manager.save_checkpoint(
                    self._get_trainer_state(), self.step, name="error"
                )
                logger.info("Emergency checkpoint saved")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            
            # Cleanup logging
            if hasattr(self, 'experiment_logger'):
                self.experiment_logger.finish()
            raise
    
    def _training_step(self):
        """Execute one training step"""
        # Collect experience
        trajectory = self._collect_experience()
        
        # Add to buffer
        if trajectory is not None:
            self.buffer.add(trajectory=trajectory)
        
        # Update algorithm if buffer is ready
        if self.buffer.ready():
            batch = self.buffer.sample()
            update_metrics = self.algorithm.update(batch)
            self.metrics.update(update_metrics)
            
            # Log training metrics immediately after update
            if update_metrics:
                # Add timing information
                if self.start_time is not None:
                    elapsed = time.time() - self.start_time
                    update_metrics['time_elapsed'] = elapsed
                    update_metrics['steps_per_second'] = self.step / elapsed if elapsed > 0 else 0
                
                # Log training metrics to monitoring systems
                self.experiment_logger.log_metrics(update_metrics, self.step, prefix='train')
            
            # Clear buffer for on-policy algorithms
            if self.config.buffer.type == 'trajectory':
                self.buffer.clear()
    
    def _collect_experience(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Collect experience from environment interaction.
        
        Returns:
            Complete trajectory if episode finished, None otherwise
        """
        observations = []
        actions = []
        rewards = []
        old_values = []
        old_log_probs = []
        dones = []
        
        # Set algorithm to training mode
        self.algorithm.train()
        
        # Reset environment if needed
        if not hasattr(self, '_current_obs') or self._episode_done:
            self._current_obs = self.environment.reset()
            self._episode_done = False
            self.episode += 1
        
        # Collect trajectory up to buffer capacity
        steps_collected = 0
        buffer_capacity = self.config.buffer.capacity
        
        while steps_collected < buffer_capacity and self.step < self.config.training.total_timesteps:
            # Get observation as tensor
            obs_tensor = torch.FloatTensor(self._current_obs).unsqueeze(0).to(self.algorithm.device)
            
            # Get action, log_prob, and value from algorithm
            with torch.no_grad():
                if hasattr(self.algorithm, 'get_action_and_value'):
                    action, log_prob, value = self.algorithm.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy().item()
                    log_prob = log_prob.cpu().numpy().item()
                    value = value.cpu().numpy().item()
                else:
                    # Fallback if get_action_and_value not implemented
                    action_tensor = self.algorithm.act(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy().item()
                    log_prob = 0.0  # Placeholder
                    value = 0.0     # Placeholder
            
            # Step environment
            next_obs, reward, done, info = self.environment.step(action)
            
            # Store experience
            observations.append(self._current_obs)
            actions.append(action)
            rewards.append(reward)
            old_values.append(value)
            old_log_probs.append(log_prob)
            dones.append(done)
            
            # Update state
            self._current_obs = next_obs
            self._episode_done = done
            self.step += 1
            steps_collected += 1
            
            # If episode is done, reset for next episode
            if done:
                self._current_obs = self.environment.reset()
                self._episode_done = False
                self.episode += 1
        
        # Return trajectory data
        if steps_collected > 0:
            trajectory = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'old_values': np.array(old_values),
                'old_log_probs': np.array(old_log_probs),
                'dones': np.array(dones)
            }
            
            # CRITICAL FIX: Add bootstrap value for truncated episodes
            # If the last step was truncated (not terminated), we need the value of the final state
            if steps_collected > 0 and hasattr(self, '_current_obs') and not self._episode_done:
                # Get value of current state for bootstrapping
                obs_tensor = torch.FloatTensor(self._current_obs).unsqueeze(0).to(self.algorithm.device)
                with torch.no_grad():
                    if hasattr(self.algorithm, 'networks') and 'critic' in self.algorithm.networks:
                        bootstrap_value = self.algorithm.networks['critic'](obs_tensor).cpu().numpy().item()
                        trajectory['bootstrap_value'] = bootstrap_value
            
            return trajectory
        
        return None
    
    def _evaluation_step(self) -> Dict[str, float]:
        """
        Run evaluation episodes.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Running evaluation at step {self.step}")
        
        eval_metrics = {
            'eval_step': float(self.step),
            'eval_episode_count': float(self.config.training.num_eval_episodes)
        }
        
        episode_returns = []
        episode_lengths = []
        
        # Set algorithm to evaluation mode
        self.algorithm.eval()
        
        try:
            for eval_episode in range(self.config.training.num_eval_episodes):
                episode_return = 0.0
                episode_length = 0
                
                obs = self.environment.reset(seed=self.config.experiment.seed + eval_episode)
                done = False
                
                while not done and episode_length < 1000:  # Max episode length
                    # Select action deterministically
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.algorithm.device)
                        action_tensor = self.algorithm.act(obs_tensor, deterministic=True)
                        action = action_tensor.cpu().numpy().item()
                    
                    # Step environment
                    obs, reward, done, info = self.environment.step(action)
                    
                    episode_return += reward
                    episode_length += 1
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
            
            # Compute evaluation statistics
            eval_metrics.update({
                'eval_return_mean': float(np.mean(episode_returns)),
                'eval_return_std': float(np.std(episode_returns)),
                'eval_return_min': float(np.min(episode_returns)),
                'eval_return_max': float(np.max(episode_returns)),
                'eval_length_mean': float(np.mean(episode_lengths)),
            })
            
            logger.info(f"Evaluation complete - Mean return: {eval_metrics['eval_return_mean']:.2f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            eval_metrics['eval_error'] = str(e)
        
        finally:
            # Return algorithm to training mode
            self.algorithm.train()
        
        return eval_metrics
    
    def _log_metrics(self):
        """Log current training metrics"""
        if not self.metrics:
            return
        
        # Add timing information
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.metrics['time_elapsed'] = elapsed
            self.metrics['steps_per_second'] = self.step / elapsed if elapsed > 0 else 0
        
        # Separate training and evaluation metrics
        train_metrics = {}
        eval_metrics = {}
        
        for key, value in self.metrics.items():
            if key.startswith('eval_'):
                eval_metrics[key] = value
            else:
                train_metrics[key] = value
        
        # Log training metrics
        if train_metrics:
            self.experiment_logger.log_metrics(train_metrics, self.step, prefix='train')
        
        # Log evaluation metrics
        if eval_metrics:
            self.experiment_logger.log_metrics(eval_metrics, self.step, prefix='eval')
    
    def _should_terminate(self) -> bool:
        """Check if training should terminate early"""
        # TODO: Add early termination conditions
        # e.g., solved environment, performance threshold, etc.
        return False
    
    def _flatten_config_for_logging(self, config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested config for hyperparameter logging"""
        flat_config = {}
        
        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self._flatten_config_for_logging(value, new_key))
            elif isinstance(value, (int, float, str, bool, type(None))):
                flat_config[new_key] = value
            else:
                flat_config[new_key] = str(value)
        
        return flat_config
    
    def _get_trainer_state(self) -> Dict[str, Any]:
        """Get current trainer state for checkpointing"""
        return {
            'algorithm': self.algorithm,
            'buffer': self.buffer,
            'environment': self.environment,
            'networks': self.networks,
            'step': self.step,
            'episode': self.episode,
            'metrics': self.metrics
        }
    
    def save_experiment_config(self):
        """Save experiment configuration to file"""
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Save config as YAML
        from src.utils.config import save_config
        config_path = config_dir / "config.yaml"
        save_config(self.config, config_path)
        
        # Save additional metadata
        metadata = {
            'experiment_name': self.config.experiment.name,
            'config_hash': self.config.get_hash(),
            'created_at': datetime.now().isoformat(),
            'experiment_dir': str(self.experiment_dir),
            'git_hash': self._get_git_hash(),
        }
        
        import json
        metadata_path = config_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Experiment config saved to {config_path}")
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash for reproducibility"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'environment'):
                self.environment.close()
            logger.info("Trainer cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def create_trainer_from_config(config_path: str, 
                             experiment_dir: Optional[str] = None,
                             config_overrides: Optional[Dict[str, Any]] = None) -> Trainer:
    """
    Create trainer from configuration file.
    
    Args:
        config_path: Path to experiment configuration file
        experiment_dir: Custom experiment directory
        config_overrides: Optional configuration overrides
        
    Returns:
        Initialized trainer instance
    """
    from src.utils.config import load_config
    
    # Load configuration
    config = load_config(config_path, config_overrides)
    
    # Create trainer
    trainer = Trainer(config, experiment_dir)
    
    # Save config for reproducibility
    trainer.save_experiment_config()
    
    return trainer