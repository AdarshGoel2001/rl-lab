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
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from collections import deque
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
        
        # Episode tracking for logging (using deques for recent episodes)
        self.episode_returns = deque(maxlen=100)  # Last 100 completed episodes
        self.episode_lengths = deque(maxlen=100)  # Last 100 completed episodes
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        
        # Track cumulative metrics for progress visibility
        self.total_episodes = 0
        self.last_progress_step = 0
        self.last_comprehensive_step = 0
        
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
            # Initialize training environment first (needed for space information)
            logger.info("Initializing training environment...")
            env_class = get_environment(self.config.environment.wrapper)
            self.environment = env_class(self.config.environment.__dict__)
            
            # Initialize separate evaluation environment
            logger.info("Initializing evaluation environment...")
            if hasattr(self.config, 'evaluation'):
                # Use explicit evaluation config
                eval_env_class = get_environment(self.config.evaluation.wrapper)
                self.eval_environment = eval_env_class(self.config.evaluation.__dict__)
                logger.info(f"Created evaluation environment with wrapper: {self.config.evaluation.wrapper}")
            else:
                # Fallback: derive single environment from training environment config
                eval_config = self.config.environment.__dict__.copy()
                # Override to use single environment wrapper
                if self.config.environment.wrapper == 'vectorized_gym':
                    eval_config['wrapper'] = 'gym'
                    # Remove vectorization-specific configs
                    eval_config.pop('num_envs', None)
                    eval_config.pop('vectorization', None)
                elif self.config.environment.wrapper == 'atari':
                    eval_config['wrapper'] = 'gym'
                    # Remove parallel environment configs
                    eval_config.pop('num_environments', None)
                    eval_config.pop('parallel_backend', None)
                    eval_config.pop('start_method', None)
                
                eval_env_class = get_environment(eval_config['wrapper'])
                self.eval_environment = eval_env_class(eval_config)
                logger.info(f"Created derived evaluation environment with wrapper: {eval_config['wrapper']}")
            
            # Get space information from training environment
            obs_space = self.environment.observation_space
            action_space = self.environment.action_space
            is_vectorized_env = getattr(self.environment, 'is_vectorized', False)
            
            logger.info(f"Training environment: {self.config.environment.name}")
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
                    
                    # Set dimensions if not specified (use per-env shapes for vectorized envs)
                    if net_config_dict.get('input_dim') is None:
                        input_shape = obs_space.shape[1:] if is_vectorized_env else obs_space.shape
                        net_config_dict['input_dim'] = input_shape
                    
                    if net_config_dict.get('output_dim') is None:
                        if net_name == 'actor':
                            if action_space.discrete:
                                net_config_dict['output_dim'] = action_space.n
                            else:
                                action_shape = action_space.shape[1:] if is_vectorized_env else action_space.shape
                                # Action shape may be multi-dimensional; flatten to scalar dimension
                                net_config_dict['output_dim'] = int(np.prod(action_shape))
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
                    input_shape = obs_space.shape[1:] if is_vectorized_env else obs_space.shape
                    net_config['input_dim'] = input_shape
                if net_config.get('output_dim') is None:
                    if action_space.discrete:
                        net_config['output_dim'] = action_space.n
                    else:
                        action_shape = action_space.shape[1:] if is_vectorized_env else action_space.shape
                        net_config['output_dim'] = int(np.prod(action_shape))
                
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
        """Attempt to resume training from latest checkpoint or load transfer checkpoint"""
        # First check if there's a specific checkpoint to load for transfer learning
        transfer_checkpoint_path = getattr(self.config.training, 'load_checkpoint', None)
        
        if transfer_checkpoint_path:
            logger.info(f"Loading checkpoint for transfer learning: {transfer_checkpoint_path}")
            checkpoint = self.checkpoint_manager.load_checkpoint(transfer_checkpoint_path, load_latest=False)
        else:
            # Default behavior: load latest checkpoint from current experiment
            checkpoint = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint is not None:
            transfer_checkpoint_path = getattr(self.config.training, 'load_checkpoint', None)
            
            if transfer_checkpoint_path:
                logger.info("Loading checkpoint for transfer learning...")
                # For transfer learning: load only network weights, start training fresh
                trainer_state = {
                    'algorithm': self.algorithm,
                    'buffer': self.buffer,
                    'environment': self.environment,
                    'networks': self.networks,
                    'metrics': self.metrics
                }
                
                # Load only algorithm weights, not training progress
                trainer_state = self.checkpoint_manager.restore_training_state(
                    trainer_state, checkpoint, transfer_learning=True
                )
                
                # Keep fresh training state for new environment
                self.step = 0
                self.metrics = {'episodes': 0, 'total_reward': 0.0}
                
                logger.info("Transfer learning initialized - starting fresh training with pretrained weights")
            else:
                logger.info("Resuming from checkpoint...")
                # Normal resume: restore everything including training progress
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
        
        print("DEBUG: Immediately after logger statements", flush=True)
        sys.stdout.flush()
        print("DEBUG: Skipping start_time for now", flush=True)
        # Skip timing for now to bypass the hang
        self.start_time = 0  # We'll fix timing later
        print("DEBUG: Skipped start_time successfully")
        
        print("DEBUG: About to check step initialization")
        # Ensure step is initialized
        if not hasattr(self, 'step'):
            self.step = 0
            print("DEBUG: self.step was not initialized, setting to 0")
        else:
            print(f"DEBUG: self.step already exists: {self.step}")
        
        print(f"DEBUG: About to start training loop, self.step = {self.step}, target = {self.config.training.total_timesteps}")
        
        try:
            while self.step < self.config.training.total_timesteps:
                print(f"DEBUG: Starting training loop iteration, step: {self.step}")
                # Training step
                self._training_step()
                
                # Evaluation
                if self.step % self.config.training.eval_frequency == 0:
                    eval_metrics = self._evaluation_step()
                    # Log eval metrics immediately with eval prefix
                    self.experiment_logger.log_metrics(eval_metrics, self.step, prefix='eval')
                    # Store prefixed eval metrics for final results (compatibility)
                    prefixed_eval = {f'eval_{k}': v for k, v in eval_metrics.items()}
                    self.metrics.update(prefixed_eval)
                
                # Checkpointing
                checkpoint_path = self.checkpoint_manager.auto_save(
                    self._get_trainer_state(), self.step
                )
                if checkpoint_path is not None:
                    logger.info(f"Auto-saved checkpoint at step {self.step}")
                
                # Frequent progress logging for bash visibility (every ~1000 steps)
                # Use floor division to handle cases where steps don't align perfectly
                if self.step // 1000 > self.last_progress_step // 1000:
                    self._log_progress_metrics()
                    self.last_progress_step = self.step
                
                # Comprehensive logging at configured frequency
                # Use floor division to handle step size misalignment
                log_freq = self.config.logging.log_frequency
                if self.step // log_freq > self.last_comprehensive_step // log_freq:
                    self._log_comprehensive_metrics()
                    self.last_comprehensive_step = self.step
                
                # Check for early termination conditions
                if self._should_terminate():
                    logger.info("Early termination triggered")
                    break
            
            # Final evaluation and checkpoint
            logger.info("Training completed, running final evaluation...")
            final_eval_metrics = self._evaluation_step()
            # Log final eval metrics
            self.experiment_logger.log_metrics(final_eval_metrics, self.step, prefix='eval')
            # Store prefixed for final results
            final_prefixed_eval = {f'eval_{k}': v for k, v in final_eval_metrics.items()}
            self.metrics.update(final_prefixed_eval)
            
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
        # Collect experience (handles both single and vectorized environments)
        trajectory = self._collect_experience()
        
        # Add to buffer
        if trajectory is not None:
            self.buffer.add(trajectory=trajectory)
        
        # Update algorithm if buffer is ready
        if self.buffer.ready():
            batch = self.buffer.sample()
            update_metrics = self.algorithm.update(batch)
            self.metrics.update(update_metrics)
            
            # Log algorithm update metrics immediately
            if update_metrics:
                # Just log the algorithm metrics - don't add extra fields here
                # Episode aggregation will be handled separately
                self.experiment_logger.log_metrics(update_metrics, self.step, prefix='train')
            
            # Clear buffer for on-policy algorithms
            if self.config.buffer.type == 'trajectory':
                self.buffer.clear()
    
    def _collect_experience(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Collect experience from environment interaction.
        Automatically handles both single and vectorized environments.
        
        Returns:
            Complete trajectory if buffer capacity reached, None otherwise
        """
        # Check if environment is vectorized
        if hasattr(self.environment, 'is_vectorized') and self.environment.is_vectorized:
            return self._collect_vectorized_experience()
        else:
            return self._collect_single_experience()
    
    def _collect_single_experience(self) -> Optional[Dict[str, np.ndarray]]:
        """Collect experience from single environment"""
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
            
            # Track episode statistics
            self.current_episode_return += reward
            self.current_episode_length += 1
            
            # Update state
            self._current_obs = next_obs
            self._episode_done = done
            self.step += 1
            steps_collected += 1
            
            # If episode is done, record episode metrics and reset
            if done:
                self.episode_returns.append(self.current_episode_return)
                self.episode_lengths.append(self.current_episode_length)
                self.total_episodes += 1
                
                self.current_episode_return = 0.0
                self.current_episode_length = 0
                
                self._current_obs = self.environment.reset()
                self._episode_done = False
                self.episode += 1
        
        # Return trajectory data
        if steps_collected > 0:
            trajectory = {
                'observations': np.stack([
                    obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
                    for obs in observations
                ], axis=0),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'old_values': np.array(old_values),
                'old_log_probs': np.array(old_log_probs),
                'dones': np.array(dones)
            }
            
            # Add bootstrap value for truncated episodes
            if steps_collected > 0 and hasattr(self, '_current_obs') and not self._episode_done:
                # Ensure current_obs is numpy for consistent tensor creation
                current_obs_np = self._current_obs.cpu().numpy() if isinstance(self._current_obs, torch.Tensor) else self._current_obs
                obs_tensor = torch.FloatTensor(current_obs_np).unsqueeze(0).to(self.algorithm.device)
                with torch.no_grad():
                    if hasattr(self.algorithm, 'networks') and 'critic' in self.algorithm.networks:
                        bootstrap_value = self.algorithm.networks['critic'](obs_tensor).cpu().numpy().item()
                        trajectory['bootstrap_value'] = bootstrap_value
            
            return trajectory
        
        return None
    
    def _collect_vectorized_experience(self) -> Optional[Dict[str, np.ndarray]]:
        """Collect experience from vectorized environments"""
        print("DEBUG: Starting _collect_vectorized_experience")
        observations = []
        actions = []
        rewards = []
        old_values = []
        old_log_probs = []
        dones = []
        
        num_envs = self.environment.num_envs
        
        # Set algorithm to training mode
        self.algorithm.train()
        
        # Reset environments if needed
        if not hasattr(self, '_current_obs') or not hasattr(self, '_episode_dones'):
            print("DEBUG: Resetting vectorized environment...")
            self._current_obs = self.environment.reset()  # Shape: (num_envs, obs_dim)
            print(f"DEBUG: Reset successful, obs shape: {self._current_obs.shape}")
            self._episode_dones = np.zeros(num_envs, dtype=bool)
            self.episode += num_envs  # Count all environments
        
        # Collect trajectory up to buffer capacity
        steps_collected = 0
        buffer_capacity = self.config.buffer.capacity
        
        while steps_collected < buffer_capacity and self.step < self.config.training.total_timesteps:
            # Get observations as tensor (already batched)
            obs_tensor = self._current_obs.to(self.algorithm.device)
            
            # Get batched actions, log_probs, and values from algorithm
            with torch.no_grad():
                if hasattr(self.algorithm, 'get_action_and_value'):
                    actions_batch, log_probs_batch, values_batch = self.algorithm.get_action_and_value(obs_tensor)
                    actions_batch = actions_batch.cpu().numpy()
                    log_probs_batch = log_probs_batch.cpu().numpy()
                    values_batch = values_batch.cpu().numpy()
                else:
                    # Fallback if get_action_and_value not implemented
                    actions_tensor = self.algorithm.act(obs_tensor, deterministic=False)
                    actions_batch = actions_tensor.cpu().numpy()
                    log_probs_batch = np.zeros(num_envs)  # Placeholder
                    values_batch = np.zeros(num_envs)     # Placeholder
            
            # Step vectorized environment
            next_obs, rewards_batch, dones_batch, infos = self.environment.step(actions_batch)
            
            # Store experience (all environments at once)
            observations.append(self._current_obs.cpu().numpy())
            actions.append(actions_batch)
            rewards.append(rewards_batch.cpu().numpy())
            old_values.append(values_batch)
            old_log_probs.append(log_probs_batch)
            dones.append(dones_batch.cpu().numpy())
            
            # Update state
            self._current_obs = next_obs
            self._episode_dones = dones_batch.cpu().numpy()
            
            # Update step count (count all environment steps)
            self.step += num_envs
            steps_collected += 1
            
            # Process completed episodes in vectorized environments
            dones_np = dones_batch.cpu().numpy() if isinstance(dones_batch, torch.Tensor) else dones_batch
            completed_episodes = np.sum(dones_np)
            
            if completed_episodes > 0:
                self.episode += completed_episodes
                self.total_episodes += completed_episodes
                
                # Extract episode returns and lengths from infos if available
                if hasattr(infos, '__iter__') and len(infos) > 0:
                    for i, (done, info) in enumerate(zip(dones_np, infos)):
                        if done and isinstance(info, dict):
                            episode_return = info.get('episode_return', info.get('episode', {}).get('r', 0.0))
                            episode_length = info.get('episode_length', info.get('episode', {}).get('l', 0))
                            
                            if episode_return != 0.0 or episode_length != 0:  # Valid episode data
                                self.episode_returns.append(episode_return)
                                self.episode_lengths.append(episode_length)
        
        # Return trajectory data
        if steps_collected > 0:
            # Stack all collected data
            # Shape will be (steps_collected, num_envs, ...)
            trajectory = {
                'observations': np.array(observations),      # (steps, num_envs, obs_dim)
                'actions': np.array(actions),                # (steps, num_envs, action_dim)
                'rewards': np.array(rewards),                # (steps, num_envs)
                'old_values': np.array(old_values),          # (steps, num_envs)
                'old_log_probs': np.array(old_log_probs),    # (steps, num_envs)
                'dones': np.array(dones)                     # (steps, num_envs)
            }
            
            # Add bootstrap values for any non-terminated environments
            if steps_collected > 0 and hasattr(self, '_current_obs'):
                obs_tensor = self._current_obs.to(self.algorithm.device)
                with torch.no_grad():
                    if hasattr(self.algorithm, 'networks') and 'critic' in self.algorithm.networks:
                        bootstrap_values = self.algorithm.networks['critic'](obs_tensor).cpu().numpy()
                        trajectory['bootstrap_values'] = bootstrap_values  # Shape: (num_envs,)
            
            return trajectory
        
        return None
    
    def _evaluation_step(self) -> Dict[str, float]:
        """
        Run evaluation episodes using the separate evaluation environment.
        
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
                
                # Use the dedicated evaluation environment
                obs = self.eval_environment.reset(seed=self.config.experiment.seed + eval_episode)
                done = False
                
                while not done and episode_length < 1000:  # Max episode length
                    # Select action deterministically
                    with torch.no_grad():
                        # Handle single environment observation shape
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.algorithm.device)
                        action_tensor = self.algorithm.act(obs_tensor, deterministic=True)
                        
                        # Handle action extraction based on environment type
                        if self.eval_environment.action_space.discrete:
                            # Discrete action: extract scalar
                            action = action_tensor.cpu().numpy().item()
                        else:
                            # Continuous action: extract array (squeeze batch dimension)
                            action = action_tensor.cpu().numpy().squeeze(0)
                    
                    # Step evaluation environment
                    obs, reward, done, info = self.eval_environment.step(action)
                    
                    episode_return += reward
                    episode_length += 1
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
            
            # Compute evaluation statistics with standardized names
            eval_metrics.update({
                'return_mean': float(np.mean(episode_returns)),
                'return_std': float(np.std(episode_returns)),
                'return_min': float(np.min(episode_returns)),
                'return_max': float(np.max(episode_returns)),
                'length_mean': float(np.mean(episode_lengths)),
            })
            
            logger.info(f"Evaluation complete - Mean return: {eval_metrics['return_mean']:.2f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(f"Error details: {repr(e)}")
            eval_metrics['eval_error'] = str(e)
        
        finally:
            # Return algorithm to training mode
            self.algorithm.train()
        
        return eval_metrics
    
    def _log_progress_metrics(self):
        """Log minimal progress metrics for frequent bash visibility"""
        progress_metrics = {
            'step': self.step,
            'episode': self.total_episodes,
        }
        
        # Add episode returns if available (key for progress visibility)
        if len(self.episode_returns) > 0:
            progress_metrics['return_mean'] = float(np.mean(list(self.episode_returns)))
            progress_metrics['length_mean'] = float(np.mean(list(self.episode_lengths)))
        
        # Add simple SPS calculation
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            progress_metrics['sps'] = self.step / elapsed if elapsed > 0 else 0
        
        self.experiment_logger.log_metrics(progress_metrics, self.step, prefix='train')
    
    def _log_comprehensive_metrics(self):
        """Log comprehensive training metrics including episode aggregations"""
        comprehensive_metrics = {
            'step': self.step,
            'episode': self.total_episodes,
        }
        
        # Add episode return aggregations if we have episode data
        if len(self.episode_returns) > 0:
            returns_array = np.array(list(self.episode_returns))
            comprehensive_metrics.update({
                'return_mean': returns_array.mean(),
                'return_std': returns_array.std(),
                'return_min': returns_array.min(),
                'return_max': returns_array.max(),
            })
        
        # Add episode length aggregations if we have episode data
        if len(self.episode_lengths) > 0:
            lengths_array = np.array(list(self.episode_lengths))
            comprehensive_metrics.update({
                'length_mean': lengths_array.mean(),
                'length_std': lengths_array.std(),
            })
        
        # Add timing information
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            comprehensive_metrics.update({
                'time_elapsed': elapsed,
                'sps': self.step / elapsed if elapsed > 0 else 0
            })
        
        # Add any other training metrics from self.metrics (exclude eval_* since they're logged separately)
        train_metrics = {}
        
        for key, value in self.metrics.items():
            if not key.startswith('eval_'):  # Skip eval metrics - they're logged separately
                train_metrics[key] = value
        
        # Merge with comprehensive metrics
        comprehensive_metrics.update(train_metrics)
        
        # Log comprehensive training metrics
        self.experiment_logger.log_metrics(comprehensive_metrics, self.step, prefix='train')
    
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
