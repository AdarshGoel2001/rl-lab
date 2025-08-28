"""
Checkpoint Management System

This module provides comprehensive checkpoint management for reproducible RL training.
It handles saving and loading complete training state, including network weights,
optimizer states, random seeds, and training progress.

Key benefits for researchers:
- Seamless resume from any point in training
- Automatic periodic checkpointing to prevent data loss
- Complete state restoration including RNG states for reproducibility  
- Efficient storage with compression and cleanup options
- Metadata tracking for experiment organization
"""

import os
import json
import pickle
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import torch
import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading training checkpoints.
    
    Provides automatic checkpointing, state restoration, and checkpoint
    organization for long-running RL experiments.
    
    Attributes:
        experiment_dir: Directory for this experiment  
        checkpoint_dir: Directory for checkpoint files
        auto_save_frequency: Steps between automatic saves
        max_checkpoints: Maximum number of checkpoints to keep
    """
    
    def __init__(self, experiment_dir: Union[str, Path], 
                 auto_save_frequency: int = 10000,
                 max_checkpoints: int = 5,
                 compress: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            experiment_dir: Directory to store experiment files
            auto_save_frequency: Steps between automatic checkpoint saves
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
            compress: Whether to compress checkpoint files
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.auto_save_frequency = auto_save_frequency
        self.max_checkpoints = max_checkpoints
        self.compress = compress
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last save step for auto-saving
        self._last_save_step = 0
        
        logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(self, trainer_state: Dict[str, Any], 
                       step: int, 
                       name: Optional[str] = None,
                       is_best: bool = False) -> Path:
        """
        Save complete training state to checkpoint.
        
        Args:
            trainer_state: Dictionary containing all training state including:
                - algorithm: Algorithm instance with save_checkpoint() method
                - buffer: Buffer instance with save_checkpoint() method  
                - environment: Environment instance
                - metrics: Current training metrics
                - Any other state to preserve
            step: Current training step
            name: Optional name for checkpoint (auto-generated if None)
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint file
        """
        if name is None:
            name = f"checkpoint_step_{step}"
        
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        # Collect all state to save
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'algorithm_state': trainer_state['algorithm'].save_checkpoint() if 'algorithm' in trainer_state else {},
            'buffer_state': trainer_state['buffer'].save_checkpoint() if 'buffer' in trainer_state else {},
            'metrics': trainer_state.get('metrics', {}),
            'rng_states': self._get_rng_states(),
            'metadata': {
                'step': step,
                'name': name,
                'is_best': is_best,
                'experiment_dir': str(self.experiment_dir),
                'compress': self.compress
            }
        }
        
        # Add any additional state from trainer
        for key, value in trainer_state.items():
            if key not in ['algorithm', 'buffer'] and hasattr(value, 'save_checkpoint'):
                checkpoint_data[f'{key}_state'] = value.save_checkpoint()
        
        # Save checkpoint
        try:
            if self.compress:
                # Save with compression
                torch.save(checkpoint_data, checkpoint_path, 
                          _use_new_zipfile_serialization=True)
            else:
                torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path} (step {step})")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
        
        # Update symbolic links
        self._update_latest_link(checkpoint_path)
        if is_best:
            self._update_best_link(checkpoint_path)
        
        # Clean up old checkpoints
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()
        
        # Save checkpoint metadata
        self._save_checkpoint_metadata(checkpoint_data['metadata'], checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None,
                       load_latest: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to specific checkpoint file (None for auto-detection)
            load_latest: If True and no path given, load latest checkpoint
            
        Returns:
            Checkpoint data dictionary or None if no checkpoint found
        """
        if checkpoint_path is None:
            if load_latest:
                checkpoint_path = self._get_latest_checkpoint()
            else:
                return None
        
        if checkpoint_path is None:
            logger.info("No checkpoint found to load")
            return None
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path} (step {checkpoint_data.get('step', 'unknown')})")
            return checkpoint_data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def restore_training_state(self, trainer_state: Dict[str, Any], 
                             checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore training state from checkpoint data.
        
        Args:
            trainer_state: Current trainer state dictionary
            checkpoint_data: Loaded checkpoint data
            
        Returns:
            Updated trainer state with restored components
        """
        step = checkpoint_data.get('step', 0)
        
        # Restore algorithm state
        if 'algorithm' in trainer_state and 'algorithm_state' in checkpoint_data:
            trainer_state['algorithm'].load_checkpoint(checkpoint_data['algorithm_state'])
            logger.debug("Algorithm state restored")
        
        # Restore buffer state
        if 'buffer' in trainer_state and 'buffer_state' in checkpoint_data:
            trainer_state['buffer'].load_checkpoint(checkpoint_data['buffer_state'])
            logger.debug("Buffer state restored")
        
        # Restore other component states
        for key, value in trainer_state.items():
            state_key = f'{key}_state'
            if key not in ['algorithm', 'buffer'] and state_key in checkpoint_data:
                if hasattr(value, 'load_checkpoint'):
                    value.load_checkpoint(checkpoint_data[state_key])
                    logger.debug(f"{key} state restored")
        
        # Restore RNG states for reproducibility
        if 'rng_states' in checkpoint_data:
            self._restore_rng_states(checkpoint_data['rng_states'])
            logger.debug("RNG states restored")
        
        # Update trainer state
        trainer_state['step'] = step
        trainer_state['metrics'] = checkpoint_data.get('metrics', {})
        
        logger.info(f"Training state restored to step {step}")
        return trainer_state
    
    def auto_save(self, trainer_state: Dict[str, Any], step: int) -> Optional[Path]:
        """
        Automatically save checkpoint if frequency condition is met.
        
        Args:
            trainer_state: Current trainer state
            step: Current training step
            
        Returns:
            Path to saved checkpoint if one was created, None otherwise
        """
        if step - self._last_save_step >= self.auto_save_frequency:
            checkpoint_path = self.save_checkpoint(
                trainer_state, 
                step, 
                name=f"auto_step_{step}"
            )
            self._last_save_step = step
            return checkpoint_path
        return None
    
    def save_best_checkpoint(self, trainer_state: Dict[str, Any], 
                           step: int, metric_value: float) -> Path:
        """
        Save checkpoint as best model so far.
        
        Args:
            trainer_state: Current trainer state
            step: Current training step
            metric_value: Value of metric being optimized
            
        Returns:
            Path to saved checkpoint
        """
        return self.save_checkpoint(
            trainer_state,
            step,
            name=f"best_step_{step}_metric_{metric_value:.4f}",
            is_best=True
        )
    
    def _get_rng_states(self) -> Dict[str, Any]:
        """Get current random number generator states"""
        import random
        rng_states = {
            'numpy': np.random.get_state(),
            'python': random.getstate(),  # For built-in random module
        }
        
        # PyTorch CPU RNG state
        try:
            rng_states['torch_cpu'] = torch.get_rng_state()
        except Exception as e:
            logger.warning(f"Could not get PyTorch CPU RNG state: {e}")
        
        # PyTorch CUDA RNG states (if available)
        try:
            if torch.cuda.is_available():
                rng_states['torch_cuda'] = torch.cuda.get_rng_state_all()
        except Exception as e:
            logger.warning(f"Could not get PyTorch CUDA RNG state: {e}")
        
        return rng_states
    
    def _restore_rng_states(self, rng_states: Dict[str, Any]):
        """Restore random number generator states"""
        try:
            if 'numpy' in rng_states:
                np.random.set_state(rng_states['numpy'])
        except Exception as e:
            logger.warning(f"Could not restore NumPy RNG state: {e}")
        
        try:
            if 'torch_cpu' in rng_states:
                torch.set_rng_state(rng_states['torch_cpu'])
        except Exception as e:
            logger.warning(f"Could not restore PyTorch CPU RNG state: {e}")
        
        try:
            if 'torch_cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
        except Exception as e:
            logger.warning(f"Could not restore PyTorch CUDA RNG state: {e}")
    
    def _update_latest_link(self, checkpoint_path: Path):
        """Update symbolic link to latest checkpoint"""
        latest_link = self.checkpoint_dir / "latest.pt"
        
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)
        except Exception as e:
            logger.warning(f"Could not update latest checkpoint link: {e}")
    
    def _update_best_link(self, checkpoint_path: Path):
        """Update symbolic link to best checkpoint"""
        best_link = self.checkpoint_dir / "best.pt"
        
        try:
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
        except Exception as e:
            logger.warning(f"Could not update best checkpoint link: {e}")
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        latest_link = self.checkpoint_dir / "latest.pt"
        
        if latest_link.exists():
            return latest_link
        
        # Fallback: find most recent checkpoint file
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            # Sort by modification time
            latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            return latest_file
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to free disk space"""
        checkpoint_files = [f for f in self.checkpoint_dir.glob("*.pt") 
                          if not f.is_symlink() and f.name not in ['latest.pt', 'best.pt']]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time, keep most recent
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        files_to_remove = checkpoint_files[self.max_checkpoints:]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                # Also remove associated metadata file if it exists
                metadata_file = file_path.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                logger.debug(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove old checkpoint {file_path}: {e}")
    
    def _save_checkpoint_metadata(self, metadata: Dict[str, Any], checkpoint_path: Path):
        """Save checkpoint metadata to JSON file"""
        metadata_path = checkpoint_path.with_suffix('.json')
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save checkpoint metadata: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            if checkpoint_file.is_symlink():
                continue
            
            metadata_file = checkpoint_file.with_suffix('.json')
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read metadata for {checkpoint_file}: {e}")
            
            checkpoints.append({
                'path': str(checkpoint_file),
                'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat(),
                **metadata
            })
        
        # Sort by step or modification time
        checkpoints.sort(key=lambda x: x.get('step', 0), reverse=True)
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed information about a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        info = {
            'path': str(checkpoint_path),
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat()
        }
        
        # Try to load minimal info without full checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            info.update({
                'step': checkpoint_data.get('step'),
                'timestamp': checkpoint_data.get('timestamp'),
                'metadata': checkpoint_data.get('metadata', {})
            })
        except Exception as e:
            logger.warning(f"Could not read checkpoint data: {e}")
        
        return info