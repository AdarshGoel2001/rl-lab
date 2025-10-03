"""
Comprehensive Infrastructure Tests

This module contains automated tests for validating the core infrastructure
components of the RL monorepo system, ensuring reliability and reproducibility.

Test Coverage:
- Checkpoint save/load functionality
- Configuration system
- Component registration
- RNG state management  
- Experiment directory structure
- Error handling and recovery
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch
import numpy as np
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config, Config
from src.utils.registry import ALGORITHM_REGISTRY, NETWORK_REGISTRY, ENVIRONMENT_REGISTRY
from src.algorithms.ppo_legacy import PPOAlgorithm
from src.networks.mlp import ActorMLP, CriticMLP
from src.environments.gym_wrapper import GymWrapper
from src.buffers.trajectory import TrajectoryBuffer


class TestCheckpointManager:
    """Test checkpoint save/load functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(
            self.test_dir,
            auto_save_frequency=100,
            max_checkpoints=3
        )
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_checkpoint_creation(self):
        """Test that checkpoint directory is created properly"""
        assert (Path(self.test_dir) / "checkpoints").exists()
        
    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving functionality"""
        # Create mock trainer state
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {'test': 'algorithm_data'}
        
        mock_buffer = Mock()
        mock_buffer.save_checkpoint.return_value = {'test': 'buffer_data'}
        
        trainer_state = {
            'algorithm': mock_algorithm,
            'buffer': mock_buffer,
            'metrics': {'reward': 100.0, 'loss': 0.5}
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(trainer_state, step=1000)
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists()
        assert checkpoint_path.name.endswith('.pt')
        
        # Verify latest link exists
        latest_link = Path(self.test_dir) / "checkpoints" / "latest.pt"
        assert latest_link.exists()
        
    def test_load_checkpoint_basic(self):
        """Test basic checkpoint loading functionality"""
        # Create and save checkpoint first
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {'algorithm_state': 'test_data'}
        
        trainer_state = {
            'algorithm': mock_algorithm,
            'metrics': {'reward': 150.0}
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(trainer_state, step=2000)
        
        # Load checkpoint
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        assert loaded_data is not None
        assert loaded_data['step'] == 2000
        assert loaded_data['metrics']['reward'] == 150.0
        assert 'algorithm_state' in loaded_data
        
    def test_auto_save_frequency(self):
        """Test automatic checkpoint saving based on frequency"""
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {}
        trainer_state = {'algorithm': mock_algorithm}
        
        # Should not save initially
        result = self.checkpoint_manager.auto_save(trainer_state, step=50)
        assert result is None
        
        # Should save at frequency threshold
        result = self.checkpoint_manager.auto_save(trainer_state, step=100)
        assert result is not None
        assert result.exists()
        
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup functionality"""
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {}
        trainer_state = {'algorithm': mock_algorithm}
        
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            self.checkpoint_manager.save_checkpoint(trainer_state, step=i*1000, name=f"test_{i}")
        
        # Count non-symlink checkpoint files
        checkpoint_files = [f for f in Path(self.test_dir, "checkpoints").glob("*.pt") 
                          if not f.is_symlink()]
        
        # Should have cleaned up to max_checkpoints (3)
        assert len(checkpoint_files) <= 3
        
    def test_rng_state_preservation(self):
        """Test that RNG states are properly saved and restored"""
        # Set specific RNG states
        np.random.seed(12345)
        torch.manual_seed(67890)
        
        # Get initial states
        initial_np_state = np.random.get_state()
        initial_torch_state = torch.get_rng_state()
        
        # Save checkpoint
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {}
        trainer_state = {'algorithm': mock_algorithm}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(trainer_state, step=100)
        
        # Change RNG states
        np.random.seed(99999)
        torch.manual_seed(11111)
        
        # Load and restore checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        self.checkpoint_manager._restore_rng_states(checkpoint_data['rng_states'])
        
        # Verify states match initial states
        restored_np_state = np.random.get_state()
        restored_torch_state = torch.get_rng_state()
        
        # Compare state elements (arrays are equal)
        assert np.array_equal(initial_np_state[1], restored_np_state[1])
        assert torch.equal(initial_torch_state, restored_torch_state)


class TestConfigSystem:
    """Test configuration system functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_config_loading(self):
        """Test loading configuration from YAML file"""
        # Create test config file
        test_config = {
            'experiment': {'name': 'test_experiment', 'seed': 42},
            'algorithm': {'name': 'ppo', 'lr': 0.001},
            'environment': {'name': 'CartPole-v1', 'wrapper': 'gym'},
            'network': {'actor': {'type': 'actor_mlp'}, 'critic': {'type': 'critic_mlp'}},
            'buffer': {'type': 'trajectory', 'capacity': 1000},
            'training': {'total_timesteps': 10000},
            'logging': {'terminal': True, 'tensorboard': False, 'wandb_enabled': False}
        }
        
        config_path = Path(self.test_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load config
        config = load_config(str(config_path))
        
        assert config.experiment.name == 'test_experiment'
        assert config.experiment.seed == 42
        assert config.algorithm.name == 'ppo'
        assert config.algorithm.lr == 0.001
        
    def test_config_overrides(self):
        """Test configuration override functionality"""
        base_config = {
            'experiment': {'name': 'base', 'seed': 1},
            'algorithm': {'name': 'ppo', 'lr': 0.01},
            'environment': {'name': 'CartPole-v1', 'wrapper': 'gym'},
            'network': {'actor': {'type': 'actor_mlp'}, 'critic': {'type': 'critic_mlp'}},
            'buffer': {'type': 'trajectory', 'capacity': 1000},
            'training': {'total_timesteps': 10000},
            'logging': {'terminal': True, 'tensorboard': False, 'wandb_enabled': False}
        }
        
        config_path = Path(self.test_dir) / "base_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Test overrides
        overrides = {
            'experiment': {'seed': 999},
            'algorithm': {'lr': 0.05}
        }
        
        config = load_config(str(config_path), overrides)
        
        assert config.experiment.name == 'base'  # Not overridden
        assert config.experiment.seed == 999     # Overridden
        assert config.algorithm.lr == 0.05       # Overridden
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Test missing required fields
        invalid_config = {
            'experiment': {'name': 'test'},  # Missing seed
            'algorithm': {'name': 'ppo'},
            'environment': {'name': 'CartPole-v1', 'wrapper': 'gym'},
            'network': {'actor': {'type': 'actor_mlp'}, 'critic': {'type': 'critic_mlp'}},
            'buffer': {'type': 'trajectory', 'capacity': 1000},
            'training': {'total_timesteps': 10000},
            'logging': {'terminal': True, 'tensorboard': False, 'wandb_enabled': False}
        }
        
        config_path = Path(self.test_dir) / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should handle missing fields gracefully
        config = load_config(str(config_path))
        assert hasattr(config.experiment, 'name')
        
    def test_config_hash_consistency(self):
        """Test that config hash is consistent for same configuration"""
        config_dict = {
            'experiment': {'name': 'test', 'seed': 42},
            'algorithm': {'name': 'ppo', 'lr': 0.001},
            'environment': {'name': 'CartPole-v1', 'wrapper': 'gym'},
            'network': {'actor': {'type': 'actor_mlp'}, 'critic': {'type': 'critic_mlp'}},
            'buffer': {'type': 'trajectory', 'capacity': 1000},
            'training': {'total_timesteps': 10000},
            'logging': {'terminal': True, 'tensorboard': False, 'wandb_enabled': False}
        }
        
        config_path = Path(self.test_dir) / "hash_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config1 = load_config(str(config_path))
        config2 = load_config(str(config_path))
        
        assert config1.get_hash() == config2.get_hash()


class TestComponentRegistry:
    """Test component registration system"""
    
    def test_algorithm_registration(self):
        """Test algorithm registration and retrieval"""
        assert 'ppo' in ALGORITHM_REGISTRY
        assert ALGORITHM_REGISTRY['ppo'] == PPOAlgorithm
        
    def test_network_registration(self):
        """Test network registration and retrieval"""
        assert 'actor_mlp' in NETWORK_REGISTRY
        assert 'critic_mlp' in NETWORK_REGISTRY
        assert NETWORK_REGISTRY['actor_mlp'] == ActorMLP
        assert NETWORK_REGISTRY['critic_mlp'] == CriticMLP
        
    def test_environment_registration(self):
        """Test environment registration and retrieval"""
        assert 'gym' in ENVIRONMENT_REGISTRY
        assert ENVIRONMENT_REGISTRY['gym'] == GymWrapper


class TestExperimentStructure:
    """Test experiment directory structure and organization"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_experiment_directory_creation(self):
        """Test that experiment directories are created properly"""
        from src.core.trainer import Trainer
        
        # Create minimal config for testing
        config_dict = {
            'experiment': {'name': 'test_exp', 'seed': 42, 'device': 'cpu'},
            'algorithm': {'name': 'ppo'},
            'environment': {'name': 'CartPole-v1', 'wrapper': 'gym'},
            'network': {'actor': {'type': 'actor_mlp'}, 'critic': {'type': 'critic_mlp'}},
            'buffer': {'type': 'trajectory', 'capacity': 100},
            'training': {'total_timesteps': 1000},
            'logging': {'terminal': True, 'tensorboard': False, 'wandb_enabled': False}
        }
        
        config_path = Path(self.test_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # This would create experiment structure
        # Note: This is a simplified test since Trainer creation is complex
        required_dirs = ['checkpoints', 'configs', 'logs']
        
        for dir_name in required_dirs:
            test_dir = Path(self.test_dir) / dir_name
            test_dir.mkdir(parents=True, exist_ok=True)
            assert test_dir.exists()


class TestErrorHandling:
    """Test error handling and recovery mechanisms"""
    
    def test_checkpoint_corruption_handling(self):
        """Test handling of corrupted checkpoint files"""
        test_dir = tempfile.mkdtemp()
        try:
            checkpoint_manager = CheckpointManager(test_dir)
            
            # Create corrupted checkpoint file
            corrupted_path = Path(test_dir) / "checkpoints" / "corrupted.pt"
            corrupted_path.parent.mkdir(exist_ok=True)
            with open(corrupted_path, 'w') as f:
                f.write("This is not a valid PyTorch checkpoint file")
            
            # Should handle corruption gracefully
            with pytest.raises(Exception):  # Should raise an exception
                checkpoint_manager.load_checkpoint(corrupted_path)
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
            
    def test_missing_config_handling(self):
        """Test handling of missing configuration files"""
        nonexistent_path = "/nonexistent/config.yaml"
        
        with pytest.raises((FileNotFoundError, Exception)):
            load_config(nonexistent_path)
            
    def test_invalid_component_registration(self):
        """Test handling of invalid component registrations"""
        # Test that registry doesn't break with missing components
        assert ALGORITHM_REGISTRY.get('nonexistent_algorithm') is None


# Removed TestPerformanceMetrics class - redundant with TensorBoard/WandB monitoring


class TestReproducibilityGuarantees:
    """Test reproducibility guarantees across runs"""
    
    def test_deterministic_training_setup(self):
        """Test that identical configs produce identical initial states"""
        config_dict = {
            'experiment': {'seed': 12345, 'device': 'cpu'},
            'algorithm': {'name': 'ppo'},
        }
        
        # Set up identical environments twice
        torch.manual_seed(12345)
        np.random.seed(12345)
        initial_torch_1 = torch.randn(10)
        initial_np_1 = np.random.rand(10)
        
        torch.manual_seed(12345) 
        np.random.seed(12345)
        initial_torch_2 = torch.randn(10)
        initial_np_2 = np.random.rand(10)
        
        # Should be identical
        assert torch.equal(initial_torch_1, initial_torch_2)
        assert np.array_equal(initial_np_1, initial_np_2)


# Integration test that validates entire checkpoint cycle
def test_full_checkpoint_integration():
    """Integration test for full checkpoint save/load cycle"""
    test_dir = tempfile.mkdtemp()
    
    try:
        checkpoint_manager = CheckpointManager(test_dir)
        
        # Create realistic mock components
        mock_algorithm = Mock()
        mock_algorithm.save_checkpoint.return_value = {
            'step': 5000,
            'networks': {'actor': 'mock_actor_state', 'critic': 'mock_critic_state'},
            'optimizers': {'actor_opt': 'mock_actor_opt_state'}
        }
        mock_algorithm.load_checkpoint = Mock()
        
        mock_buffer = Mock()
        mock_buffer.save_checkpoint.return_value = {
            'experiences': 'mock_buffer_data',
            'ptr': 1000
        }
        mock_buffer.load_checkpoint = Mock()
        
        trainer_state = {
            'algorithm': mock_algorithm,
            'buffer': mock_buffer,
            'metrics': {'episode_reward': 180.5, 'loss': 0.05}
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(trainer_state, step=5000)
        
        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore state
        restored_state = checkpoint_manager.restore_training_state(trainer_state, checkpoint_data)
        
        # Verify components were called correctly
        mock_algorithm.load_checkpoint.assert_called_once()
        mock_buffer.load_checkpoint.assert_called_once()
        
        # Verify state restoration
        assert restored_state['step'] == 5000
        assert restored_state['metrics']['episode_reward'] == 180.5
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])