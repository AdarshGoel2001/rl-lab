#!/usr/bin/env python3
"""
Comprehensive test suite for vectorized environment implementation
This script tests all aspects of the vectorized pipeline to identify and debug issues.
"""

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.environments.vectorized_gym_wrapper import VectorizedGymWrapper
from src.buffers.trajectory import TrajectoryBuffer  
from src.algorithms.ppo_legacy import PPOAlgorithm
from src.utils.registry import auto_import_modules

# Import modules to register components
auto_import_modules()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorizedEnvironmentTester:
    def __init__(self):
        self.test_results = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"   → {message}")
        print()
    
    def test_1_environment_creation(self):
        """Test basic vectorized environment creation"""
        test_name = "Vectorized Environment Creation"
        
        try:
            config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            
            env = VectorizedGymWrapper(config)
            
            # Check basic properties
            assert env.num_envs == 4
            assert env.is_vectorized == True
            assert hasattr(env, 'vec_env')
            
            # Check space specifications
            obs_space = env.observation_space
            action_space = env.action_space
            
            expected_obs_shape = (4, 4)  # (num_envs, CartPole obs_dim)
            expected_action_shape = (4, 2)  # (num_envs, CartPole action_dim)
            
            assert obs_space.shape == expected_obs_shape, f"Expected obs shape {expected_obs_shape}, got {obs_space.shape}"
            assert action_space.shape == expected_action_shape, f"Expected action shape {expected_action_shape}, got {action_space.shape}"
            
            env.close()
            self.log_test(test_name, True, f"Created {env.num_envs} envs, obs_shape={obs_space.shape}, action_shape={action_space.shape}")
            return env
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return None
    
    def test_2_environment_reset(self):
        """Test environment reset functionality"""
        test_name = "Environment Reset"
        
        try:
            config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            
            env = VectorizedGymWrapper(config)
            
            # Test reset
            obs = env.reset(seed=42)
            
            # Check output types and shapes
            assert isinstance(obs, torch.Tensor), f"Expected torch.Tensor, got {type(obs)}"
            expected_shape = (4, 4)  # (num_envs, obs_dim)
            assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
            assert obs.dtype == torch.float32, f"Expected float32, got {obs.dtype}"
            
            # Test that observations are different (environments should be independent)
            obs_numpy = obs.numpy()
            all_same = np.allclose(obs_numpy[0], obs_numpy[1]) and np.allclose(obs_numpy[1], obs_numpy[2])
            
            env.close()
            self.log_test(test_name, True, f"Reset shape: {obs.shape}, dtype: {obs.dtype}, diverse: {not all_same}")
            return True
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return False
    
    def test_3_environment_step(self):
        """Test environment step functionality"""
        test_name = "Environment Step"
        
        try:
            config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            
            env = VectorizedGymWrapper(config)
            obs = env.reset(seed=42)
            
            # Test random actions
            actions = torch.randint(0, 2, (4,))  # Random actions for each env
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Check output types and shapes
            assert isinstance(next_obs, torch.Tensor), f"Expected torch.Tensor obs, got {type(next_obs)}"
            assert isinstance(rewards, torch.Tensor), f"Expected torch.Tensor rewards, got {type(rewards)}"
            assert isinstance(dones, torch.Tensor), f"Expected torch.Tensor dones, got {type(dones)}"
            assert isinstance(infos, list), f"Expected list infos, got {type(infos)}"
            
            # Check shapes
            assert next_obs.shape == (4, 4), f"Expected obs shape (4, 4), got {next_obs.shape}"
            assert rewards.shape == (4,), f"Expected rewards shape (4,), got {rewards.shape}"
            assert dones.shape == (4,), f"Expected dones shape (4,), got {dones.shape}"
            assert len(infos) == 4, f"Expected 4 info dicts, got {len(infos)}"
            
            # Check dtypes
            assert next_obs.dtype == torch.float32, f"Expected float32 obs, got {next_obs.dtype}"
            assert rewards.dtype == torch.float32, f"Expected float32 rewards, got {rewards.dtype}"
            assert dones.dtype == torch.bool, f"Expected bool dones, got {dones.dtype}"
            
            # Check that all infos are dicts
            for i, info in enumerate(infos):
                assert isinstance(info, dict), f"Info[{i}] is not a dict: {type(info)}"
            
            env.close()
            self.log_test(test_name, True, f"Step shapes: obs{next_obs.shape}, rewards{rewards.shape}, dones{dones.shape}, infos={len(infos)}")
            return True
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return False
    
    def test_4_data_collection_simulation(self):
        """Test data collection like the trainer does"""
        test_name = "Data Collection Simulation"
        
        try:
            config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            
            env = VectorizedGymWrapper(config)
            obs = env.reset(seed=42)
            
            # Simulate trajectory collection (like trainer.py does)
            trajectory_length = 10
            observations = []
            actions = []
            rewards = []
            dones = []
            old_values = []
            old_log_probs = []
            
            current_obs = obs
            
            for step in range(trajectory_length):
                # Random actions and dummy values/log_probs
                actions_step = torch.randint(0, 2, (4,))
                values_step = torch.randn(4)
                log_probs_step = torch.randn(4)
                
                # Store current data
                observations.append(current_obs.cpu().numpy())
                actions.append(actions_step.cpu().numpy())
                old_values.append(values_step.cpu().numpy())
                old_log_probs.append(log_probs_step.cpu().numpy())
                
                # Step environment
                next_obs, rewards_step, dones_step, infos = env.step(actions_step)
                
                # Store step results
                rewards.append(rewards_step.cpu().numpy())
                dones.append(dones_step.cpu().numpy())
                
                current_obs = next_obs
            
            # Convert to numpy arrays (like trainer does)
            trajectory = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'dones': np.array(dones),
                'old_values': np.array(old_values),
                'old_log_probs': np.array(old_log_probs)
            }
            
            # Check shapes
            expected_shapes = {
                'observations': (trajectory_length, 4, 4),  # (T, B, obs_dim)
                'actions': (trajectory_length, 4),          # (T, B)
                'rewards': (trajectory_length, 4),          # (T, B)
                'dones': (trajectory_length, 4),            # (T, B)
                'old_values': (trajectory_length, 4),       # (T, B)
                'old_log_probs': (trajectory_length, 4),    # (T, B)
            }
            
            shape_errors = []
            for key, expected_shape in expected_shapes.items():
                actual_shape = trajectory[key].shape
                if actual_shape != expected_shape:
                    shape_errors.append(f"{key}: expected {expected_shape}, got {actual_shape}")
            
            if shape_errors:
                error_msg = "; ".join(shape_errors)
                env.close()
                self.log_test(test_name, False, error_msg)
                return None
            
            env.close()
            self.log_test(test_name, True, f"Trajectory shapes correct, total experiences: {trajectory_length * 4}")
            return trajectory
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return None
    
    def test_5_buffer_integration(self, trajectory=None):
        """Test buffer integration with vectorized data"""
        test_name = "Buffer Integration"
        
        try:
            # Use provided trajectory or create a new one
            if trajectory is None:
                trajectory = self.test_4_data_collection_simulation()
                if trajectory is None:
                    self.log_test(test_name, False, "Could not create trajectory for buffer test")
                    return False
            
            # Create buffer
            buffer_config = {
                'capacity': 2048,
                'observation_shape': (4,),  # Single env obs shape
                'action_shape': (),         # Scalar action
                'device': 'cpu'
            }
            
            buffer = TrajectoryBuffer(buffer_config)
            
            # Add trajectory to buffer
            print(f"Adding trajectory with shapes: {[(k, v.shape) for k, v in trajectory.items()]}")
            buffer.add_trajectory(trajectory)
            
            # Check buffer size
            expected_size = trajectory['observations'].shape[0] * trajectory['observations'].shape[1]  # T * B
            actual_size = buffer.size
            
            assert actual_size == expected_size, f"Expected buffer size {expected_size}, got {actual_size}"
            
            # Test sampling
            sample_size = 32
            batch = buffer.sample(sample_size)
            
            # Check batch structure
            required_keys = ['observations', 'actions', 'rewards', 'dones', 'old_values', 'old_log_probs']
            missing_keys = [key for key in required_keys if key not in batch]
            
            if missing_keys:
                self.log_test(test_name, False, f"Missing batch keys: {missing_keys}")
                return False
            
            # Check batch shapes
            expected_batch_shapes = {
                'observations': (sample_size, 4),  # (batch_size, obs_dim)
                'actions': (sample_size,),         # (batch_size,)
                'rewards': (sample_size,),         # (batch_size,)
                'dones': (sample_size,),           # (batch_size,)
                'old_values': (sample_size,),      # (batch_size,)
                'old_log_probs': (sample_size,),   # (batch_size,)
            }
            
            batch_shape_errors = []
            for key, expected_shape in expected_batch_shapes.items():
                actual_shape = batch[key].shape
                if actual_shape != expected_shape:
                    batch_shape_errors.append(f"{key}: expected {expected_shape}, got {actual_shape}")
            
            if batch_shape_errors:
                error_msg = "; ".join(batch_shape_errors)
                self.log_test(test_name, False, error_msg)
                return False
            
            self.log_test(test_name, True, f"Buffer size: {actual_size}, batch shapes correct")
            return True
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return False
    
    def test_6_algorithm_integration(self):
        """Test PPO algorithm with vectorized data"""
        test_name = "Algorithm Integration"
        
        try:
            # Create environment
            env_config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            
            env = VectorizedGymWrapper(env_config)
            
            # Create PPO algorithm
            ppo_config = {
                'obs_dim': 4,  # Single env obs shape
                'action_dim': 2,  # Number of discrete actions
                'action_space_type': 'discrete',
                'lr': 3e-4,
                'device': 'cpu',
                'network': {
                    'actor': {
                        'type': 'mlp',
                        'hidden_dims': [64, 64],
                        'activation': 'tanh'
                    },
                    'critic': {
                        'type': 'mlp',
                        'hidden_dims': [64, 64],
                        'activation': 'tanh'
                    }
                },
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5
            }
            
            ppo = PPOAlgorithm(ppo_config)
            
            # Test action selection
            obs = env.reset(seed=42)
            actions, log_probs, values = ppo.get_action_and_value(obs)
            
            # Check action outputs
            assert actions.shape == (4,), f"Expected actions shape (4,), got {actions.shape}"
            assert log_probs.shape == (4,), f"Expected log_probs shape (4,), got {log_probs.shape}"
            assert values.shape == (4,), f"Expected values shape (4,), got {values.shape}"
            
            # Create dummy batch for update test
            batch = {
                'observations': torch.randn(32, 4),
                'actions': torch.randint(0, 2, (32,)),
                'rewards': torch.randn(32),
                'dones': torch.randint(0, 2, (32,)).bool(),
                'old_values': torch.randn(32),
                'old_log_probs': torch.randn(32),
                'advantages': torch.randn(32) * 0.5 + 0.1,  # Ensure non-zero std
                'returns': torch.randn(32),
            }
            
            # Test algorithm update
            metrics = ppo.update(batch)
            
            # Check that metrics are returned
            assert isinstance(metrics, dict), f"Expected dict metrics, got {type(metrics)}"
            assert 'loss' in metrics or 'total_loss' in metrics, f"No loss in metrics: {list(metrics.keys())}"
            
            env.close()
            self.log_test(test_name, True, f"Action shapes correct, update metrics: {list(metrics.keys())}")
            return True
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return False
    
    def test_7_full_pipeline(self):
        """Test complete training pipeline with vectorized environment"""
        test_name = "Full Pipeline Integration"
        
        try:
            # Create environment
            env_config = {
                'name': 'CartPole-v1',
                'num_envs': 4,
                'vectorization': 'sync',
                'normalize_obs': False,
                'normalize_reward': False,
            }
            env = VectorizedGymWrapper(env_config)
            
            # Create buffer
            buffer_config = {
                'capacity': 2048,
                'observation_shape': (4,),
                'action_shape': (),
                'device': 'cpu'
            }
            buffer = TrajectoryBuffer(buffer_config)
            
            # Create algorithm
            ppo_config = {
                'obs_dim': 4,
                'action_dim': 2,
                'action_space_type': 'discrete',
                'lr': 3e-4,
                'device': 'cpu',
                'network': {
                    'actor': {
                        'type': 'mlp',
                        'hidden_dims': [64, 64],
                        'activation': 'tanh'
                    },
                    'critic': {
                        'type': 'mlp',
                        'hidden_dims': [64, 64], 
                        'activation': 'tanh'
                    }
                },
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5
            }
            ppo = PPOAlgorithm(ppo_config)
            
            # Simulate short training episode
            obs = env.reset(seed=42)
            trajectory_length = 20
            
            observations = []
            actions_list = []
            rewards_list = []
            dones_list = []
            old_values_list = []
            old_log_probs_list = []
            
            for step in range(trajectory_length):
                # Get actions from PPO
                actions, log_probs, values = ppo.get_action_and_value(obs)
                
                # Store data
                observations.append(obs.detach().cpu().numpy())
                actions_list.append(actions.detach().cpu().numpy())
                old_values_list.append(values.detach().cpu().numpy())
                old_log_probs_list.append(log_probs.detach().cpu().numpy())
                
                # Step environment
                next_obs, rewards, dones, infos = env.step(actions)
                
                # Store step results
                rewards_list.append(rewards.detach().cpu().numpy())
                dones_list.append(dones.detach().cpu().numpy())
                
                obs = next_obs
            
            # Create trajectory
            trajectory = {
                'observations': np.array(observations),
                'actions': np.array(actions_list),
                'rewards': np.array(rewards_list),
                'dones': np.array(dones_list),
                'old_values': np.array(old_values_list),
                'old_log_probs': np.array(old_log_probs_list)
            }
            
            # Add to buffer and sample
            buffer.add_trajectory(trajectory)
            batch = buffer.sample(64)
            
            # Update algorithm
            metrics = ppo.update(batch)
            
            env.close()
            self.log_test(test_name, True, f"Complete pipeline working, buffer size: {buffer.size}, metrics: {list(metrics.keys())}")
            return True
            
        except Exception as e:
            self.log_test(test_name, False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("="*60)
        print("COMPREHENSIVE VECTORIZED ENVIRONMENT TEST SUITE")
        print("="*60)
        print()
        
        # Run tests in sequence, passing data between them when needed
        env = self.test_1_environment_creation()
        self.test_2_environment_reset()
        self.test_3_environment_step()
        trajectory = self.test_4_data_collection_simulation()
        self.test_5_buffer_integration(trajectory)
        self.test_6_algorithm_integration()
        self.test_7_full_pipeline()
        
        # Summary
        print("="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"Passed: {passed}/{total} tests")
        print()
        
        if passed < total:
            print("FAILED TESTS:")
            for test_name, success, message in self.test_results:
                if not success:
                    print(f"❌ {test_name}: {message}")
        else:
            print("🎉 ALL TESTS PASSED!")
        
        print()
        return passed == total

def main():
    """Main test runner"""
    tester = VectorizedEnvironmentTester()
    success = tester.run_all_tests()
    
    if not success:
        print("Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("All tests passed! The vectorized environment implementation appears to be working correctly.")
        sys.exit(0)

if __name__ == "__main__":
    main()