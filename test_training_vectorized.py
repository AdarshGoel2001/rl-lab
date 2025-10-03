#!/usr/bin/env python3
"""
Test script to verify that the entire training pipeline works with vectorized environments.
This tests the complete integration from trainer -> vectorized env -> buffer -> algorithm.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_training_pipeline():
    """Test complete training pipeline with vectorized environments"""
    print("Testing complete training pipeline with vectorized environments...")
    
    try:
        # Set up config like a real training run
        from src.utils.config import Config
        from src.core.trainer import Trainer
        
        # Create a minimal config for testing
        config_dict = {
            'experiment': {
                'name': 'test_vectorized_pipeline',
                'device': 'cpu'
            },
            'environment': {
                'name': 'CartPole-v1',
                'wrapper': 'vectorized_gym',
                'num_envs': 4,
                'vectorization': 'sync'
            },
            'algorithm': {
                'name': 'ppo',
                'lr': 3e-4,
                'gamma': 0.99,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01
            },
            'buffer': {
                'type': 'trajectory',
                'capacity': 512,  # Small for testing
                'batch_size': 64
            },
            'network': {
                'actor': {
                    'type': 'mlp',
                    'hidden_dims': [32, 32]
                },
                'critic': {
                    'type': 'mlp', 
                    'hidden_dims': [32, 32]
                }
            },
            'training': {
                'total_timesteps': 1000,  # Short training for test
                'eval_frequency': 500,
                'checkpoint_frequency': 1000
            },
            'logging': {
                'terminal': True,
                'tensorboard': False,
                'wandb': {'enabled': False}
            }
        }
        
        config = Config(config_dict)
        
        # Create trainer
        print("Creating trainer with vectorized environment...")
        trainer = Trainer(config)
        
        print(f"Environment type: {type(trainer.environment).__name__}")
        print(f"Buffer type: {type(trainer.buffer).__name__}")
        print(f"Algorithm type: {type(trainer.algorithm).__name__}")
        print(f"Environment num_envs: {getattr(trainer.environment, 'num_envs', 'N/A')}")
        print(f"Buffer capacity: {trainer.buffer.capacity}")
        
        # Test single training step
        print("\nTesting single training step...")
        initial_step = trainer.step
        
        # Run a few training steps
        for i in range(3):
            print(f"\nTraining step {i+1}:")
            try:
                trainer._training_step()
                print(f"  Step count: {trainer.step}")
                print(f"  Buffer size: {trainer.buffer._size}")
                print(f"  Buffer ready: {trainer.buffer.ready()}")
                
                # If buffer is ready, it should have done an update
                if trainer.buffer.ready():
                    print("  ✅ Buffer was ready and algorithm update occurred")
                else:
                    print("  ⏳ Buffer not ready yet, collecting more data")
                    
            except Exception as e:
                print(f"  ❌ Error in training step: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\nFinal training state:")
        print(f"- Total steps: {trainer.step}")
        print(f"- Buffer size: {trainer.buffer._size}")
        print(f"- Episodes completed: {trainer.total_episodes}")
        
        # Clean up
        trainer.environment.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_integration():
    """Test buffer integration specifically"""
    print("\n" + "="*60)
    print("Testing buffer integration with vectorized trainer data...")
    
    try:
        from src.environments.vectorized_gym_wrapper import VectorizedGymWrapper
        from src.algorithms.ppo_legacy import PPO
        from src.buffers.trajectory import TrajectoryBuffer
        from src.networks.mlp import MLP
        import torch.nn as nn
        
        # Set up components
        env_config = {
            'name': 'CartPole-v1',
            'num_envs': 2,
            'vectorization': 'sync'
        }
        env = VectorizedGymWrapper(env_config)
        
        buffer_config = {
            'capacity': 256,
            'batch_size': 32
        }
        buffer = TrajectoryBuffer(buffer_config)
        
        # Create simple networks for PPO
        obs_dim = env.observation_space.shape[-1]  # Remove batch dimension
        action_dim = env.action_space.n if env.action_space.discrete else env.action_space.shape[-1]
        
        networks = {
            'actor': MLP({
                'input_dim': obs_dim,
                'hidden_dims': [16, 16],
                'output_dim': action_dim,
                'activation': 'relu'
            }),
            'critic': MLP({
                'input_dim': obs_dim,
                'hidden_dims': [16, 16], 
                'output_dim': 1,
                'activation': 'relu'
            })
        }
        
        algorithm_config = {
            'lr': 3e-4,
            'gamma': 0.99,
            'clip_ratio': 0.2
        }
        algorithm = PPO(algorithm_config, networks, env.observation_space, env.action_space)
        
        # Test data collection and buffer integration
        print("Testing data collection...")
        
        obs = env.reset(seed=42)
        print(f"Initial obs shape: {obs.shape}")
        
        # Collect some experience
        observations = []
        actions = []
        rewards = []
        old_values = []
        old_log_probs = []
        dones = []
        
        for step in range(64):  # Collect 64 timesteps
            # Get action from algorithm
            action, log_prob, value = algorithm.get_action_and_value(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store data
            observations.append(obs.cpu().numpy())
            actions.append(action.cpu().numpy())
            rewards.append(reward.cpu().numpy())
            old_values.append(value.cpu().numpy())
            old_log_probs.append(log_prob.cpu().numpy())
            dones.append(done.cpu().numpy())
            
            obs = next_obs
        
        # Create trajectory (like trainer does)
        trajectory = {
            'observations': np.array(observations),      # (64, 2, 4)
            'actions': np.array(actions),                # (64, 2)
            'rewards': np.array(rewards),                # (64, 2)
            'old_values': np.array(old_values),          # (64, 2)
            'old_log_probs': np.array(old_log_probs),    # (64, 2)
            'dones': np.array(dones)                     # (64, 2)
        }
        
        print(f"Trajectory shapes:")
        for key, value in trajectory.items():
            print(f"  {key}: {value.shape}")
        
        # Add to buffer
        print("\nAdding trajectory to buffer...")
        buffer.add_trajectory(trajectory)
        print(f"Buffer size after adding: {buffer._size}")
        
        expected_size = 64 * 2  # 64 timesteps * 2 environments
        if buffer._size != expected_size:
            print(f"❌ Buffer size mismatch! Expected {expected_size}, got {buffer._size}")
            return False
        
        # Test sampling
        print("\nTesting buffer sampling...")
        if buffer.ready():
            batch = buffer.sample(32)
            print(f"Sample successful!")
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
            
            # Test algorithm update with this batch
            print("\nTesting algorithm update with sampled batch...")
            update_metrics = algorithm.update(batch)
            print(f"Algorithm update successful!")
            print(f"Update metrics: {list(update_metrics.keys())}")
        else:
            print("Buffer not ready for sampling yet")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error in buffer integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Vectorized Training Pipeline Test Suite")
    print("=" * 50)
    
    # Test complete pipeline
    success1 = test_training_pipeline()
    
    # Test buffer integration
    success2 = test_buffer_integration()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("✅ All pipeline tests passed! Vectorized training is working correctly.")
    else:
        print("❌ Some pipeline tests failed.")
        sys.exit(1)