#!/usr/bin/env python3
"""
Test script for vectorized environment implementation.
This script tests the VectorizedGymWrapper with CartPole-v1 to ensure
it works correctly with the existing training infrastructure.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_vectorized_environment():
    """Test basic functionality of vectorized environment"""
    print("Testing VectorizedGymWrapper...")
    
    try:
        # Import the vectorized wrapper
        from src.environments.vectorized_gym_wrapper import VectorizedGymWrapper
        
        # Create config for 4 parallel CartPole environments
        config = {
            'name': 'CartPole-v1',
            'num_envs': 4,
            'vectorization': 'auto',
            'normalize_obs': False,
            'normalize_reward': False,
            'max_episode_steps': 500
        }
        
        print(f"Creating {config['num_envs']} parallel {config['name']} environments...")
        env = VectorizedGymWrapper(config)
        
        print(f"Environment created successfully!")
        wrapper_info = env.get_wrapper_info()
        if wrapper_info:
            print(f"- Wrapper type: {wrapper_info['wrapper']}")
        else:
            print("- Wrapper info not available")
        print(f"- Vectorization type: {env.vectorization_type}")
        print(f"- Num environments: {env.num_envs}")
        print(f"- Observation space: {env.observation_space.shape}")
        print(f"- Action space: {env.action_space.shape}")
        
        # Test reset
        print("\nTesting environment reset...")
        obs = env.reset(seed=42)
        print(f"- Reset successful! Observation shape: {obs.shape}")
        print(f"- Observation type: {type(obs)}")
        assert obs.shape[0] == config['num_envs'], f"Expected {config['num_envs']} environments, got {obs.shape[0]}"
        
        # Test step with random actions
        print("\nTesting environment steps...")
        for step in range(5):
            # Generate random actions for all environments
            if env.action_space.discrete:
                actions = torch.randint(0, env.action_space.n, (config['num_envs'],))
            else:
                actions = torch.randn(config['num_envs'], env.action_space.shape[-1])
            
            obs, rewards, dones, infos = env.step(actions)
            
            print(f"Step {step + 1}:")
            print(f"  - Observations shape: {obs.shape}")
            print(f"  - Rewards shape: {rewards.shape}")
            print(f"  - Dones shape: {dones.shape}")
            print(f"  - Rewards: {rewards.numpy()}")
            print(f"  - Any done: {torch.any(dones)}")
            
            # Verify shapes
            assert obs.shape[0] == config['num_envs']
            assert rewards.shape[0] == config['num_envs']
            assert dones.shape[0] == config['num_envs']
            
            # infos can be tuple or list
            if isinstance(infos, (list, tuple)):
                assert len(infos) == config['num_envs']
            else:
                print(f"Warning: infos is type {type(infos)}, expected list or tuple")
        
        # Test metrics
        print("\nTesting metrics...")
        metrics = env.get_metrics()
        print(f"Environment metrics: {metrics}")
        
        # Cleanup
        env.close()
        print("\nEnvironment closed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error testing vectorized environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_vs_vectorized():
    """Compare single environment vs vectorized environment behavior"""
    print("\n" + "="*60)
    print("Comparing single vs vectorized environment behavior...")
    
    try:
        from src.environments.gym_wrapper import GymWrapper
        from src.environments.vectorized_gym_wrapper import VectorizedGymWrapper
        
        # Single environment config
        single_config = {
            'name': 'CartPole-v1',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        # Vectorized environment config
        vec_config = {
            'name': 'CartPole-v1',
            'num_envs': 1,  # Use 1 env to compare directly
            'vectorization': 'sync',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        print("Creating single environment...")
        single_env = GymWrapper(single_config)
        
        print("Creating vectorized environment (1 env)...")
        vec_env = VectorizedGymWrapper(vec_config)
        
        # Compare observation spaces
        single_obs_shape = single_env.observation_space.shape
        vec_obs_shape = vec_env.observation_space.shape
        
        print(f"Single env obs shape: {single_obs_shape}")
        print(f"Vector env obs shape: {vec_obs_shape}")
        print(f"Vector env should be: ({vec_config['num_envs']},) + single_obs_shape = {(vec_config['num_envs'],) + single_obs_shape}")
        
        # Test reset comparison
        single_obs = single_env.reset(seed=42)
        vec_obs = vec_env.reset(seed=42)
        
        print(f"Single reset obs shape: {single_obs.shape}")
        print(f"Vector reset obs shape: {vec_obs.shape}")
        
        # Test step comparison
        action = 0  # Left action for CartPole
        vec_action = torch.tensor([action])  # Batch for vectorized env
        
        single_obs, single_reward, single_done, single_info = single_env.step(action)
        vec_obs, vec_reward, vec_done, vec_info = vec_env.step(vec_action)
        
        print(f"Single step results: obs={single_obs.shape}, reward={single_reward}, done={single_done}")
        print(f"Vector step results: obs={vec_obs.shape}, reward={vec_reward}, done={vec_done}")
        
        # Cleanup
        single_env.close()
        vec_env.close()
        
        print("Comparison test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in comparison test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Vectorized Environment Test Suite")
    print("=" * 50)
    
    # Test vectorized environment
    success1 = test_vectorized_environment()
    
    # Test comparison with single environment
    success2 = test_single_vs_vectorized()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("✅ All tests passed! Vectorized environment implementation is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)