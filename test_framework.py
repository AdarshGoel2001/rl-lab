#!/usr/bin/env python3
"""
Framework Test Script

Quick test to verify that all components are working together.
This test creates a minimal experiment setup and runs a few steps.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_component_registration():
    """Test that components can be imported and registered"""
    print("Testing component registration...")
    
    # Import registry and trigger auto-import
    from src.utils.registry import (
        auto_import_modules, list_registered_components,
        ALGORITHM_REGISTRY, NETWORK_REGISTRY, ENVIRONMENT_REGISTRY, BUFFER_REGISTRY
    )
    
    # Auto-import modules to populate registries
    auto_import_modules()
    
    # Check registrations
    components = list_registered_components()
    print(f"Registered components: {components}")
    
    # Verify we have our test components
    assert 'random' in ALGORITHM_REGISTRY, "Random algorithm not registered"
    assert 'mlp' in NETWORK_REGISTRY, "MLP network not registered" 
    assert 'gym' in ENVIRONMENT_REGISTRY, "Gym environment not registered"
    assert 'trajectory' in BUFFER_REGISTRY, "Trajectory buffer not registered"
    
    print("✓ Component registration test passed")


def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    from src.utils.config import load_config
    
    config_path = Path(__file__).parent / "configs" / "experiments" / "test_cartpole.yaml"
    config = load_config(config_path)
    
    assert config.experiment.name == "test_cartpole_random"
    assert config.algorithm.name == "random"
    assert config.environment.name == "CartPole-v1"
    
    print("✓ Configuration loading test passed")


def test_component_creation():
    """Test creating individual components"""
    print("Testing component creation...")
    
    from src.utils.config import load_config
    from src.utils.registry import get_algorithm, get_environment, get_network, get_buffer
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "experiments" / "test_cartpole.yaml"
    config = load_config(config_path)
    
    # Test environment creation
    env_class = get_environment(config.environment.wrapper)
    env = env_class(config.environment.__dict__)
    
    obs_space = env.observation_space
    action_space = env.action_space
    print(f"Environment created: {config.environment.name}")
    print(f"  Obs space: {obs_space.shape}, Action space: {action_space.shape}")
    
    # Test network creation
    net_config = config.network.__dict__.copy()
    net_config['input_dim'] = obs_space.shape
    net_config['output_dim'] = action_space.shape[0] if not action_space.discrete else action_space.n
    
    net_class = get_network(config.network.type)
    network = net_class(net_config)
    print(f"Network created: {config.network.type}")
    print(f"  Parameters: {network.parameter_count()}")
    
    # Test algorithm creation
    alg_config = config.algorithm.__dict__.copy()
    alg_config['action_space'] = action_space
    alg_config['observation_space'] = obs_space
    
    alg_class = get_algorithm(config.algorithm.name)
    algorithm = alg_class(alg_config)
    print(f"Algorithm created: {config.algorithm.name}")
    
    # Test buffer creation
    buffer_class = get_buffer(config.buffer.type)
    buffer = buffer_class(config.buffer.__dict__)
    print(f"Buffer created: {config.buffer.type}")
    
    # Clean up
    env.close()
    
    print("✓ Component creation test passed")


def test_basic_interaction():
    """Test basic environment interaction"""
    print("Testing basic environment interaction...")
    
    try:
        import gymnasium as gym
    except ImportError:
        print("⚠ Gymnasium not installed, skipping interaction test")
        return
    
    from src.environments.gym_wrapper import GymWrapper
    
    # Create environment
    env_config = {
        'name': 'CartPole-v1',
        'normalize_obs': False,
        'normalize_reward': False
    }
    
    env = GymWrapper(env_config)
    
    # Test reset and step
    obs = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    # Random action
    import torch
    if env.action_space.discrete:
        action = torch.randint(0, env.action_space.n, (1,))
    else:
        action = torch.randn(env.action_space.shape)
    
    next_obs, reward, done, info = env.step(action)
    print(f"Step result: obs_shape={next_obs.shape}, reward={reward}, done={done}")
    
    env.close()
    print("✓ Basic interaction test passed")


def main():
    """Run all tests"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("🚀 Testing RL Framework Components\n")
        
        test_component_registration()
        print()
        
        test_config_loading() 
        print()
        
        test_component_creation()
        print()
        
        test_basic_interaction()
        print()
        
        print("🎉 All tests passed! The framework is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run a test experiment: python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())