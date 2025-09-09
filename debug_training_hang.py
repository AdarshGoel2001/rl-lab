#!/usr/bin/env python3
"""
Debug script to isolate where training hangs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_components():
    """Test each component of the training pipeline individually"""
    
    print("=== Testing Training Components ===")
    
    # Load config
    with open('configs/experiments/ppo_atari_pong_vectorized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Test environment creation and reset
    print("\n1. Testing Environment...")
    from src.environments.vectorized_gym_wrapper import VectorizedGymWrapper
    
    env_config = config['environment']
    env_config['num_envs'] = 2  # Smaller for testing
    
    env = VectorizedGymWrapper(env_config)
    print(f"✓ Created environment: {env.num_envs} envs")
    
    obs = env.reset(seed=42)
    print(f"✓ Reset successful: {obs.shape}, device: {obs.device}")
    
    # 2. Test network creation
    print("\n2. Testing Networks...")
    from src.networks.cnn import NatureCNN
    
    # Create networks
    actor_config = config['network']['actor'].copy()
    actor_config['input_dim'] = (84, 84, 4)
    actor_config['output_dim'] = env.action_space.n
    
    actor_net = NatureCNN(actor_config)
    print(f"✓ Created actor network: {actor_config['input_dim']} -> {actor_config['output_dim']}")
    
    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    actor_net = actor_net.to(device)
    obs = obs.to(device)
    
    print(f"✓ Moved to device: {device}")
    
    # 3. Test forward pass
    print("\n3. Testing Network Forward Pass...")
    try:
        with torch.no_grad():
            output = actor_net(obs)
            print(f"✓ Network forward pass: {obs.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ Network forward pass failed: {e}")
        raise
    
    # 4. Test algorithm creation
    print("\n4. Testing Algorithm Creation...")
    from src.algorithms.ppo import PPOAlgorithm
    
    algo_config = config['algorithm'].copy()
    algo_config['device'] = device
    algo_config['action_space_n'] = env.action_space.n
    algo_config['observation_space'] = env.observation_space
    
    # Create dummy networks dict for algorithm
    networks = {
        'actor': actor_net,
        'critic': NatureCNN({
            'input_dim': (84, 84, 4),
            'output_dim': 1,
            'channels': [32, 64, 64],
            'hidden_dim': 512,
            'activation': 'relu'
        }).to(device)
    }
    
    try:
        algorithm = PPOAlgorithm(algo_config)
        print(f"✓ Created PPO algorithm")
        
        # Now set the networks manually
        algorithm.networks = networks
        algorithm.actor_network = networks['actor']
        algorithm.critic_network = networks['critic']
        
    except Exception as e:
        print(f"✗ Algorithm creation failed: {e}")
        raise
    
    # 5. Test get_action_and_value
    print("\n5. Testing get_action_and_value...")
    try:
        with torch.no_grad():
            if hasattr(algorithm, 'get_action_and_value'):
                actions, log_probs, values = algorithm.get_action_and_value(obs)
                print(f"✓ get_action_and_value: actions {actions.shape}, log_probs {log_probs.shape}, values {values.shape}")
            else:
                print("✗ get_action_and_value method not found")
    except Exception as e:
        print(f"✗ get_action_and_value failed: {e}")
        raise
    
    # 6. Test environment step
    print("\n6. Testing Environment Step...")
    try:
        actions_np = actions.cpu().numpy()
        next_obs, rewards, dones, infos = env.step(actions_np)
        print(f"✓ Environment step: {next_obs.shape}, rewards: {rewards.shape}, dones: {dones.shape}")
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        raise
    
    env.close()
    print("\n✅ All components working individually!")
    
    # 7. Now test the problematic trainer method
    print("\n7. Testing Trainer Integration...")
    test_trainer_collection(config)


def test_trainer_collection(config):
    """Test the specific trainer method that's hanging"""
    
    print("Testing trainer _collect_vectorized_experience...")
    
    from src.core.trainer import Trainer
    from src.utils.config import load_config
    
    # Create trainer but don't run full training
    config_obj = load_config('configs/experiments/ppo_atari_pong_vectorized.yaml')
    trainer = Trainer(config_obj)
    
    print("✓ Trainer created successfully")
    print(f"Environment type: {type(trainer.environment)}")
    print(f"Is vectorized: {getattr(trainer.environment, 'is_vectorized', False)}")
    
    # Manually test the problematic method
    try:
        print("Calling _collect_vectorized_experience...")
        trajectory = trainer._collect_vectorized_experience()
        print(f"✓ First trajectory collection successful: {trajectory is not None}")
    except Exception as e:
        print(f"✗ _collect_vectorized_experience failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        test_training_components()
        print("\n🎉 All tests passed! Training should work.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)