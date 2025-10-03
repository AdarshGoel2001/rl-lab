#!/usr/bin/env python3
"""
Debug script to isolate the PPO division by zero issue
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.algorithms.ppo_legacy import PPOAlgorithm
from src.utils.registry import auto_import_modules

# Import modules to register components
auto_import_modules()

def test_ppo_algorithm():
    print("Testing PPO algorithm in isolation...")
    
    try:
        # Create PPO algorithm
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
            'max_grad_norm': 0.5,
            'normalize_advantages': True
        }
        
        ppo = PPOAlgorithm(ppo_config)
        print("✅ PPO algorithm created successfully")
        
        # Test action generation
        obs = torch.randn(4, 4)
        actions, log_probs, values = ppo.get_action_and_value(obs)
        print(f"✅ Action generation works: actions={actions.shape}, log_probs={log_probs.shape}, values={values.shape}")
        
        # Test with different types of advantage data
        test_cases = [
            ("Normal advantages", torch.randn(32) * 0.5 + 0.1),
            ("Zero std advantages", torch.ones(32) * 0.5),
            ("All zero advantages", torch.zeros(32)),
            ("Very small advantages", torch.randn(32) * 1e-10),
        ]
        
        for case_name, advantages in test_cases:
            print(f"\nTesting {case_name}:")
            print(f"  Mean: {advantages.mean().item():.6f}, Std: {advantages.std().item():.6f}")
            
            batch = {
                'observations': torch.randn(32, 4),
                'actions': torch.randint(0, 2, (32,)),
                'rewards': torch.randn(32),
                'dones': torch.randint(0, 2, (32,)).bool(),
                'old_values': torch.randn(32),
                'old_log_probs': torch.randn(32),
                'advantages': advantages,
                'returns': torch.randn(32),
            }
            
            try:
                metrics = ppo.update(batch)
                print(f"  ✅ Update successful: {list(metrics.keys())}")
            except Exception as e:
                print(f"  ❌ Update failed: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ PPO algorithm creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ppo_algorithm()