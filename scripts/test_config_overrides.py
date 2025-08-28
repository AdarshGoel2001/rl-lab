#!/usr/bin/env python3
"""
Configuration Override Test

Tests that command-line configuration overrides work correctly.
This ensures the flexibility of the configuration system.

Usage:
    python scripts/test_config_overrides.py --config configs/experiments/ppo_cartpole.yaml
    python scripts/test_config_overrides.py --all-tests
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trainer import create_trainer_from_config


def test_single_override(config_path: str, override_dict: Dict[str, Any], 
                        test_name: str, test_steps: int = 1000) -> Dict[str, Any]:
    """
    Test a single configuration override.
    
    Args:
        config_path: Base config file
        override_dict: Override dictionary
        test_name: Name for this test
        test_steps: Steps to run
        
    Returns:
        Test result dictionary
    """
    print(f"\nTesting: {test_name}")
    print(f"Override: {override_dict}")
    
    try:
        # Add base settings
        full_overrides = {
            'training': {'total_timesteps': test_steps},
            'experiment': {'name': f'config_test_{test_name.lower().replace(" ", "_")}'},
            'logging': {'tensorboard': False, 'wandb_enabled': False},
            **override_dict
        }
        
        trainer = create_trainer_from_config(config_path, None, full_overrides)
        
        # Verify the override was applied by checking the config
        config = trainer.config
        
        # Extract the overridden value for verification
        verification_result = {}
        
        # Check common override paths
        if 'experiment' in override_dict:
            for key, value in override_dict['experiment'].items():
                actual_value = getattr(config.experiment, key, None)
                verification_result[f'experiment.{key}'] = {
                    'expected': value,
                    'actual': actual_value,
                    'matches': actual_value == value
                }
        
        if 'algorithm' in override_dict:
            for key, value in override_dict['algorithm'].items():
                actual_value = getattr(config.algorithm, key, None)
                verification_result[f'algorithm.{key}'] = {
                    'expected': value,
                    'actual': actual_value,
                    'matches': actual_value == value
                }
        
        if 'training' in override_dict:
            for key, value in override_dict['training'].items():
                if key != 'total_timesteps':  # Skip our test override
                    actual_value = getattr(config.training, key, None)
                    verification_result[f'training.{key}'] = {
                        'expected': value,
                        'actual': actual_value,
                        'matches': actual_value == value
                    }
        
        # Quick training run to ensure system still works
        results = trainer.train()
        
        # Check if all overrides were applied correctly
        all_correct = all(v['matches'] for v in verification_result.values())
        
        print(f"   Config verification: {'✅ PASSED' if all_correct else '❌ FAILED'}")
        for path, check in verification_result.items():
            status = '✅' if check['matches'] else '❌'
            print(f"     {path}: {check['expected']} → {check['actual']} {status}")
        
        print(f"   Training: ✅ Completed {results.get('final_step', 0)} steps")
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if all_correct else 'FAILED',
            'overrides_applied': verification_result,
            'training_results': {
                'final_step': results.get('final_step', 0),
                'final_reward': results.get('eval_return_mean', 0)
            },
            'experiment_dir': str(trainer.experiment_dir)
        }
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return {
            'test_name': test_name,
            'status': 'FAILED',
            'error': str(e)
        }


def run_config_override_tests(config_path: str, test_steps: int = 1000) -> Dict[str, Any]:
    """
    Run comprehensive configuration override tests.
    
    Args:
        config_path: Base config file
        test_steps: Steps per test
        
    Returns:
        Complete test results
    """
    print(f"\n=== Configuration Override Tests ===")
    print(f"Base config: {config_path}")
    print(f"Steps per test: {test_steps}")
    
    # Define test cases
    test_cases = [
        {
            'name': 'Seed Override',
            'overrides': {'experiment': {'seed': 999}},
            'description': 'Test changing random seed'
        },
        {
            'name': 'Learning Rate Override', 
            'overrides': {'algorithm': {'lr': 0.001}},
            'description': 'Test changing algorithm learning rate'
        },
        {
            'name': 'Device Override',
            'overrides': {'experiment': {'device': 'cpu'}},
            'description': 'Test changing device setting'
        },
        {
            'name': 'Multiple Overrides',
            'overrides': {
                'experiment': {'seed': 777},
                'algorithm': {'lr': 0.0005}
            },
            'description': 'Test multiple simultaneous overrides'
        },
        {
            'name': 'Nested Override',
            'overrides': {'algorithm': {'clip_ratio': 0.1}},
            'description': 'Test overriding nested config values'
        }
    ]
    
    results = []
    
    # Run each test case
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Description: {test_case['description']}")
        
        result = test_single_override(
            config_path,
            test_case['overrides'],
            test_case['name'],
            test_steps
        )
        
        results.append(result)
    
    # Analyze overall results
    successful_tests = [r for r in results if r['status'] == 'PASSED']
    failed_tests = [r for r in results if r['status'] == 'FAILED']
    
    overall_status = 'PASSED' if len(failed_tests) == 0 else 'FAILED'
    
    print(f"\n=== Overall Results ===")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Overall: {'✅ PASSED' if overall_status == 'PASSED' else '❌ FAILED'}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"   - {test['test_name']}: {test.get('error', 'Config verification failed')}")
    
    return {
        'status': overall_status,
        'total_tests': len(results),
        'passed_tests': len(successful_tests),
        'failed_tests': len(failed_tests),
        'test_results': results,
        'config_path': config_path,
        'steps_per_test': test_steps
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test configuration override functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str,
                       default='configs/experiments/ppo_cartpole.yaml',
                       help='Base config file for tests')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Training steps per test')
    parser.add_argument('--test', type=str,
                       choices=['seed', 'lr', 'device', 'multiple', 'nested'],
                       help='Run specific test only')
    parser.add_argument('--all-tests', action='store_true',
                       help='Run all configuration override tests')
    parser.add_argument('--output', type=str,
                       help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            # Run specific test
            test_configs = {
                'seed': {'experiment': {'seed': 999}},
                'lr': {'algorithm': {'lr': 0.001}}, 
                'device': {'experiment': {'device': 'cpu'}},
                'multiple': {
                    'experiment': {'seed': 777},
                    'algorithm': {'lr': 0.0005}
                },
                'nested': {'algorithm': {'clip_ratio': 0.1}}
            }
            
            if args.test not in test_configs:
                print(f"❌ Unknown test: {args.test}")
                return 1
                
            result = test_single_override(
                args.config,
                test_configs[args.test],
                args.test.title() + " Test",
                args.steps
            )
            
            results = {
                'status': result['status'],
                'single_test': result
            }
            
        else:
            # Run all tests
            results = run_config_override_tests(args.config, args.steps)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")
        
        return 0 if results['status'] == 'PASSED' else 1
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())