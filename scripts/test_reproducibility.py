#!/usr/bin/env python3
"""
Reproducibility Test

Tests that identical configurations with the same seed produce identical results.
This is critical for scientific reproducibility and debugging.

Usage:
    python scripts/test_reproducibility.py --config configs/experiments/ppo_cartpole.yaml
    python scripts/test_reproducibility.py --config configs/experiments/ppo_cartpole.yaml --seed 42 --runs 3
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trainer import create_trainer_from_config


def run_reproducibility_test(config_path: str, seed: int = 42, num_runs: int = 2, 
                           test_steps: int = 2000) -> Dict[str, Any]:
    """
    Run reproducibility test with identical configurations.
    
    Args:
        config_path: Path to config file
        seed: Random seed to use for all runs
        num_runs: Number of identical runs to perform
        test_steps: Number of training steps per run
        
    Returns:
        Test results dictionary
    """
    print(f"\n=== Reproducibility Test ===")
    print(f"Config: {config_path}")
    print(f"Seed: {seed}")
    print(f"Runs: {num_runs}")
    print(f"Steps per run: {test_steps}")
    
    results = []
    
    # Run multiple identical experiments
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}: Starting training...")
        
        overrides = {
            'experiment': {
                'name': f'reproducibility_test_run_{run_idx + 1}',
                'seed': seed  # Same seed for all runs
            },
            'training': {'total_timesteps': test_steps},
            'logging': {'tensorboard': False, 'wandb_enabled': False}  # Disable logging for test
        }
        
        try:
            trainer = create_trainer_from_config(config_path, None, overrides)
            run_results = trainer.train()
            
            # Extract key metrics
            final_reward = run_results.get('eval_return_mean', 0.0)
            final_step = run_results.get('final_step', 0)
            training_time = run_results.get('training_time', 0.0)
            
            results.append({
                'run': run_idx + 1,
                'final_reward': final_reward,
                'final_step': final_step,
                'training_time': training_time,
                'experiment_dir': str(trainer.experiment_dir),
                'full_results': run_results
            })
            
            print(f"   Final reward: {final_reward:.4f}")
            print(f"   Final step: {final_step}")
            print(f"   Training time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Run {run_idx + 1} failed: {e}")
            results.append({
                'run': run_idx + 1,
                'error': str(e)
            })
    
    # Analyze reproducibility
    print(f"\n=== Analysis ===")
    
    # Filter successful runs
    successful_runs = [r for r in results if 'error' not in r]
    
    if len(successful_runs) < 2:
        return {
            'status': 'FAILED',
            'error': f'Need at least 2 successful runs, got {len(successful_runs)}',
            'results': results
        }
    
    # Compare final rewards
    final_rewards = [r['final_reward'] for r in successful_runs]
    reward_mean = np.mean(final_rewards)
    reward_std = np.std(final_rewards)
    reward_range = max(final_rewards) - min(final_rewards)
    
    print(f"Final Rewards:")
    for i, reward in enumerate(final_rewards):
        print(f"   Run {i+1}: {reward:.6f}")
    print(f"   Mean: {reward_mean:.6f}")
    print(f"   Std:  {reward_std:.6f}")
    print(f"   Range: {reward_range:.6f}")
    
    # Compare final steps (should be identical)
    final_steps = [r['final_step'] for r in successful_runs]
    steps_identical = len(set(final_steps)) == 1
    
    print(f"\nFinal Steps:")
    for i, step in enumerate(final_steps):
        print(f"   Run {i+1}: {step}")
    print(f"   Identical: {steps_identical}")
    
    # Determine if test passed
    # For reproducibility, we expect very small differences (< 0.1% of mean reward)
    tolerance = max(0.01, abs(reward_mean) * 0.001)  # 0.1% tolerance or minimum 0.01
    
    reward_reproducible = reward_std < tolerance
    overall_passed = reward_reproducible and steps_identical
    
    print(f"\n=== Results ===")
    print(f"Reward reproducible (std < {tolerance:.4f}): {reward_reproducible}")
    print(f"Steps identical: {steps_identical}")
    print(f"Overall: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    
    if not overall_passed:
        if not reward_reproducible:
            print(f"   Reward variation ({reward_std:.6f}) exceeds tolerance ({tolerance:.4f})")
        if not steps_identical:
            print(f"   Final steps not identical: {set(final_steps)}")
    
    return {
        'status': 'PASSED' if overall_passed else 'FAILED',
        'num_runs': len(successful_runs),
        'failed_runs': len(results) - len(successful_runs),
        'reward_stats': {
            'mean': float(reward_mean),
            'std': float(reward_std),
            'range': float(reward_range),
            'values': final_rewards,
            'reproducible': reward_reproducible,
            'tolerance': tolerance
        },
        'step_stats': {
            'values': final_steps,
            'identical': steps_identical
        },
        'config_used': {
            'seed': seed,
            'test_steps': test_steps,
            'config_path': config_path
        },
        'detailed_results': results
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test experiment reproducibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str,
                       default='configs/experiments/ppo_cartpole.yaml', 
                       help='Config file to use for test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility test')
    parser.add_argument('--runs', type=int, default=2,
                       help='Number of identical runs to perform')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Training steps per run')
    parser.add_argument('--output', type=str,
                       help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    if args.runs < 2:
        print("❌ Error: Need at least 2 runs for reproducibility test")
        return 1
    
    try:
        results = run_reproducibility_test(
            args.config,
            args.seed, 
            args.runs,
            args.steps
        )
        
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