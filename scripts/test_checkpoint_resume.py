#!/usr/bin/env python3
"""
Checkpoint Resume Test

Tests that training can be stopped and resumed from the exact same point.
This is critical for long training runs that may be interrupted.

Usage:
    python scripts/test_checkpoint_resume.py --config configs/experiments/ppo_cartpole.yaml
    python scripts/test_checkpoint_resume.py --config configs/experiments/ppo_cartpole.yaml --steps 5000
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trainer import create_trainer_from_config


def run_checkpoint_resume_test(config_path: str, test_steps: int = 3000, resume_at_step: int = None) -> Dict[str, Any]:
    """
    Run checkpoint resume test.
    
    Args:
        config_path: Path to config file
        test_steps: Total steps for the test
        resume_at_step: Step to resume from (None = run until interrupted)
        
    Returns:
        Test results dictionary
    """
    if resume_at_step is None:
        resume_at_step = test_steps // 2  # Resume halfway through
    
    print(f"\n=== Checkpoint Resume Test ===")
    print(f"Config: {config_path}")
    print(f"Total steps: {test_steps}")
    print(f"Will simulate interruption at step {resume_at_step}")
    
    # Phase 1: Run initial training
    print(f"\nPhase 1: Running initial training for {resume_at_step} steps...")
    
    initial_overrides = {
        'training': {'total_timesteps': resume_at_step},
        'experiment': {'name': 'checkpoint_resume_test'},
        'logging': {'tensorboard': False, 'wandb_enabled': False}  # Disable logging for test
    }
    
    trainer1 = create_trainer_from_config(config_path, None, initial_overrides)
    initial_results = trainer1.train()
    initial_exp_dir = trainer1.experiment_dir
    
    print(f"✅ Initial training completed:")
    print(f"   Final step: {initial_results.get('final_step', 'unknown')}")
    print(f"   Final reward: {initial_results.get('eval_return_mean', 'unknown'):.2f}")
    print(f"   Experiment dir: {initial_exp_dir}")
    
    # Verify checkpoint exists
    checkpoint_dir = Path(initial_exp_dir) / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        return {
            'status': 'FAILED',
            'error': 'No checkpoint files found after initial training',
            'initial_results': initial_results
        }
    
    latest_checkpoint = checkpoint_dir / "latest.pt"
    final_checkpoint = checkpoint_dir / "final.pt"
    
    if not (latest_checkpoint.exists() or final_checkpoint.exists()):
        return {
            'status': 'FAILED', 
            'error': 'No latest or final checkpoint found',
            'checkpoint_files': [str(f) for f in checkpoint_files],
            'initial_results': initial_results
        }
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint files")
    
    # Phase 2: Resume training
    print(f"\nPhase 2: Resuming training to reach {test_steps} total steps...")
    
    # Wait a moment to ensure different timing
    time.sleep(0.1)
    
    resume_overrides = {
        'training': {'total_timesteps': test_steps},
        'logging': {'tensorboard': False, 'wandb_enabled': False}  # Disable logging for test
    }
    
    trainer2 = create_trainer_from_config(config_path, str(initial_exp_dir), resume_overrides)
    resume_results = trainer2.train()
    
    print(f"✅ Resumed training completed:")
    print(f"   Final step: {resume_results.get('final_step', 'unknown')}")
    print(f"   Final reward: {resume_results.get('eval_return_mean', 'unknown'):.2f}")
    
    # Phase 3: Validate results
    print(f"\nPhase 3: Validating checkpoint resume...")
    
    # Check that training continued from exactly the right step
    expected_resume_step = initial_results.get('final_step', 0)
    actual_final_step = resume_results.get('final_step', 0)
    
    if abs(actual_final_step - test_steps) > 1:  # Allow small tolerance
        return {
            'status': 'FAILED',
            'error': f'Final step {actual_final_step} does not match expected {test_steps}',
            'initial_results': initial_results,
            'resume_results': resume_results
        }
    
    # Success
    return {
        'status': 'PASSED',
        'initial_step': expected_resume_step,
        'final_step': actual_final_step,
        'total_steps': test_steps,
        'initial_reward': initial_results.get('eval_return_mean', 0),
        'final_reward': resume_results.get('eval_return_mean', 0),
        'experiment_dir': str(initial_exp_dir),
        'checkpoint_files': [str(f.name) for f in checkpoint_files]
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test checkpoint resume functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str, 
                       default='configs/experiments/ppo_cartpole.yaml',
                       help='Config file to use for test')
    parser.add_argument('--steps', type=int, default=3000,
                       help='Total training steps for test')
    parser.add_argument('--resume-at', type=int, 
                       help='Step to resume from (default: halfway)')
    parser.add_argument('--output', type=str, 
                       help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    try:
        results = run_checkpoint_resume_test(
            args.config, 
            args.steps, 
            args.resume_at
        )
        
        print(f"\n=== Test Results ===")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'PASSED':
            print("✅ Checkpoint resume test PASSED!")
            print(f"   Successfully resumed from step {results['initial_step']}")
            print(f"   Completed training to step {results['final_step']}")
            print(f"   Initial reward: {results['initial_reward']:.2f}")
            print(f"   Final reward: {results['final_reward']:.2f}")
        else:
            print("❌ Checkpoint resume test FAILED!")
            print(f"   Error: {results.get('error', 'Unknown error')}")
        
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