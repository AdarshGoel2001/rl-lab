#!/usr/bin/env python3
"""
Run All Infrastructure Tests

Runs all core infrastructure tests for the RL monorepo system.
These tests validate the critical components that TensorBoard/W&B cannot test.

Usage:
    python scripts/run_all_tests.py
    python scripts/run_all_tests.py --config configs/experiments/ppo_cartpole.yaml
    python scripts/run_all_tests.py --quick
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Dict, List


def run_test_script(script_name: str, args: List[str] = None) -> Dict[str, any]:
    """Run a test script and return results."""
    if args is None:
        args = []
    
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"\n{'='*50}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return {
            'script': script_name,
            'returncode': result.returncode,
            'status': 'PASSED' if result.returncode == 0 else 'FAILED'
        }
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return {
            'script': script_name,
            'returncode': -1,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run all RL infrastructure tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str,
                       default='configs/experiments/ppo_cartpole.yaml',
                       help='Config file for tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run tests with reduced steps for speed')
    parser.add_argument('--test', choices=['checkpoint', 'reproducibility', 'config'],
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    # Determine test parameters
    steps = 1000 if args.quick else 2000
    
    print(f"\n🧪 Running RL Infrastructure Tests")
    print(f"Config: {args.config}")
    print(f"Steps: {steps} ({'quick' if args.quick else 'normal'})")
    
    results = []
    
    # Define tests to run
    if args.test == 'checkpoint':
        tests = [('test_checkpoint_resume.py', ['--config', args.config, '--steps', str(steps)])]
    elif args.test == 'reproducibility':
        tests = [('test_reproducibility.py', ['--config', args.config, '--steps', str(steps)])]
    elif args.test == 'config':
        tests = [('test_config_overrides.py', ['--config', args.config, '--steps', str(steps), '--all-tests'])]
    else:
        # Run all tests
        tests = [
            ('test_config_overrides.py', ['--config', args.config, '--steps', str(steps), '--all-tests']),
            ('test_reproducibility.py', ['--config', args.config, '--steps', str(steps)]),
            ('test_checkpoint_resume.py', ['--config', args.config, '--steps', str(steps)])
        ]
    
    # Run tests
    for script_name, test_args in tests:
        result = run_test_script(script_name, test_args)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    
    passed_tests = [r for r in results if r['status'] == 'PASSED']
    failed_tests = [r for r in results if r['status'] in ['FAILED', 'ERROR']]
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {len(passed_tests)} ✅")
    print(f"Failed: {len(failed_tests)} ❌")
    
    for result in results:
        status_symbol = '✅' if result['status'] == 'PASSED' else '❌'
        print(f"  {result['script']}: {result['status']} {status_symbol}")
    
    overall_status = 'PASSED' if len(failed_tests) == 0 else 'FAILED'
    print(f"\nOverall: {overall_status} {'✅' if overall_status == 'PASSED' else '❌'}")
    
    if overall_status == 'PASSED':
        print("\n🎉 All infrastructure tests passed!")
        print("Your RL system core infrastructure is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return 0 if overall_status == 'PASSED' else 1


if __name__ == '__main__':
    sys.exit(main())