#!/usr/bin/env python3
"""
Hyperparameter Sweep Script

Command-line interface for running hyperparameter optimization sweeps using Optuna.
Supports running new sweeps, resuming interrupted ones, and analyzing results.

Usage examples:
  # Run new sweep
  python scripts/sweep.py --config configs/sweeps/ppo_cartpole_sweep.yaml
  
  # Resume existing sweep
  python scripts/sweep.py --resume ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # List existing studies
  python scripts/sweep.py --list --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # Run sweep with custom overrides
  python scripts/sweep.py --config configs/sweeps/ppo_cartpole_sweep.yaml --n-trials 50 --timeout 3600
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.sweep import SweepOrchestrator, create_sweep_orchestrator, resume_sweep, list_studies


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('optuna').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def run_new_sweep(config_path: str, n_trials: Optional[int] = None, 
                  timeout: Optional[int] = None, n_jobs: Optional[int] = None) -> int:
    """
    Run a new hyperparameter sweep.
    
    Args:
        config_path: Path to sweep configuration file
        n_trials: Override number of trials
        timeout: Override timeout in seconds
        n_jobs: Override number of parallel jobs
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Create orchestrator
        orchestrator = create_sweep_orchestrator(config_path)
        
        # Apply command-line overrides
        if n_trials is not None:
            orchestrator.sweep_config.execution['n_trials'] = n_trials
            
        if timeout is not None:
            orchestrator.sweep_config.execution['timeout'] = timeout
            
        if n_jobs is not None:
            orchestrator.sweep_config.execution['n_jobs'] = n_jobs
        
        # Run sweep
        study = orchestrator.run_sweep()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETED: {orchestrator.sweep_config.name}")
        print(f"{'='*60}")
        print(f"Study name: {study.study_name}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"Results saved to: {orchestrator.experiment_dir}")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Sweep failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1


def run_resume_sweep(study_name: str, storage: str, n_trials: Optional[int] = None,
                     timeout: Optional[int] = None) -> int:
    """
    Resume an existing sweep.
    
    Args:
        study_name: Name of study to resume
        storage: Storage URL
        n_trials: Number of additional trials to run
        timeout: Timeout for additional trials
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load existing study
        study = resume_sweep(study_name, storage)
        
        print(f"Resumed study: {study_name}")
        print(f"Completed trials: {len(study.trials)}")
        print(f"Current best value: {study.best_value if study.best_trial else 'None'}")
        
        # Continue optimization if requested
        if n_trials is not None or timeout is not None:
            print(f"Continuing optimization...")
            
            # Note: For full resume functionality, we'd need to recreate the 
            # orchestrator from the saved configuration. For now, just show status.
            print("Full resume functionality requires the original sweep config.")
            print("Use --config with the original sweep configuration for full resume.")
            return 0
        
        return 0
        
    except Exception as e:
        logging.error(f"Resume failed: {e}")
        return 1


def list_existing_studies(storage: str) -> int:
    """
    List all studies in storage.
    
    Args:
        storage: Storage URL
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        studies = list_studies(storage)
        
        if not studies:
            print(f"No studies found in storage: {storage}")
            return 0
        
        print(f"Studies in storage: {storage}")
        print("-" * 40)
        for study_name in studies:
            print(f"  {study_name}")
        print(f"\nTotal: {len(studies)} studies")
        
        return 0
        
    except Exception as e:
        logging.error(f"List studies failed: {e}")
        return 1


def validate_config_file(config_path: str) -> bool:
    """
    Validate that sweep configuration file exists and is readable.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
        return True
    except Exception as e:
        logging.error(f"Invalid configuration file: {e}")
        return False


def main():
    """Main entry point for sweep script"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new sweep
  python scripts/sweep.py --config configs/sweeps/ppo_cartpole_sweep.yaml
  
  # Resume existing sweep  
  python scripts/sweep.py --resume ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # List existing studies
  python scripts/sweep.py --list --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # Run with overrides
  python scripts/sweep.py --config configs/sweeps/ppo_cartpole_sweep.yaml --n-trials 50 --timeout 3600
        """
    )
    
    # Primary action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--config', 
        type=str,
        help="Path to sweep configuration file"
    )
    action_group.add_argument(
        '--resume',
        type=str,
        help="Resume existing sweep by study name"
    )
    action_group.add_argument(
        '--list',
        action='store_true',
        help="List existing studies in storage"
    )
    
    # Storage configuration
    parser.add_argument(
        '--storage',
        type=str,
        help="Optuna storage URL (required for --resume and --list)"
    )
    
    # Execution overrides
    parser.add_argument(
        '--n-trials',
        type=int,
        help="Override number of trials to run"
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help="Override timeout in seconds"
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        help="Override number of parallel jobs (use 1 for single GPU)"
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if args.resume or args.list:
        if not args.storage:
            logging.error("--storage is required when using --resume or --list")
            return 1
    
    # Execute requested action
    if args.config:
        # Validate config file
        if not validate_config_file(args.config):
            return 1
        
        # Run new sweep
        return run_new_sweep(
            config_path=args.config,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs
        )
    
    elif args.resume:
        # Resume existing sweep
        return run_resume_sweep(
            study_name=args.resume,
            storage=args.storage,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
    
    elif args.list:
        # List existing studies
        return list_existing_studies(args.storage)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)