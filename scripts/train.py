#!/usr/bin/env python3
"""
Main Training Script

This is the entry point for running RL experiments. It provides a command-line
interface for starting training runs with configuration files.

Usage examples:
    python scripts/train.py --config configs/experiments/ppo_cartpole.yaml
    python scripts/train.py --config configs/experiments/dqn_atari.yaml --seed 42
    python scripts/train.py --resume experiments/ppo_cartpole_20240101_120000/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trainer import create_trainer_from_config
from src.utils.config import ConfigError


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run RL training experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from experiment directory'
    )
    
    # Overrides
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed override'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps', 'auto'],
        help='Device override (cpu/cuda/mps/auto)'
    )
    
    parser.add_argument(
        '--total-timesteps',
        type=int,
        help='Total training timesteps override'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name override'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (more verbose logging)'
    )
    
    # Dry run
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Initialize components but do not start training'
    )
    
    # Logging level
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration overrides from command line arguments"""
    overrides = {}
    
    if args.seed is not None:
        overrides['experiment'] = {'seed': args.seed}
    
    if args.device is not None:
        if 'experiment' not in overrides:
            overrides['experiment'] = {}
        overrides['experiment']['device'] = args.device
    
    if args.total_timesteps is not None:
        overrides['training'] = {'total_timesteps': args.total_timesteps}
    
    if args.name is not None:
        if 'experiment' not in overrides:
            overrides['experiment'] = {}
        overrides['experiment']['name'] = args.name
    
    if args.debug:
        if 'experiment' not in overrides:
            overrides['experiment'] = {}
        overrides['experiment']['debug'] = True
    
    return overrides


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else args.log_level
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting RL training script")
    
    try:
        # Determine config path and experiment directory
        if args.resume:
            # Resuming from existing experiment
            resume_dir = Path(args.resume)
            if not resume_dir.exists():
                raise ValueError(f"Resume directory does not exist: {resume_dir}")
            
            # Use the NEW config file instead of old checkpoint config
            config_path = Path(args.config)
            if not config_path.exists():
                raise ValueError(f"Configuration file does not exist: {config_path}")
            
            experiment_dir = resume_dir
            logger.info(f"Resuming experiment from: {resume_dir}")
            
        else:
            # Starting new experiment
            config_path = Path(args.config)
            if not config_path.exists():
                raise ValueError(f"Configuration file does not exist: {config_path}")
            
            experiment_dir = None
            logger.info(f"Starting new experiment with config: {config_path}")
        
        # Build configuration overrides
        config_overrides = build_config_overrides(args)
        if config_overrides:
            logger.info(f"Configuration overrides: {config_overrides}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer_from_config(
            str(config_path),
            str(experiment_dir) if experiment_dir else None,
            config_overrides
        )
        
        logger.info(f"Experiment directory: {trainer.experiment_dir}")
        logger.info(f"Configuration hash: {trainer.config.get_hash()}")
        
        # Dry run mode
        if args.dry_run:
            logger.info("Dry run mode - components initialized successfully")
            logger.info("Configuration:")
            import yaml
            print(yaml.dump(trainer.config.to_dict(), default_flow_style=False))
            return
        
        # Run training
        logger.info("Starting training loop...")
        results = trainer.train()
        
        # Print final results
        logger.info("Training completed successfully!")
        logger.info("Final results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            if 'trainer' in locals():
                trainer.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


if __name__ == '__main__':
    sys.exit(main())