#!/usr/bin/env python3
"""Config Validation Script

Validates that configuration files are correctly loaded and applied
without running actual training. This helps catch config bugs early.

Usage:
    python scripts/validate_config.py --config configs/experiments/your_config.yaml
    python scripts/validate_config.py --config configs/experiments/your_config.yaml --expected key1=value1 key2=value2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import logging

# Setup path to import src modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.paradigms.world_model.trainer import Trainer

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def parse_expected_values(expected_args: List[str]) -> Dict[str, Any]:
    """Parse expected values from command line arguments.

    Args:
        expected_args: List of key=value strings

    Returns:
        Dictionary of expected key-value pairs
    """
    expected = {}
    for arg in expected_args:
        if '=' not in arg:
            print_error(f"Invalid expected value format: {arg}. Use key=value")
            continue

        key, value = arg.split('=', 1)

        # Try to convert to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.lower() == 'none' or value.lower() == 'null':
            value = None
        else:
            try:
                # Try int first
                value = int(value)
            except ValueError:
                try:
                    # Then float
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass

        expected[key] = value

    return expected


def extract_paradigm_config_from_trainer(config_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Initialize trainer to extract actual paradigm config.

    Returns:
        Tuple of (paradigm_config_dict, warnings_list)
    """
    # Suppress INFO logs during initialization, we only want our output
    logging.getLogger('src').setLevel(logging.WARNING)

    # Capture warnings from our config loading
    class WarningCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.warnings = []

        def emit(self, record):
            if record.levelno == logging.WARNING:
                self.warnings.append(record.getMessage())

    warning_handler = WarningCapture()
    logging.getLogger('src.paradigms.world_model.trainer').addHandler(warning_handler)

    try:
        # Load the config
        config = load_config(config_path)

        # Create a minimal trainer instance (won't actually train)
        # This will trigger our config loading logic
        from src.paradigms.world_model.trainer import Trainer

        # We need to extract the paradigm config the same way the trainer does
        # Let's simulate what happens in _create_paradigm
        paradigm_config = {}

        # First, load from 'algorithm' section
        algorithm_obj = getattr(config, 'algorithm', None)
        if algorithm_obj:
            # Convert config object to dict
            if hasattr(algorithm_obj, '__dict__'):
                algorithm_config = vars(algorithm_obj)
            elif isinstance(algorithm_obj, dict):
                algorithm_config = algorithm_obj
            else:
                algorithm_config = {}

            if algorithm_config:
                paradigm_config.update(algorithm_config)

        # Then override with 'paradigm_config' section if present
        paradigm_config_obj = getattr(config, 'paradigm_config', None)
        if paradigm_config_obj:
            # Convert config object to dict
            if hasattr(paradigm_config_obj, '__dict__'):
                paradigm_config_section = vars(paradigm_config_obj)
            elif isinstance(paradigm_config_obj, dict):
                paradigm_config_section = paradigm_config_obj
            else:
                paradigm_config_section = {}

            if paradigm_config_section:
                paradigm_config.update(paradigm_config_section)

        return paradigm_config, warning_handler.warnings

    except Exception as e:
        print_error(f"Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load raw YAML config for comparison."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config_path: str, expected_values: Dict[str, Any] = None):
    """Validate a configuration file.

    Args:
        config_path: Path to config YAML file
        expected_values: Optional dict of expected key-value pairs to verify
    """
    print_header("CONFIG VALIDATION REPORT")

    print_info(f"Config file: {config_path}")

    # Check if file exists
    if not Path(config_path).exists():
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    print_success("Config file found")

    # Load raw YAML
    print_info("Loading raw YAML...")
    raw_config = load_yaml_config(config_path)

    # Extract paradigm config as trainer would
    print_info("Extracting paradigm config as trainer would...")
    paradigm_config, warnings = extract_paradigm_config_from_trainer(config_path)

    # Show which section was used
    print_header("CONFIG SOURCE DETECTION")

    has_algorithm = 'algorithm' in raw_config and raw_config.get('algorithm')
    has_paradigm_config = 'paradigm_config' in raw_config and raw_config.get('paradigm_config')

    if has_algorithm and has_paradigm_config:
        print_warning("Both 'algorithm' and 'paradigm_config' sections found!")
        print_info("  'paradigm_config' values will OVERRIDE 'algorithm' values")
        print_info("  Recommendation: Remove 'paradigm_config' section to avoid confusion")
    elif has_algorithm:
        print_success("Using 'algorithm' section (preferred)")
    elif has_paradigm_config:
        print_warning("Using 'paradigm_config' section (legacy - consider renaming to 'algorithm')")
    else:
        print_error("No 'algorithm' or 'paradigm_config' section found!")
        print_info("  Will use default values only")

    # Display warnings from config loading
    if warnings:
        print_header("WARNINGS FROM CONFIG LOADING")
        for warning in warnings:
            print_warning(warning)

    # Display actual values that will be used
    print_header("PARADIGM CONFIG VALUES (as loaded by trainer)")

    critical_params = [
        'imagination_horizon',
        'world_model_lr',
        'actor_lr',
        'critic_lr',
        'entropy_coef',
        'gamma',
        'lambda_return',
        'critic_target_standardize',
        'critic_real_return_mix',
        'actor_normalize_returns',
        'world_model_warmup_steps',
        'max_grad_norm',
        'world_model_updates_per_batch',
        'actor_updates_per_batch',
        'critic_updates_per_batch'
    ]

    all_passed = True

    for param in critical_params:
        value = paradigm_config.get(param, 'NOT SET (will use default)')

        # Format the value nicely
        if isinstance(value, float):
            if value < 0.01:
                value_str = f"{value:.2e}"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        # Check against expected values if provided
        status = "  "
        if expected_values and param in expected_values:
            expected = expected_values[param]
            if param in paradigm_config:
                actual = paradigm_config[param]

                # Compare with tolerance for floats
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    matches = abs(float(expected) - float(actual)) < 1e-9
                else:
                    matches = expected == actual

                if matches:
                    status = f"{Colors.GREEN}✓{Colors.RESET} "
                    print(f"{status}{param:35s} = {value_str:20s} (expected: {expected})")
                else:
                    status = f"{Colors.RED}✗{Colors.RESET} "
                    print(f"{status}{param:35s} = {value_str:20s} (expected: {expected}, MISMATCH!)")
                    all_passed = False
            else:
                status = f"{Colors.YELLOW}⚠{Colors.RESET} "
                print(f"{status}{param:35s} = NOT SET (expected: {expected}, MISSING!)")
                all_passed = False
        else:
            print(f"  {param:35s} = {value_str}")

    # Additional validation checks
    print_header("VALIDATION CHECKS")

    checks_passed = 0
    checks_total = 0

    # Check 1: imagination_horizon
    checks_total += 1
    horizon = paradigm_config.get('imagination_horizon')
    if horizon is None:
        print_warning(f"imagination_horizon not set (will default to 15)")
        checks_passed += 1
    elif horizon == 1:
        print_error(f"imagination_horizon=1 is extremely low! This defeats the purpose of world models.")
        print_info("  Recommendation: Set to 10-15 for meaningful multi-step planning")
    elif horizon < 5:
        print_warning(f"imagination_horizon={horizon} is quite low")
        print_info("  Recommendation: Consider 10-15 for better credit assignment")
        checks_passed += 1
    else:
        print_success(f"imagination_horizon={horizon} looks good")
        checks_passed += 1

    # Check 2: entropy_coef
    checks_total += 1
    entropy = paradigm_config.get('entropy_coef')
    if entropy is None:
        print_warning("entropy_coef not set (will default to 0.01)")
        checks_passed += 1
    elif entropy < 1e-5:
        print_warning(f"entropy_coef={entropy:.2e} is very low - may cause premature convergence")
        checks_passed += 1
    elif entropy > 0.1:
        print_warning(f"entropy_coef={entropy:.2e} is quite high - may prevent convergence")
        checks_passed += 1
    else:
        print_success(f"entropy_coef={entropy:.2e} looks reasonable")
        checks_passed += 1

    # Check 3: learning rates
    checks_total += 1
    wm_lr = paradigm_config.get('world_model_lr')
    actor_lr = paradigm_config.get('actor_lr')
    critic_lr = paradigm_config.get('critic_lr')

    if wm_lr and actor_lr:
        try:
            wm_lr_f = float(wm_lr)
            actor_lr_f = float(actor_lr)
            if wm_lr_f < actor_lr_f:
                print_warning(f"world_model_lr ({wm_lr_f:.2e}) < actor_lr ({actor_lr_f:.2e})")
                print_info("  This is unusual - typically world model LR should be >= actor LR")
                checks_passed += 1
            else:
                print_success("Learning rate ratios look reasonable")
                checks_passed += 1
        except (ValueError, TypeError):
            print_warning("Could not compare learning rates (type conversion error)")
            checks_passed += 1
    else:
        print_success("Learning rate ratios look reasonable")
        checks_passed += 1

    # Check 4: critic configuration
    checks_total += 1
    standardize = paradigm_config.get('critic_target_standardize')
    if standardize is True:
        print_warning("critic_target_standardize=true may hurt performance on simple environments")
        print_info("  For CartPole, consider setting to false to preserve reward scale")
        checks_passed += 1
    else:
        print_success(f"critic_target_standardize={standardize}")
        checks_passed += 1

    # Final summary
    print_header("VALIDATION SUMMARY")

    if expected_values:
        if all_passed:
            print_success("All expected values match! ✓")
        else:
            print_error("Some expected values don't match! See details above.")

    print_info(f"Validation checks: {checks_passed}/{checks_total} passed")

    if checks_passed == checks_total:
        print_success("Configuration looks good!")
        return 0
    else:
        print_warning("Configuration has some warnings (see above)")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate RL training configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python scripts/validate_config.py --config configs/experiments/my_config.yaml

  # Validate with expected values
  python scripts/validate_config.py --config configs/experiments/my_config.yaml \\
      --expected imagination_horizon=10 entropy_coef=0.001 critic_target_standardize=false
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--expected', '-e',
        nargs='*',
        default=[],
        help='Expected key=value pairs to validate (e.g., imagination_horizon=10 entropy_coef=0.001)'
    )

    args = parser.parse_args()

    # Parse expected values
    expected = parse_expected_values(args.expected) if args.expected else None

    # Run validation
    exit_code = validate_config(args.config, expected)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
