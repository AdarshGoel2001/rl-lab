#!/usr/bin/env python3
"""
Quick Setup Script for RL Lab

This script helps set up the framework for immediate use by installing
minimal dependencies and testing the system.
"""

import subprocess
import sys
from pathlib import Path


def install_minimal_deps():
    """Install minimal dependencies for testing"""
    minimal_deps = [
        'torch',
        'numpy', 
        'gymnasium',
        'pyyaml',
        'tqdm'
    ]
    
    print("Installing minimal dependencies...")
    for dep in minimal_deps:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
            return False
    
    print("✓ Minimal dependencies installed")
    return True


def test_imports():
    """Test that key modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        import numpy as np
        import gymnasium as gym
        import yaml
        print("✓ All key modules imported successfully")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


def run_quick_test():
    """Run a quick test of the framework"""
    print("Running quick framework test...")
    
    try:
        # Test component registration
        sys.path.insert(0, str(Path(__file__).parent))
        from src.utils.registry import auto_import_modules, list_registered_components
        
        auto_import_modules()
        components = list_registered_components()
        
        print(f"Registered components: {components}")
        
        if len(components['algorithms']) > 0:
            print("✓ Framework components loaded successfully")
            return True
        else:
            print("⚠ No components registered, but framework structure is working")
            return True
            
    except Exception as e:
        print(f"Framework test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 RL Lab Quick Setup")
    print("=" * 40)
    
    # Install minimal dependencies
    if not install_minimal_deps():
        print("❌ Failed to install dependencies")
        return 1
    
    print()
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return 1
    
    print()
    
    # Test framework
    if not run_quick_test():
        print("❌ Framework test failed")  
        return 1
    
    print()
    print("🎉 Quick setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Run full test: python test_framework.py")
    print("2. Try a dry run: python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run")
    print("3. Install full dependencies: pip install -r requirements.txt")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())