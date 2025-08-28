#!/usr/bin/env python3
"""
Installation Validation Script for RL Lab

This script validates that all components of the RL Lab framework
are properly installed and provides detailed diagnostics for any issues.
"""

import sys
import platform
import subprocess
from pathlib import Path
import importlib.util


def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)


def print_success(message):
    """Print success message"""
    print(f"✅ {message}")


def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")


def print_error(message):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")


def check_python_version():
    """Check Python version compatibility"""
    print_header("Python Environment")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Python version: {version_str}")
    print_info(f"Platform: {platform.platform()}")
    print_info(f"Architecture: {platform.machine()}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version_str}")
        return False


def check_package_import(package_name, import_name=None):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print_error(f"{package_name}: Not installed")
            return False
        
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print_success(f"{package_name}: {version}")
        return True
        
    except ImportError as e:
        print_error(f"{package_name}: Import failed - {e}")
        return False
    except Exception as e:
        print_warning(f"{package_name}: Available but error getting version - {e}")
        return True


def check_core_dependencies():
    """Check core Python dependencies"""
    print_header("Core Dependencies")
    
    core_packages = [
        ('PyTorch', 'torch'),
        ('NumPy', 'numpy'),
        ('Gymnasium', 'gymnasium'),
        ('PyYAML', 'yaml'),
        ('tqdm', 'tqdm'),
        ('Pandas', 'pandas'),
        ('SciPy', 'scipy'),
    ]
    
    success_count = 0
    total_count = len(core_packages)
    
    for package_name, import_name in core_packages:
        if check_package_import(package_name, import_name):
            success_count += 1
    
    print(f"\n📊 Core Dependencies: {success_count}/{total_count} installed")
    return success_count == total_count


def check_rl_dependencies():
    """Check RL-specific dependencies"""
    print_header("RL-Specific Dependencies")
    
    rl_packages = [
        ('TensorBoard', 'tensorboard'),
        ('WandB', 'wandb'),
        ('Optuna', 'optuna'),
        ('DM-Control', 'dm_control'),
        ('Hydra', 'hydra'),
    ]
    
    success_count = 0
    total_count = len(rl_packages)
    
    for package_name, import_name in rl_packages:
        if check_package_import(package_name, import_name):
            success_count += 1
    
    print(f"\n📊 RL Dependencies: {success_count}/{total_count} installed")
    return success_count >= total_count * 0.7  # Allow some optional packages to be missing


def check_pytorch_backend():
    """Check PyTorch backend capabilities"""
    print_header("PyTorch Backend")
    
    try:
        import torch
        print_info(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.version.cuda}")
            print_info(f"CUDA devices: {torch.cuda.device_count()}")
        else:
            print_info("CUDA not available (normal for CPU-only or macOS)")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("MPS (Metal Performance Shaders) available")
        else:
            print_info("MPS not available (normal for non-Apple Silicon)")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.mm(x, x)
        print_success("Basic tensor operations working")
        
        return True
        
    except Exception as e:
        print_error(f"PyTorch backend check failed: {e}")
        return False


def check_framework_components():
    """Check RL Lab framework components"""
    print_header("RL Lab Framework")
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    try:
        # Test component imports
        from src.utils.registry import auto_import_modules, list_registered_components
        print_success("Registry system importable")
        
        # Test auto-import
        auto_import_modules()
        components = list_registered_components()
        
        print_info(f"Registered algorithms: {len(components['algorithms'])}")
        print_info(f"Registered networks: {len(components['networks'])}")  
        print_info(f"Registered environments: {len(components['environments'])}")
        print_info(f"Registered buffers: {len(components['buffers'])}")
        
        total_components = sum(len(v) for v in components.values())
        if total_components > 0:
            print_success(f"Framework components loaded ({total_components} total)")
            return True
        else:
            print_warning("No framework components registered")
            return False
            
    except Exception as e:
        print_error(f"Framework import failed: {e}")
        return False


def check_environment_creation():
    """Test environment creation"""
    print_header("Environment Test")
    
    try:
        from src.environments.gym_wrapper import GymWrapper
        import torch
        import numpy as np
        
        env_config = {
            'name': 'CartPole-v1',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        env = GymWrapper(env_config)
        obs = env.reset(seed=42)
        print_success("Environment creation and reset successful")
        
        # Test different action formats
        test_actions = [
            ("PyTorch tensor scalar", torch.tensor(0)),
            ("PyTorch tensor with shape", torch.tensor([1])),
            ("NumPy scalar", np.array(0)),
            ("NumPy array", np.array([1])),
            ("Python int", 1),
        ]
        
        for action_name, action in test_actions:
            try:
                next_obs, reward, done, info = env.step(action)
                print_success(f"Environment step successful with {action_name}")
                
                # Reset if done
                if done:
                    obs = env.reset(seed=42)
                    
            except Exception as e:
                print_error(f"Environment step failed with {action_name}: {e}")
                # Don't fail the whole test for individual action format issues
                continue
        
        print_success("Environment action handling test completed")
        env.close()
        return True
        
    except Exception as e:
        print_error(f"Environment test failed: {e}")
        return False


def provide_installation_guidance():
    """Provide installation guidance based on platform"""
    print_header("Installation Guidance")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("🍎 macOS Installation Options:")
        print("   1. Conda (Recommended):")
        print("      conda env create -f environment-macos.yml")
        print("   2. Auto-setup script:")
        print("      ./setup_env.sh")
        print("   3. Pip fallback:")
        print("      pip install -r requirements-macos.txt")
        
    elif system == "Linux":
        print("🐧 Linux Installation Options:")
        print("   1. With CUDA (if you have NVIDIA GPU):")
        print("      conda env create -f environment-linux-cuda.yml")
        print("   2. CPU-only:")
        print("      conda env create -f environment-linux-cpu.yml")
        print("   3. Auto-detect:")
        print("      ./setup_env.sh")
        print("   4. Pip fallback:")
        print("      pip install -r requirements.txt")
    
    else:
        print(f"⚠️  {system} - Use generic installation:")
        print("   1. Conda:")
        print("      conda env create -f environment.yml")
        print("   2. Pip:")
        print("      pip install -r requirements.txt")
    
    print("\n🔧 Troubleshooting:")
    print("   • Make sure conda/python environment is activated")
    print("   • Update pip: python -m pip install --upgrade pip")  
    print("   • For permission issues, use --user flag with pip")
    print("   • Check internet connection for downloads")


def main():
    """Main validation function"""
    print("🚀 RL Lab Installation Validator")
    print("This script will check your installation and provide guidance")
    
    checks = [
        ("Python Environment", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("RL Dependencies", check_rl_dependencies),
        ("PyTorch Backend", check_pytorch_backend),
        ("Framework Components", check_framework_components),
        ("Environment Test", check_environment_creation),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Check failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("🎉 Installation is complete and working!")
        print_info("You can now run: python test_framework.py")
        return 0
    else:
        print_warning(f"⚠️  {total - passed} checks failed")
        provide_installation_guidance()
        return 1


if __name__ == '__main__':
    sys.exit(main())