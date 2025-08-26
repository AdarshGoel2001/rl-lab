# RL Lab Setup Guide - Fixed for macOS

## 🎯 Quick Start (Fixed CUDA Issue)

The original `environment.yml` had CUDA dependencies that don't work on macOS. This has been **fixed** with platform-specific environments.

### **Step 1: Install Using Fixed Environment**
```bash
# Option A: Auto-detect platform (Recommended)
./setup_env.sh

# Option B: Explicitly use macOS environment  
conda env create -f environment-macos.yml

# Option C: Use pip if conda fails
./install_pip.sh
```

### **Step 2: Activate Environment**
```bash
conda activate rl-lab
# or use the enhanced activation script
source activate_env.sh
```

### **Step 3: Validate Installation**
```bash
# Comprehensive validation with diagnostics
python validate_installation.py

# Quick framework test
python test_framework.py

# Try a dry run
python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run
```

## 🔧 What Was Fixed

### **Original Problem**
```
LibMambaUnsatisfiableError: Encountered problems while solving:
  - nothing provides cuda 11.8.* needed by pytorch-cuda-11.8-h8dd9ede_2
```

### **Solutions Implemented**

1. **Removed CUDA Dependencies**: Fixed `environment.yml` to work on macOS
2. **Platform-Specific Environments**: 
   - `environment-macos.yml` - Apple Silicon optimized
   - `environment-linux-cuda.yml` - Linux with GPU
   - `environment-linux-cpu.yml` - Linux CPU-only
3. **Smart Setup Scripts**: Auto-detect platform and choose correct environment
4. **Pip Fallbacks**: Pure pip installation when conda fails
5. **Installation Validator**: Comprehensive diagnostics and guidance

## 📋 Available Installation Methods

### **Method 1: Smart Setup (Recommended)**
```bash
./setup_env.sh
```
**What it does:**
- Auto-detects your platform (macOS/Linux/Windows)
- Chooses appropriate environment file
- Handles existing environment updates
- Provides clear error messages

### **Method 2: Manual Platform Selection**
```bash
# Force specific environment
./setup_env.sh environment-macos.yml
./setup_env.sh environment-linux-cuda.yml
./setup_env.sh environment-linux-cpu.yml
```

### **Method 3: Pure Pip Installation**
```bash
# Auto-detect platform requirements
./install_pip.sh

# Or manual pip installation
pip install -r requirements-macos.txt  # for macOS
pip install -r requirements.txt        # for Linux/Windows
pip install -r requirements-minimal.txt # minimal install
```

### **Method 4: Manual Conda**
```bash
# Create environment with specific file
conda env create -f environment-macos.yml
conda activate rl-lab
```

## 🧪 Testing Your Installation

### **Level 1: Installation Validation**
```bash
python validate_installation.py
```
**Expected Output:**
```
✅ PASS Python Environment
✅ PASS Core Dependencies  
✅ PASS RL Dependencies
✅ PASS PyTorch Backend
✅ PASS Framework Components
✅ PASS Environment Test

📊 Overall: 6/6 checks passed
🎉 Installation is complete and working!
```

### **Level 2: Framework Components**
```bash
python test_framework.py
```
**Expected Output:**
```
✅ Component registration test passed
✅ Configuration loading test passed  
✅ Component creation test passed
✅ Basic interaction test passed
🎉 All tests passed! The framework is working correctly.
```

### **Level 3: Full Integration**
```bash
python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run
```

## 🚨 Troubleshooting Common Issues

### **"CUDA not available" (Normal on macOS)**
This is **expected** on Apple Silicon Macs. PyTorch will use:
- **MPS (Metal Performance Shaders)** for GPU acceleration on Apple Silicon
- **CPU** as fallback

### **"Failed to create environment"**
```bash
# Try the pip fallback
./install_pip.sh

# Or minimal installation
pip install -r requirements-minimal.txt
```

### **"No module named 'torch'"**
```bash
# Make sure environment is activated
conda activate rl-lab

# Verify installation  
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### **"Framework components not registered"**
```bash
# Make sure you're in the right directory
cd /path/to/rl-lab

# Add to Python path manually
export PYTHONPATH=$(pwd):$PYTHONPATH

# Or use the activation script
source activate_env.sh
```

### **Import errors during testing**
This means dependencies aren't fully installed:
```bash
# Run validation to see what's missing
python validate_installation.py

# Install missing packages
pip install missing_package_name
```

## 🎯 Platform-Specific Notes

### **macOS/Apple Silicon**
- ✅ Uses MPS acceleration automatically
- ✅ No CUDA dependencies needed
- ✅ Native ARM64 packages when available
- ⚠️ Some packages may need Rosetta 2

### **Linux with NVIDIA GPU**
```bash
# Check CUDA version first
nvidia-smi

# Use matching environment
./setup_env.sh environment-linux-cuda.yml
```

### **Linux CPU-only**
```bash
./setup_env.sh environment-linux-cpu.yml
```

## ✅ Next Steps After Successful Installation

1. **Test the framework**: `python test_framework.py`
2. **Run example experiment**: `python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run`
3. **Read the documentation**: Check `CLAUDE.md` for framework details
4. **Implement your first algorithm**: Follow patterns in `src/algorithms/`
5. **Configure experiments**: Create custom YAML configs

## 🆘 Still Having Issues?

1. **Run the validator**: `python validate_installation.py`
2. **Check specific error messages** in the validator output
3. **Try pip fallback**: `./install_pip.sh`
4. **Use minimal install**: `pip install -r requirements-minimal.txt`
5. **Create issue** with validator output if problems persist

The setup is now **completely fixed** for macOS and should work seamlessly on your Apple Silicon Mac! 🎉