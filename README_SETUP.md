# RL Lab Setup Guide

Welcome to the RL Lab framework! This guide will help you set up your conda environment and get started with reinforcement learning experiments.

## Quick Start (Recommended)

### 1. Create Conda Environment
```bash
# Create environment from yml file
conda env create -f environment.yml

# Or use the setup script
./setup_env.sh
```

### 2. Activate Environment
```bash
# Standard conda activation
conda activate rl-lab

# Or use the activation script (adds PYTHONPATH automatically)
source activate_env.sh
```

### 3. Test Installation
```bash
# Test framework components
python test_framework.py

# Test with a dry run
python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run
```

## Alternative Setup Methods

### Option 1: Pure Conda (Recommended)
Uses conda for most packages, pip only for RL-specific packages:
```bash
conda env create -f environment.yml
conda activate rl-lab
```

### Option 2: Pip with Requirements
If you prefer pip or conda is unavailable:
```bash
conda create -n rl-lab python=3.10
conda activate rl-lab
pip install -r requirements.txt
```

### Option 3: Minimal Quick Setup
For testing or CI environments:
```bash
python quick_setup.py
```

## What Gets Installed

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Gymnasium**: OpenAI Gym environments 
- **NumPy/SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### RL-Specific
- **DM-Control**: DeepMind Control Suite
- **TensorBoard**: Training visualization
- **WandB**: Experiment tracking
- **Optuna**: Hyperparameter optimization

### Development Tools
- **PyTest**: Testing framework
- **Black**: Code formatting
- **Flake8/MyPy**: Code linting and type checking
- **JupyterLab**: Interactive development

## Testing Your Installation

### Level 1: Component Tests
```bash
python test_framework.py
```
**What it tests:**
- ✅ All framework components import correctly
- ✅ Registry system finds algorithms, networks, environments  
- ✅ Configuration system loads and validates YAML files
- ✅ Basic environment interaction works

### Level 2: Integration Tests  
```bash
python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run
```
**What it tests:**
- ✅ Complete experiment setup (without training)
- ✅ All components work together correctly
- ✅ Configuration validation catches errors
- ✅ Experiment directory structure creation

### Level 3: Full Training Test
```bash
python scripts/train.py --config configs/experiments/test_cartpole.yaml --total-timesteps 1000
```
**What it tests:**
- ✅ Actual training loop execution
- ✅ Checkpoint saving and loading
- ✅ Metrics collection and logging
- ✅ Complete workflow from start to finish

## Troubleshooting Common Issues

### "No module named 'torch'"
- Make sure you activated the conda environment: `conda activate rl-lab`
- Verify installation: `conda list torch`

### "Gymnasium environment not found"
- Some environments require additional dependencies
- For Atari: `pip install "gymnasium[atari]"`
- For Box2D: `pip install "gymnasium[box2d]"`

### "CUDA not available" 
- For GPU support, ensure you have compatible NVIDIA drivers
- Modify `environment.yml` to match your CUDA version
- For CPU-only: remove `pytorch-cuda` from environment.yml

### Import errors in test_framework.py
- Check that all dependencies installed successfully
- Verify PYTHONPATH includes current directory
- Try `source activate_env.sh` instead of `conda activate rl-lab`

## Environment Management Tips

### Daily Workflow
```bash
# Start your day
source activate_env.sh

# Run experiments
python scripts/train.py --config my_config.yaml

# Deactivate when done
conda deactivate
```

### Updating Dependencies
```bash
# Update environment from yml file
conda env update -f environment.yml

# Add new packages
conda install new_package
# or
pip install new_package
```

### Environment Info
```bash
# List environments
conda env list

# Show installed packages
conda list

# Export current environment
conda env export > current_environment.yml
```

## Next Steps

Once your environment is set up and tests pass:

1. **Read the Framework Overview**: Check `CLAUDE.md` for architecture details
2. **Try the Examples**: Run the test experiments in `configs/experiments/`
3. **Implement Your Algorithm**: Follow patterns in `src/algorithms/`
4. **Configure Experiments**: Create your own YAML configs
5. **Scale Up**: Add more environments and algorithms as needed

## Need Help?

- 🐛 **Issues**: Check the troubleshooting section above
- 📚 **Documentation**: See `CLAUDE.md` for framework details  
- 🧪 **Testing**: Run `python test_framework.py` for diagnostics
- 💡 **Examples**: Look at `configs/experiments/` for config templates