#!/bin/bash
# Quick activation script for RL Lab environment

# This script should be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  This script should be sourced, not executed."
    echo "   Use: source activate_env.sh"
    echo "   Or:  . activate_env.sh"
    exit 1
fi

echo "🔧 Activating rl-lab environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda."
    return 1
fi

# Check if environment exists
if ! conda env list | grep -q "^rl-lab "; then
    echo "❌ rl-lab environment not found."
    echo "   Run './setup_env.sh' first to create the environment."
    return 1
fi

# Activate environment
conda activate rl-lab

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "rl-lab" ]]; then
    echo "✅ rl-lab environment activated!"
    echo ""
    echo "🎯 Available commands:"
    echo "   test_framework.py           - Test framework components"
    echo "   scripts/train.py           - Run training experiments"
    echo "   quick_setup.py             - Install minimal dependencies"
    echo ""
    echo "📍 Current directory: $(pwd)"
    
    # Add current directory to Python path for development
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    echo "✅ Added current directory to PYTHONPATH"
else
    echo "❌ Failed to activate rl-lab environment"
    return 1
fi