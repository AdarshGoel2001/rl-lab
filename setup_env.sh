#!/bin/bash
# Conda Environment Setup Script for RL Lab with Platform Detection

set -e  # Exit on any error

echo "🚀 Setting up RL Lab conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Detect platform and choose appropriate environment file
detect_platform() {
    local env_file="environment.yml"
    local platform=$(uname -s)
    local arch=$(uname -m)
    
    case "$platform" in
        "Darwin")
            env_file="environment-macos.yml"
            echo "🍎 Detected macOS ($arch)"
            ;;
        "Linux")
            if command -v nvidia-smi &> /dev/null; then
                env_file="environment-linux-cuda.yml"
                echo "🐧 Detected Linux with NVIDIA GPU"
                echo "   CUDA version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
            else
                env_file="environment-linux-cpu.yml" 
                echo "🐧 Detected Linux (CPU-only)"
            fi
            ;;
        *)
            echo "⚠️  Platform $platform not specifically supported, using default environment.yml"
            ;;
    esac
    
    echo "$env_file"
}

# Allow manual override of environment file
ENV_FILE=${1:-$(detect_platform)}

echo "📋 Using environment file: $ENV_FILE"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file $ENV_FILE not found!"
    echo "   Available files:"
    ls -1 environment*.yml 2>/dev/null || echo "   No environment files found"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^rl-lab "; then
    echo "⚠️  rl-lab environment already exists."
    read -p "Do you want to update it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Updating existing environment..."
        conda env update -f "$ENV_FILE"
    else
        echo "ℹ️  Skipping environment creation."
    fi
else
    echo "📦 Creating new rl-lab environment..."
    conda env create -f "$ENV_FILE"
fi

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate environment:     conda activate rl-lab"
echo "   2. Run tests:               python test_framework.py"
echo "   3. Try dry run:             python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run"
echo ""
echo "💡 Tips:"
echo "   • Use 'source activate_env.sh' for activation with PYTHONPATH setup"
echo "   • Use './setup_env.sh environment-linux-cuda.yml' to force specific environment"
echo "   • If installation fails, try pip fallback: pip install -r requirements.txt"