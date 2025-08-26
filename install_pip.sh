#!/bin/bash
# Pip Installation Script for RL Lab (when conda is not available)

set -e  # Exit on any error

echo "🚀 Setting up RL Lab with pip..."

# Detect platform and choose appropriate requirements file
detect_requirements() {
    local req_file="requirements.txt"
    local platform=$(uname -s)
    local arch=$(uname -m)
    
    case "$platform" in
        "Darwin")
            req_file="requirements-macos.txt"
            echo "🍎 Detected macOS ($arch) - using macOS-optimized requirements"
            ;;
        "Linux")
            echo "🐧 Detected Linux - using default requirements"
            req_file="requirements.txt"
            ;;
        *)
            echo "⚠️  Platform $platform - using default requirements"
            ;;
    esac
    
    echo "$req_file"
}

# Allow manual override of requirements file
REQ_FILE=${1:-$(detect_requirements)}

echo "📋 Using requirements file: $REQ_FILE"

if [ ! -f "$REQ_FILE" ]; then
    echo "❌ Requirements file $REQ_FILE not found!"
    echo "   Available files:"
    ls -1 requirements*.txt 2>/dev/null || echo "   No requirements files found"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required, found $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Recommend virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  No virtual environment detected."
    echo "   Recommended: create a virtual environment first"
    echo "   python3 -m venv rl-lab-env"
    echo "   source rl-lab-env/bin/activate"
    echo ""
    read -p "Continue without virtual environment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "ℹ️  Aborted. Create a virtual environment and try again."
        exit 0
    fi
fi

# Upgrade pip first
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📦 Installing packages from $REQ_FILE..."
python3 -m pip install -r "$REQ_FILE"

echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Test installation:       python test_framework.py"
echo "   2. Try dry run:             python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run"
echo ""
echo "💡 Tips:"
echo "   • Make sure to activate your virtual environment before using the framework"
echo "   • Use 'export PYTHONPATH=\$(pwd):\$PYTHONPATH' to add current directory to Python path"