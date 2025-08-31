#!/bin/bash
# Enhanced RL Lab Environment Setup Script
# Supports macOS, Linux (CPU/CUDA), and provides intelligent fallbacks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RL Lab Environment Setup${NC}"
echo -e "Platform: $(uname -s) $(uname -m)"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Detect Python and recommend version
check_python() {
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        local version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$version" < "3.8" ]]; then
            print_error "Python $version detected. RL Lab requires Python 3.8+"
            exit 1
        elif [[ "$version" < "3.10" ]]; then
            print_warning "Python $version detected. Python 3.10+ recommended for best compatibility"
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+ first."
        exit 1
    fi
    print_info "Using Python: $PYTHON_CMD ($(${PYTHON_CMD} --version))"
}

# Enhanced platform detection with GPU check
detect_platform_and_acceleration() {
    local platform=$(uname -s)
    local arch=$(uname -m)
    
    case "$platform" in
        "Darwin")
            if [[ "$arch" == "arm64" ]]; then
                PLATFORM="macos-apple-silicon"
                ACCELERATION="MPS (Metal Performance Shaders)"
            else
                PLATFORM="macos-intel"
                ACCELERATION="CPU"
            fi
            ;;
        "Linux")
            if command -v nvidia-smi &> /dev/null; then
                PLATFORM="linux-cuda"
                ACCELERATION="NVIDIA CUDA"
                local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
                print_info "NVIDIA GPU detected (Driver: $cuda_version)"
            else
                PLATFORM="linux-cpu" 
                ACCELERATION="CPU"
            fi
            ;;
        "MINGW"*|"MSYS"*|"CYGWIN"*)
            PLATFORM="windows"
            if command -v nvidia-smi &> /dev/null; then
                ACCELERATION="NVIDIA CUDA"
            else
                ACCELERATION="CPU"
            fi
            ;;
        *)
            PLATFORM="unknown"
            ACCELERATION="CPU"
            print_warning "Platform $platform not specifically supported, using CPU-only setup"
            ;;
    esac
    
    print_info "Detected: $PLATFORM with $ACCELERATION acceleration"
}

# Setup with conda (preferred method)
setup_conda() {
    print_info "Setting up with conda..."
    
    if [ ! -f "environment.yml" ]; then
        print_error "environment.yml not found!"
        return 1
    fi
    
    # Check if environment exists
    if conda env list | grep -q "^rl-lab "; then
        print_warning "rl-lab environment already exists."
        read -p "Update existing environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env update -f environment.yml
            print_status "Environment updated successfully"
        else
            print_info "Skipping environment update"
        fi
    else
        conda env create -f environment.yml
        print_status "Environment created successfully"
    fi
    
    return 0
}

# Fallback setup with pip
setup_pip() {
    print_info "Setting up with pip..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        return 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_status "Pip installation completed"
    print_info "To activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
    
    return 0
}

# Development setup (editable install)
setup_dev() {
    print_info "Installing in development mode..."
    if conda env list | grep -q "^rl-lab "; then
        eval "$(conda shell.bash hook)"
        conda activate rl-lab
    fi
    
    if [ -f "setup.py" ]; then
        pip install -e .
        print_status "Development installation completed"
    else
        print_warning "setup.py not found, skipping development install"
    fi
}

# Run validation tests
run_tests() {
    print_info "Running validation tests..."
    
    # Basic import test
    if $PYTHON_CMD -c "import torch, numpy, gymnasium; print('✅ Core imports successful')" 2>/dev/null; then
        print_status "Core dependencies working"
    else
        print_error "Core dependency test failed"
        return 1
    fi
    
    # PyTorch acceleration test
    if $PYTHON_CMD -c "
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'🔧 PyTorch using: {device}')
if device.type == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name()}')
elif device.type == 'mps':
    print('   Apple Silicon MPS acceleration enabled')
" 2>/dev/null; then
        print_status "PyTorch acceleration test passed"
    else
        print_warning "PyTorch acceleration test failed (CPU fallback will be used)"
    fi
    
    # Framework test if available
    if [ -f "test_framework.py" ]; then
        if $PYTHON_CMD test_framework.py &>/dev/null; then
            print_status "Framework validation passed"
        else
            print_warning "Framework validation had issues (check test_framework.py output)"
        fi
    fi
}

# Main setup flow
main() {
    check_python
    detect_platform_and_acceleration
    echo ""
    
    # Try conda first, fallback to pip
    if command -v conda &> /dev/null; then
        print_info "Conda detected - using conda setup (recommended)"
        if setup_conda; then
            setup_dev
        else
            print_warning "Conda setup failed, falling back to pip..."
            setup_pip
        fi
    else
        print_info "Conda not found - using pip setup"
        setup_pip
    fi
    
    echo ""
    print_info "Running validation tests..."
    run_tests
    
    echo ""
    print_status "🎉 Setup complete!"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    if command -v conda &> /dev/null && conda env list | grep -q "^rl-lab "; then
        echo "  1. Activate environment: conda activate rl-lab"
    else
        echo "  1. Activate environment: source venv/bin/activate"
    fi
    echo "  2. Run a test experiment: python scripts/train.py --config configs/experiments/test_cartpole.yaml --dry-run"
    echo ""
    echo -e "${BLUE}Troubleshooting:${NC}"
    echo "  • If PyTorch isn't using GPU/MPS, check drivers or system compatibility"
    echo "  • For environment issues, delete and recreate: conda env remove -n rl-lab"
    echo "  • For pip issues, delete venv folder and rerun script"
    echo "  • Check system requirements in the README"
}

# Handle command line arguments
case "${1:-}" in
    "--pip-only")
        check_python
        setup_pip
        ;;
    "--test-only")
        run_tests
        ;;
    "--help"|"-h")
        echo "RL Lab Environment Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --pip-only     Use pip instead of conda"
        echo "  --test-only    Only run validation tests"
        echo "  --help, -h     Show this help message"
        echo ""
        echo "Default: Auto-detect platform and use conda if available, otherwise pip"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac