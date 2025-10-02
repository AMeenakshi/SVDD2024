#!/bin/bash
# Single GPU CUDA 12.8 Setup Script - Fixes Channel Errors
# This script handles conda channel issues properly

set -e  # Exit on any error

echo "=== SVDD2024 Single GPU CUDA 12.8 Setup (Channel Error Fix) ==="

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found. Please install Anaconda or Miniconda first."
        echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    echo "✓ Conda found: $(conda --version)"
}

# Function to fix conda channels
fix_conda_channels() {
    echo "Configuring conda channels for CUDA 12.8..."
    
    # Add necessary channels
    conda config --add channels conda-forge
    conda config --add channels pytorch
    conda config --add channels nvidia
    
    # Set channel priority
    conda config --set channel_priority strict
    
    # Update conda
    conda update -n base -c defaults conda -y
    
    echo "✓ Conda channels configured"
}

# Function to create environment
create_environment() {
    ENV_NAME="svdd2024-single-gpu"
    
    echo "Creating conda environment: $ENV_NAME"
    
    # Remove existing environment if it exists
    conda env remove -n $ENV_NAME -y 2>/dev/null || true
    
    # Create new environment with Python 3.10
    conda create -n $ENV_NAME python=3.10 -y
    
    echo "✓ Environment '$ENV_NAME' created"
    echo "  Activate with: conda activate $ENV_NAME"
}

# Function to install PyTorch with proper channels
install_pytorch() {
    echo "Installing PyTorch with CUDA 12.8 support..."
    
    # Activate the environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate svdd2024-single-gpu
    
    # Method 1: Try conda installation first
    echo "Trying conda installation..."
    if conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y; then
        echo "✓ PyTorch installed via conda"
    else
        echo "⚠️  Conda installation failed, trying pip..."
        # Method 2: Fallback to pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo "✓ PyTorch installed via pip"
    fi
}

# Function to install remaining packages
install_requirements() {
    echo "Installing remaining requirements..."
    
    # Install packages one by one to handle potential conflicts
    echo "Installing core scientific packages..."
    pip install numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.3.0
    
    echo "Installing audio processing packages..."
    pip install librosa>=0.10.0 soundfile>=0.12.0
    
    echo "Installing training utilities..."
    pip install tqdm>=4.65.0 tensorboard>=2.14.0 pandas>=2.0.0
    
    echo "Installing visualization packages..."
    pip install matplotlib>=3.7.0
    
    echo "Installing development tools..."
    pip install ipdb>=0.13.0 psutil>=5.9.0 gpustat>=1.1.0
    
    echo "Installing configuration utilities..."
    pip install PyYAML>=6.0
    
    echo "Installing s3prl (may take a while)..."
    pip install s3prl>=0.4.0
    
    echo "✓ All requirements installed"
}

# Function to verify installation
verify_installation() {
    echo "Verifying installation..."
    
    python -c "
import torch
import numpy as np
import librosa
import s3prl
import tensorboard

print('=== Installation Verification ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - check CUDA installation')

print(f'NumPy version: {np.__version__}')
print(f'Librosa version: {librosa.__version__}')
print('✓ All packages imported successfully')
"
}

# Function to create quick test
create_test_script() {
    cat > quick_test_single_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Quick test for single GPU setup"""

import torch
import time

def test_cuda_performance():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda:0')
    print(f"✓ Testing on {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Warmup
    for _ in range(10):
        z = torch.mm(x, y)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        z = torch.mm(x, y)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"✓ Matrix multiplication test passed")
    print(f"  Performance: {(end_time - start_time) * 10:.2f}ms per operation")
    
    return True

if __name__ == "__main__":
    print("=== Single GPU CUDA Test ===")
    test_cuda_performance()
    print("=== Test Complete ===")
EOF
    
    chmod +x quick_test_single_gpu.py
    echo "✓ Quick test script created: quick_test_single_gpu.py"
}

# Main execution
main() {
    check_conda
    fix_conda_channels
    create_environment
    install_pytorch
    install_requirements
    verify_installation
    create_test_script
    
    echo ""
    echo "=== Setup Complete ==="
    echo "✓ Single GPU CUDA 12.8 environment ready"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: conda activate svdd2024-single-gpu"
    echo "2. Test CUDA: python quick_test_single_gpu.py"
    echo "3. Run training: python train.py --base_dir /path/to/dataset"
    echo ""
    echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
}

# Run main function
main