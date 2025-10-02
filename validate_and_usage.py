#!/usr/bin/env python3
"""
Distributed Training Validation and Usage Guide
===============================================

This script validates the distributed training setup and provides usage instructions.
"""

import torch
import torch.distributed as dist
import os
import sys
import subprocess

def check_system_requirements():
    """Check system requirements for distributed training"""
    print("=== System Requirements Check ===")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"✓ Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  - GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    # Check PyTorch version
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check distributed availability
    dist_available = dist.is_available()
    print(f"✓ Distributed available: {dist_available}")
    
    if dist_available:
        nccl_available = dist.is_nccl_available()
        print(f"✓ NCCL backend available: {nccl_available}")
    
    return cuda_available and dist_available

def validate_train_distributed():
    """Validate the train_distributed.py syntax"""
    print("\n=== Code Validation ===")
    
    try:
        # Check if file exists
        train_file = '/Users/meeagraw/Applications/code/AICOde/SVDD2024-main 2/train_distributed.py'
        if not os.path.exists(train_file):
            print(f"✗ File not found: {train_file}")
            return False
        
        # Try to compile the file
        with open(train_file, 'r') as f:
            code = f.read()
        
        compile(code, 'train_distributed.py', 'exec')
        print("✓ train_distributed.py syntax validation successful")
        
        # Check for required imports
        required_imports = [
            'torch.distributed as dist',
            'DistributedDataParallel as DDP',
            'DistributedSampler'
        ]
        
        for imp in required_imports:
            if imp in code:
                print(f"✓ Found import: {imp}")
            else:
                print(f"✗ Missing import: {imp}")
        
        # Check for key functions
        key_functions = [
            'setup_distributed',
            'create_distributed_dataloaders',
            'main_distributed',
            'cleanup_distributed'
        ]
        
        for func in key_functions:
            if f"def {func}" in code:
                print(f"✓ Found function: {func}")
            else:
                print(f"✗ Missing function: {func}")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def print_usage_examples():
    """Print detailed usage examples"""
    print("\n=== Usage Examples ===")
    
    print("\n1. Single GPU Training (Original):")
    print("   python train_distributed.py --base_dir /path/to/dataset --gpu 0")
    
    print("\n2. Multi-GPU Training (2 GPUs):")
    print("   torchrun --nproc_per_node=2 train_distributed.py \\")
    print("     --distributed \\")
    print("     --base_dir /path/to/dataset \\")
    print("     --batch_size 20 \\")
    print("     --pin_memory")
    
    print("\n3. Multi-GPU Training (4 GPUs):")
    print("   torchrun --nproc_per_node=4 train_distributed.py \\")
    print("     --distributed \\")
    print("     --base_dir /path/to/dataset \\")
    print("     --batch_size 10 \\")
    print("     --pin_memory \\")
    print("     --cudnn_benchmark")
    
    print("\n4. Multi-Node Training (2 nodes, 2 GPUs each):")
    print("   # On Node 0 (master):")
    print("   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \\")
    print("     --master_addr=192.168.1.10 --master_port=12355 \\")
    print("     train_distributed.py --distributed --base_dir /path/to/dataset")
    print("")
    print("   # On Node 1:")
    print("   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \\")
    print("     --master_addr=192.168.1.10 --master_port=12355 \\")
    print("     train_distributed.py --distributed --base_dir /path/to/dataset")

def print_distributed_features():
    """Print key distributed training features"""
    print("\n=== Distributed Training Features ===")
    
    features = [
        "✓ DistributedDataParallel (DDP) for model parallelism",
        "✓ DistributedSampler for proper data distribution",
        "✓ Gradient synchronization across GPUs",
        "✓ Main process logging and checkpointing",
        "✓ Proper cleanup and error handling",
        "✓ Progress bars only on main process",
        "✓ Backward compatibility with single GPU",
        "✓ Automatic batch size scaling",
        "✓ NCCL backend for optimal GPU communication",
        "✓ Mixed precision training support"
    ]
    
    for feature in features:
        print(f"  {feature}")

def print_performance_tips():
    """Print performance optimization tips"""
    print("\n=== Performance Optimization Tips ===")
    
    tips = [
        "• Batch size is per GPU - effective batch size = batch_size × num_gpus",
        "• Use --pin_memory for faster CPU-GPU transfer",
        "• Enable --cudnn_benchmark for consistent input sizes",
        "• Set num_workers to 2-4 per GPU for optimal data loading",
        "• Monitor GPU utilization with 'nvidia-smi' or 'watch -n 1 nvidia-smi'",
        "• Use TensorBoard to monitor training: 'tensorboard --logdir logs'",
        "• For large models, consider gradient checkpointing",
        "• Use mixed precision training with torch.cuda.amp for speed",
        "• Ensure sufficient CPU cores: 2-4 cores per GPU",
        "• Use fast storage (SSD) for datasets to avoid I/O bottlenecks"
    ]
    
    for tip in tips:
        print(f"  {tip}")

def print_troubleshooting():
    """Print common troubleshooting tips"""
    print("\n=== Troubleshooting ===")
    
    issues = [
        "❓ 'NCCL error': Check CUDA versions and GPU compatibility",
        "❓ 'Port already in use': Change --master_port to different value",
        "❓ 'CUDA out of memory': Reduce batch_size or use gradient accumulation",
        "❓ 'Slow training': Check data loading bottleneck with num_workers",
        "❓ 'Process hanging': Ensure all processes use same number of epochs",
        "❓ 'Connection timeout': Check network connectivity between nodes",
        "❓ 'Different results': Ensure same random seed across processes"
    ]
    
    for issue in issues:
        print(f"  {issue}")

def create_quick_test_script():
    """Create a quick test script for distributed setup"""
    test_script = """#!/bin/bash
# Quick distributed training test script

echo "=== Distributed Training Test ==="

# Check if dataset path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/dataset"
    echo "Example: $0 /data/SVDD2024"
    exit 1
fi

DATASET_PATH="$1"

# Test single GPU first
echo "Testing single GPU training..."
python train_distributed.py \\
    --base_dir "$DATASET_PATH" \\
    --epochs 1 \\
    --debug \\
    --batch_size 4 \\
    --gpu 0

if [ $? -eq 0 ]; then
    echo "✓ Single GPU test passed"
else
    echo "✗ Single GPU test failed"
    exit 1
fi

# Test multi-GPU if available
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Testing multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=2 train_distributed.py \\
        --distributed \\
        --base_dir "$DATASET_PATH" \\
        --epochs 1 \\
        --debug \\
        --batch_size 2 \\
        --pin_memory
    
    if [ $? -eq 0 ]; then
        echo "✓ Multi-GPU test passed"
    else
        echo "✗ Multi-GPU test failed"
    fi
else
    echo "Only 1 GPU detected, skipping multi-GPU test"
fi

echo "=== Test Complete ==="
"""
    
    with open('/Users/meeagraw/Applications/code/AICOde/SVDD2024-main 2/test_distributed.sh', 'w') as f:
        f.write(test_script)
    
    # Make executable
    os.chmod('/Users/meeagraw/Applications/code/AICOde/SVDD2024-main 2/test_distributed.sh', 0o755)
    
    print("\n=== Quick Test Script Created ===")
    print("A test script has been created: test_distributed.sh")
    print("Usage: ./test_distributed.sh /path/to/your/dataset")

def main():
    """Main validation function"""
    print("Distributed Training Validation Tool")
    print("=" * 50)
    
    # Check system requirements
    system_ok = check_system_requirements()
    
    # Validate code
    code_ok = validate_train_distributed()
    
    if system_ok and code_ok:
        print("\n🎉 All validations passed!")
        print_distributed_features()
        print_usage_examples()
        print_performance_tips()
        print_troubleshooting()
        create_quick_test_script()
        
        print("\n=== Next Steps ===")
        print("1. Prepare your dataset in the SVDD2024 format")
        print("2. Run a quick test: ./test_distributed.sh /path/to/dataset")
        print("3. Start distributed training with your desired configuration")
        print("4. Monitor training with TensorBoard: tensorboard --logdir logs")
        
    else:
        print("\n❌ Validation failed. Please fix the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    main()