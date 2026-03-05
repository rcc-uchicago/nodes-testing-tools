#!/usr/bin/env python3
"""
Multi-GPU System Diagnostic Tool

Checks your multi-GPU setup and identifies potential issues.

Usage:
    python check_multi_gpu.py
"""

import torch
import sys
import subprocess
import os

def check_cuda_available():
    """Check if CUDA is available"""
    print("="*80)
    print("CUDA Availability Check")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available")
        print("   PyTorch cannot access NVIDIA GPUs")
        print("   Check: nvidia-smi and PyTorch CUDA installation")
        return False
    else:
        print("✓ CUDA is available")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        return True

def check_gpu_count():
    """Check number of GPUs"""
    print("\n" + "="*80)
    print("GPU Count Check")
    print("="*80)
    
    gpu_count = torch.cuda.device_count()
    print(f"Detected GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠️  Multi-GPU benchmarks require at least 2 GPUs")
        print("   You can still run single-GPU benchmarks")
        return False
    else:
        print(f"✓ {gpu_count} GPUs detected (sufficient for multi-GPU tests)")
        return True

def check_gpu_info():
    """Display GPU information"""
    print("\n" + "="*80)
    print("GPU Information")
    print("="*80)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")

def check_nccl():
    """Check NCCL availability"""
    print("\n" + "="*80)
    print("NCCL Backend Check")
    print("="*80)
    
    try:
        import torch.distributed as dist
        nccl_available = dist.is_nccl_available()
        
        if nccl_available:
            print("✓ NCCL backend is available")
            print("  Multi-GPU communication can use optimized NCCL")
        else:
            print("❌ NCCL backend is NOT available")
            print("   Multi-GPU benchmarks may fail")
            print("   Reinstall PyTorch with NCCL support")
        
        return nccl_available
    except Exception as e:
        print(f"❌ Error checking NCCL: {e}")
        return False

def check_gpu_topology():
    """Check GPU topology using nvidia-smi"""
    print("\n" + "="*80)
    print("GPU Topology Check")
    print("="*80)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(result.stdout)
            
            # Analyze topology
            if "NV" in result.stdout:
                print("\n✓ NVLink detected between some GPUs!")
                print("  Expected multi-GPU bandwidth: 400-900 GB/s")
            elif "PIX" in result.stdout or "PHB" in result.stdout:
                print("\n⚠️  GPUs connected via PCIe (no NVLink)")
                print("  Expected multi-GPU bandwidth: 20-50 GB/s")
            
            if "SYS" in result.stdout:
                print("\n⚠️  Some GPUs are in different NUMA nodes (SYS)")
                print("  This may reduce performance for GPU-to-GPU transfers")
        else:
            print("Could not run nvidia-smi topo command")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found in PATH")
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi topo command timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_basic_multi_gpu():
    """Test basic multi-GPU tensor operations"""
    print("\n" + "="*80)
    print("Basic Multi-GPU Test")
    print("="*80)
    
    if torch.cuda.device_count() < 2:
        print("⚠️  Need at least 2 GPUs for this test")
        return False
    
    try:
        # Test GPU 0
        print("\nTesting GPU 0...")
        x0 = torch.randn(100, 100, device='cuda:0')
        y0 = x0 @ x0
        torch.cuda.synchronize()
        print("✓ GPU 0: Operations work")
        
        # Test GPU 1
        print("\nTesting GPU 1...")
        x1 = torch.randn(100, 100, device='cuda:1')
        y1 = x1 @ x1
        torch.cuda.synchronize()
        print("✓ GPU 1: Operations work")
        
        # Test GPU-to-GPU copy
        print("\nTesting GPU 0 → GPU 1 copy...")
        x1_copy = x0.to('cuda:1')
        torch.cuda.synchronize()
        print("✓ GPU-to-GPU copy works")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        return False

def test_distributed_init():
    """Test distributed initialization"""
    print("\n" + "="*80)
    print("Distributed Initialization Test")
    print("="*80)
    
    if torch.cuda.device_count() < 2:
        print("⚠️  Need at least 2 GPUs for this test")
        return False
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    try:
        import torch.distributed as dist
        
        print("\nAttempting to initialize distributed backend...")
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print("✓ Distributed backend initialized successfully")
        print("  Multi-GPU benchmarks should work")
        
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"❌ Distributed initialization failed: {e}")
        print("\nPossible issues:")
        print("  - NCCL not properly installed")
        print("  - GPU driver issues")
        print("  - Firewall blocking localhost communication")
        return False

def print_recommendations():
    """Print recommendations based on system"""
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count == 0:
        print("\n❌ No GPUs detected")
        print("   - Check nvidia-smi output")
        print("   - Reinstall NVIDIA drivers")
        print("   - Install CUDA-enabled PyTorch")
        
    elif gpu_count == 1:
        print("\n✓ Single GPU detected")
        print("   Run benchmark with:")
        print("   python gpu_benchmark.py --preset small")
        
    else:
        print(f"\n✓ {gpu_count} GPUs detected")
        print("   Run single-GPU benchmark:")
        print("   python gpu_benchmark.py --preset medium")
        print("\n   Run multi-GPU benchmark:")
        print("   python gpu_benchmark.py --multi-gpu")
        
        # Check topology
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "NV" in result.stdout:
                print("\n   NVLink detected - expect high GPU-to-GPU bandwidth!")
            else:
                print("\n   PCIe connection - expect moderate GPU-to-GPU bandwidth")
                
        except:
            pass

def main():
    print("\n" + "="*80)
    print("Multi-GPU System Diagnostic Tool")
    print("="*80)
    print("\nThis tool checks your system for multi-GPU benchmark compatibility\n")
    
    # Run all checks
    results = []
    
    results.append(("CUDA Available", check_cuda_available()))
    
    if results[0][1]:  # Only continue if CUDA is available
        results.append(("Multiple GPUs", check_gpu_count()))
        check_gpu_info()
        results.append(("NCCL Available", check_nccl()))
        check_gpu_topology()
        results.append(("Basic Multi-GPU", test_basic_multi_gpu()))
        results.append(("Distributed Init", test_distributed_init()))
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓" if passed else "❌"
        print(f"{status} {test_name}")
    
    print_recommendations()
    
    print("\n" + "="*80)
    print("Diagnostic Complete")
    print("="*80)
    
    # Exit code
    all_passed = all(result[1] for result in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
