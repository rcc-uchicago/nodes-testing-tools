# NVIDIA GPU ML Benchmark Suite

Comprehensive PyTorch-based benchmark to distinguish between NVIDIA GPU architectures including A100, H100, L40S, and B200/B300.

## Features

The benchmark tests key ML workload characteristics:

- **Matrix Multiplication (GEMM)**: Core operation for dense layers, tests tensor core performance
- **2D Convolution**: Tests spatial operations common in CNNs
- **Multi-Head Attention**: Tests transformer-critical operations
- **Memory Bandwidth**: Tests data transfer capabilities
- **Mixed Precision**: Tests FP32, FP16, and BF16 performance

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
# or for latest CUDA version:
# pip install torch
```

## Usage

### Basic Usage

```bash
# Default benchmark (requires ~20-30GB memory)
python gpu_benchmark.py

# For GPUs with limited memory, use presets:
python gpu_benchmark.py --preset small    # 8-16GB GPUs
python gpu_benchmark.py --preset medium   # 24-40GB GPUs
python gpu_benchmark.py --preset large    # 40-80GB GPUs
python gpu_benchmark.py --preset xlarge   # 80GB+ GPUs
```

### Recommended Presets by GPU

| GPU Model | Memory | Preset | Command |
|-----------|--------|--------|---------|
| GTX 1080 Ti, Quadro GP100 | 11-16GB | `small` | `python gpu_benchmark.py --preset small` |
| RTX 3060, RTX 3080 | 12-24GB | `small` | `python gpu_benchmark.py --preset small` |
| RTX 3090, RTX 4090, A10 | 24GB | `medium` | `python gpu_benchmark.py --preset medium` |
| A100 40GB, L40 | 40-48GB | `medium` | `python gpu_benchmark.py --preset medium` |
| A100 80GB, H100, L40S | 48-80GB | `large` | `python gpu_benchmark.py --preset large` |
| H100 80GB, B200 | 80GB+ | `xlarge` | `python gpu_benchmark.py --preset xlarge` |

### Advanced Options

```bash
# Specify GPU device (single GPU mode)
python gpu_benchmark.py --device 0

# Custom output file
python gpu_benchmark.py --output my_results.json

# Quick benchmark (fewer iterations)
python gpu_benchmark.py --quick

# Scale up all data sizes by 2x (double all dimensions)
python gpu_benchmark.py --scale 2.0

# Scale down by 50% (useful for smaller GPUs)
python gpu_benchmark.py --scale 0.5

# Custom matrix size for GEMM tests
python gpu_benchmark.py --matmul-size 32768

# Custom batch size for convolution and attention
python gpu_benchmark.py --batch-size 256

# Custom sequence length for attention tests
python gpu_benchmark.py --seq-length 8192

# Combine multiple options
python gpu_benchmark.py --scale 1.5 --batch-size 128 --output scaled_results.json
```

### Multi-GPU Benchmarking

```bash
# Run on all available GPUs
python gpu_benchmark.py --multi-gpu

# Specify number of GPUs to use
python gpu_benchmark.py --multi-gpu --num-gpus 4

# Multi-GPU with custom scaling
python gpu_benchmark.py --multi-gpu --scale 2.0

# Multi-GPU with all options
python gpu_benchmark.py --multi-gpu --num-gpus 8 --scale 1.5 --output multi_gpu_results.json
```

**Multi-GPU Tests Include:**
- **Data Parallel Training**: Distributed training with gradient synchronization
- **All-Reduce Bandwidth**: Collective communication performance (critical for distributed training)
- **Peer-to-Peer (P2P) Transfer**: Direct GPU-to-GPU bandwidth (NVLink/PCIe)

These tests help identify:
- NVLink vs PCIe interconnect performance
- Gradient synchronization overhead
- Multi-GPU scaling efficiency

### Scaling Guide

**Presets (Recommended):**
- `--preset small`: 0.35x scale for 8-16GB GPUs
- `--preset medium`: 0.75x scale for 24-40GB GPUs  
- `--preset large`: 1.5x scale for 40-80GB GPUs
- `--preset xlarge`: 2.5x scale for 80GB+ GPUs

**Manual Scaling:**
The `--scale` parameter multiplies all tensor dimensions:
- `--scale 0.25`: Quarter size (8GB GPUs)
- `--scale 0.35`: Small size (12-16GB GPUs) - **Good for Quadro GP100**
- `--scale 0.5`: Half size (16-24GB GPUs)
- `--scale 0.75`: Medium size (24-40GB GPUs)
- `--scale 1.0`: Default sizes (40GB+ GPUs)
- `--scale 1.5`: Large size (60-80GB GPUs)
- `--scale 2.0`: XL size (80GB+ GPUs)

Individual parameters override the scale factor:
- `--matmul-size`: Sets matrix dimension for GEMM (MxM square matrices)
- `--batch-size`: Sets batch size for convolution and attention
- `--seq-length`: Sets sequence length for attention tests

**Memory Requirements by Scale:**
- 0.25x: ~4-6 GB
- 0.35x: ~6-10 GB (**Quadro GP100 16GB: Use this**)
- 0.5x: ~8-12 GB
- 0.75x: ~15-20 GB
- 1.0x: ~20-30 GB (default)
- 1.5x: ~40-50 GB
- 2.0x: ~60-80 GB

## Expected Performance Characteristics

### Single GPU Performance

### Older/Budget GPUs

#### Quadro GP100 (16GB)

**Specifications:**
- Architecture: Pascal (2016)
- CUDA Cores: 3,584
- Memory: 16 GB HBM2
- Memory Bandwidth: 717 GB/s
- FP16: ~21 TFLOPS (with FP16 storage, FP32 compute)
- FP32: ~10.3 TFLOPS

**Expected Results (with --preset small):**
- FP32 GEMM: 8-10 TFLOPS
- FP16 GEMM: 8-10 TFLOPS (Pascal doesn't have tensor cores, so FP16 is similar to FP32)
- Memory Bandwidth: 600-700 GB/s (device-to-device)
- **PCIe Gen3 x16**: H2D/D2H: 10-13 GB/s
- **Note**: Pascal lacks tensor cores, so you won't see the 10-20x FP16 speedup of modern GPUs

**Key Differentiator from Modern GPUs:**
- No tensor cores (introduced in Volta/V100)
- FP16 performance similar to FP32
- Much older architecture (2016 vs 2020+)

### A100 (80GB PCIe/SXM)

**Specifications:**
- CUDA Cores: 6,912
- Tensor Cores: 432 (3rd gen)
- Memory: 40/80 GB HBM2e
- Memory Bandwidth: 1,555/2,039 GB/s
- FP16 Tensor: ~312 TFLOPS
- FP32: ~19.5 TFLOPS

**Expected Results:**
- FP16 GEMM: 250-300 TFLOPS (large matrices)
- FP32 GEMM: 15-19 TFLOPS
- BF16 GEMM: 250-300 TFLOPS
- Memory Bandwidth: 1,400-1,900 GB/s
- Attention (FP16, large): 200-250 TFLOPS
- **PCIe Variant**: H2D/D2H: 22-27 GB/s (Gen4)
- **SXM Variant**: H2D/D2H: 40-50 GB/s (NVLink-to-CPU on DGX)

### H100 (80GB PCIe/SXM)

**Specifications:**
- CUDA Cores: 14,592 (SXM) / 14,592 (PCIe)
- Tensor Cores: 456 (4th gen)
- Memory: 80 GB HBM3
- Memory Bandwidth: 2,000 GB/s (PCIe) / 3,350 GB/s (SXM)
- FP16 Tensor: ~989 TFLOPS (SXM with sparsity)
- FP32: ~67 TFLOPS (SXM)

**Expected Results:**
- FP16 GEMM: 700-900 TFLOPS (large matrices, SXM)
- FP32 GEMM: 50-65 TFLOPS
- BF16 GEMM: 700-900 TFLOPS
- Memory Bandwidth: 2,800-3,200 GB/s (SXM)
- Attention (FP16, large): 600-800 TFLOPS
- **Key Differentiator**: 3-4x faster than A100 on FP16/BF16
- **PCIe Variant**: H2D/D2H: 22-27 GB/s (Gen4)
- **SXM Variant**: H2D/D2H: 60-80 GB/s (NVLink-to-CPU on DGX/HGX)

### L40S

**Specifications:**
- CUDA Cores: 18,176
- Tensor Cores: 568 (4th gen)
- Memory: 48 GB GDDR6
- Memory Bandwidth: 864 GB/s
- FP16 Tensor: ~362 TFLOPS
- FP32: ~91 TFLOPS

**Expected Results:**
- FP16 GEMM: 280-340 TFLOPS (large matrices)
- FP32 GEMM: 70-85 TFLOPS
- BF16 GEMM: 280-340 TFLOPS
- Memory Bandwidth: 750-850 GB/s
- Attention (FP16, large): 250-300 TFLOPS
- **Key Differentiator**: Similar to A100 on tensor ops, but much higher FP32, lower memory bandwidth
- **PCIe Only**: H2D/D2H: 22-27 GB/s (Gen4) - No SXM variant available

### B200/B300 (Blackwell)

**Specifications (B200):**
- CUDA Cores: ~20,000+ (estimated)
- Tensor Cores: 5th gen
- Memory: 192 GB HBM3e
- Memory Bandwidth: ~8,000 GB/s
- FP16 Tensor: ~2,250 TFLOPS (estimated)
- FP4 Tensor: ~9,000 TFLOPS (estimated)

**Expected Results:**
- FP16 GEMM: 1,800-2,200 TFLOPS (large matrices)
- FP32 GEMM: 100-140 TFLOPS
- BF16 GEMM: 1,800-2,200 TFLOPS
- Memory Bandwidth: 7,000-8,000 GB/s
- Attention (FP16, large): 1,500-2,000 TFLOPS
- **Key Differentiators**: 2-3x faster than H100, massive memory bandwidth

### Multi-GPU Interconnect Performance

#### NVLink Generations

**NVLink 3.0 (A100):**
- 600 GB/s bidirectional per GPU (12 links × 25 GB/s per link)
- All-Reduce (100MB): ~40-50 GB/s effective
- P2P Transfer: 400-500 GB/s

**NVLink 4.0 (H100):**
- 900 GB/s bidirectional per GPU (18 links × 25 GB/s per link)
- All-Reduce (100MB): ~70-80 GB/s effective
- P2P Transfer: 600-700 GB/s

**NVLink 5.0 (B200/B300):**
- 1,800 GB/s bidirectional per GPU (estimated)
- All-Reduce (100MB): ~150-180 GB/s effective (estimated)
- P2P Transfer: 1,200-1,500 GB/s (estimated)

#### PCIe Performance

**PCIe Gen4 x16:**
- Theoretical: 32 GB/s bidirectional
- All-Reduce (100MB): ~10-15 GB/s effective
- P2P Transfer: 20-25 GB/s

**PCIe Gen5 x16:**
- Theoretical: 64 GB/s bidirectional
- All-Reduce (100MB): ~20-30 GB/s effective
- P2P Transfer: 40-50 GB/s

#### Identifying Interconnect Type

Compare your **All-Reduce** and **P2P** results:

- **NVLink 3.0 (A100)**: All-Reduce >40 GB/s, P2P >400 GB/s
- **NVLink 4.0 (H100)**: All-Reduce >70 GB/s, P2P >600 GB/s  
- **NVLink 5.0 (B200)**: All-Reduce >150 GB/s, P2P >1200 GB/s
- **PCIe Gen4**: All-Reduce 10-15 GB/s, P2P 20-25 GB/s
- **PCIe Gen5**: All-Reduce 20-30 GB/s, P2P 40-50 GB/s

## Interpreting Results

### PCIe vs SXM Interface Identification

The host-device bandwidth tests help distinguish between PCIe and SXM/NVLink-connected GPUs:

#### PCIe Interface Bandwidth

**PCIe Gen3 x16:**
- Theoretical: 15.75 GB/s per direction
- Host-to-Device (H2D): 10-13 GB/s
- Device-to-Host (D2H): 10-13 GB/s
- Device-to-Device: Matches GPU memory bandwidth

**PCIe Gen4 x16:**
- Theoretical: 31.5 GB/s per direction
- Host-to-Device (H2D): 22-27 GB/s
- Device-to-Host (D2H): 22-27 GB/s
- Device-to-Device: Matches GPU memory bandwidth

**PCIe Gen5 x16:**
- Theoretical: 63 GB/s per direction
- Host-to-Device (H2D): 50-60 GB/s
- Device-to-Host (D2H): 50-60 GB/s
- Device-to-Device: Matches GPU memory bandwidth

#### SXM Interface (NVLink to CPU)

**SXM4 (A100 w/ NVLink to CPU):**
- Host-to-Device (H2D): 40-50 GB/s
- Device-to-Host (D2H): 40-50 GB/s
- Device-to-Device: 1,500-1,900 GB/s (HBM2e)

**SXM5 (H100 w/ NVLink to CPU):**
- Host-to-Device (H2D): 60-80 GB/s
- Device-to-Host (D2H): 60-80 GB/s
- Device-to-Device: 2,800-3,200 GB/s (HBM3)

**Note:** SXM systems with NVLink-to-CPU connections (like DGX systems) show significantly higher host-device bandwidth than PCIe. Most consumer/workstation systems use PCIe even with SXM-form-factor GPUs.

#### Quick Reference Table

| Interface | H2D Bandwidth | D2H Bandwidth | Typical Systems |
|-----------|---------------|---------------|-----------------|
| PCIe Gen3 x16 | 10-13 GB/s | 10-13 GB/s | Older workstations, Quadro GP100 |
| PCIe Gen4 x16 | 22-27 GB/s | 22-27 GB/s | Modern workstations, A100 PCIe, H100 PCIe |
| PCIe Gen5 x16 | 50-60 GB/s | 50-60 GB/s | Latest workstations (2023+) |
| SXM + NVLink-to-CPU | 40-80 GB/s | 40-80 GB/s | DGX A100, DGX H100, HGX systems |

### Architecture Identification Guide

1. **H100 vs A100**: 
   - H100 should show 3-4x higher FP16/BF16 performance
   - H100 has 60%+ higher memory bandwidth (SXM)
   - H100 shows 3x+ higher FP32 performance

2. **L40S vs A100**:
   - Similar FP16/BF16 tensor performance
   - L40S has 4-5x higher FP32 performance
   - L40S has ~40% lower memory bandwidth
   - L40S uses GDDR6 (lower bandwidth) vs HBM2e

3. **B200/B300 vs H100**:
   - B200 should show 2-3x higher FP16/BF16 performance
   - B200 has 2-3x higher memory bandwidth
   - Significantly larger memory capacity (192 GB)

4. **General Rules**:
   - Tensor core performance scales with matrix size (larger = better utilization)
   - FP16/BF16 should be 15-20x faster than FP32 on modern GPUs
   - Memory bandwidth tests help distinguish HBM vs GDDR variants

### Performance Factors

- **Tensor Core Utilization**: Achieves peak on matrix sizes divisible by 16 (FP16/BF16)
- **Memory Bound**: Small operations may be limited by bandwidth rather than compute
- **Thermal Throttling**: Performance may decrease under sustained load
- **PCIe vs SXM**: SXM variants typically have higher bandwidth and power

## Output Format

Results are saved in JSON format:

```json
{
  "gpu_info": {
    "name": "NVIDIA H100 PCIe",
    "memory_gb": 80.0,
    "cuda_capability": "9.0"
  },
  "results": [
    {
      "test_name": "GEMM",
      "operation": "MatMul (8192x8192) @ (8192x8192)",
      "dtype": "float16",
      "throughput_tflops": 756.32,
      "time_ms": 1.45,
      ...
    },
    {
      "test_name": "Memory",
      "operation": "Host-to-Device (PCIe) 1024 MB",
      "dtype": "float32",
      "memory_bandwidth_gb_s": 24.5,
      "time_ms": 41.8,
      ...
    }
  ]
}
```

### Example: Identifying PCIe Generation

If your benchmark shows:
- **H2D: 10-13 GB/s** → PCIe Gen3 x16
- **H2D: 22-27 GB/s** → PCIe Gen4 x16
- **H2D: 50-60 GB/s** → PCIe Gen5 x16
- **H2D: 40-80 GB/s** → SXM with NVLink-to-CPU (DGX/HGX system)

## Benchmark Details

### Matrix Multiplication (GEMM)
- Small: 4096x4096x4096 (256MB per matrix in FP32)
- Large: 8192x8192x8192 (2GB per matrix in FP32)
- Very Large: 16384x16384x16384 (8GB per matrix in FP32)

### Convolution
- ResNet-style 3x3 convolutions
- Various feature map sizes and channel counts
- Tests spatial processing capabilities

### Attention
- Small: BERT-like (512 seq length, 12 heads, 64 head dim)
- Large: GPT-like (2048 seq length, 32 heads, 128 head dim)
- Very Large: 4096 sequence length

### Memory Bandwidth
- Sequential copy operations
- Sizes: 512 MB, 1 GB, 2 GB
- Tests peak memory transfer rates

## Troubleshooting

### Recommended Scale Factors by GPU

**For 8-12GB GPUs (GTX 1080 Ti, RTX 2080 Ti):**
```bash
python gpu_benchmark.py --preset small
# or manually: --scale 0.25
```

**For 12-16GB GPUs (Quadro GP100, RTX 3060, RTX 3080 10GB):**
```bash
python gpu_benchmark.py --preset small
# or manually: --scale 0.35
```

**For 16-24GB GPUs (RTX 3080 Ti, RTX 3090, RTX 4080):**
```bash
python gpu_benchmark.py --preset small
# or manually: --scale 0.5
```

**For 24-40GB GPUs (RTX 4090, A10, A100 40GB):**
```bash
python gpu_benchmark.py --preset medium
# or manually: --scale 0.75
```

**For 40-48GB GPUs (L40S, A100 40GB):**
```bash
python gpu_benchmark.py --preset medium
# or for more stress: --preset large
```

**For 80GB GPUs (A100 80GB, H100):**
```bash
python gpu_benchmark.py --preset large
# or for maximum stress: --scale 2.0
```

**For 192GB GPUs (B200/B300):**
```bash
python gpu_benchmark.py --preset xlarge
# or for maximum stress: --scale 3.0
```

### Out of Memory Errors

**For Quadro GP100 (16GB) and similar GPUs:**

If you get OOM during "Large Attention" or other tests:
```bash
# Use the small preset (recommended)
python gpu_benchmark.py --preset small

# Or go even smaller if needed
python gpu_benchmark.py --scale 0.25

# Or skip attention tests entirely with custom sizes
python gpu_benchmark.py --matmul-size 2048 --batch-size 8 --seq-length 256
```

**General OOM Solutions:**
1. **Use presets**: `--preset small` for <16GB GPUs
2. **Reduce scale**: Use `--scale 0.25` or `--scale 0.35`
3. **Custom sizes**: `--matmul-size 2048 --batch-size 16`
4. **Clear cache**: The script automatically clears cache between tests, but you can restart if needed
5. **Close other applications**: Make sure no other programs are using GPU memory

### Low Performance

- Ensure GPU is not throttling (check temperatures)
- Close other GPU-intensive applications
- Check PCIe generation (Gen3 vs Gen4)
- Verify no power limits are set

### Multi-GPU Issues

**Warning: "using GPU X as device used by this process is currently unknown"**
- This warning is **harmless** and can be safely ignored
- PyTorch's NCCL backend shows this warning but the device mapping is correct
- The benchmark explicitly sets `torch.cuda.set_device(rank)` before initialization
- Everything will work properly despite the warning
- To suppress: Set environment variable `export NCCL_DEBUG=WARN` before running

**Error: "MASTER_ADDR expected, but not set"**
- This should be fixed in the latest version
- The script automatically sets MASTER_ADDR=localhost
- If you still see this, you can manually set:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  python gpu_benchmark.py --multi-gpu
  ```

**Error: "NCCL error" or "distributed initialization failed"**
- Check that all GPUs are visible: `nvidia-smi`
- Verify NCCL is installed: `python -c "import torch.distributed as dist; print(dist.is_nccl_available())"`
- Try reducing number of GPUs: `--num-gpus 2`
- Check for mixed GPU architectures (can cause issues)

**Multi-GPU test shows low bandwidth**
- Check GPU topology: `nvidia-smi topo -m`
- Look for NVLink connections (NV# instead of PHB)
- PCIe Gen3/Gen4 will show much lower bandwidth than NVLink
- Ensure GPUs are in same NUMA node for best performance

**Only seeing single-GPU results**
- Verify you used `--multi-gpu` flag
- Check that multiple GPUs were detected at startup
- Look for "Multi-GPU Mode: X GPUs" in output

### CUDA Errors

- Update PyTorch: `pip install --upgrade torch`
- Check CUDA version compatibility
- Verify GPU drivers are up to date

## License

MIT License - Feel free to modify and use for your benchmarking needs.

## Contributing

Suggestions for additional benchmarks or architecture-specific tests are welcome!
