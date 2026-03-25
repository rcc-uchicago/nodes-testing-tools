# Profiling PyTorch Training with NVIDIA Nsight Tools

This guide shows how to extract performance metrics (memory bandwidth, FLOPS, kernel execution time) from PyTorch training code using Nsight Systems and Nsight Compute.

## Device-to-Device Memory Copies in Training

### When D2D Copies Occur

Device-to-device (D2D) copies happen in several training scenarios:

1. **Model Parallelism**: Splitting model layers across multiple GPUs
2. **Pipeline Parallelism**: Passing activations between GPU stages
3. **Tensor Reshaping**: Sometimes triggers copies (non-contiguous → contiguous)
4. **Mixed Precision Training**: Type conversions (FP32 ↔ FP16)
5. **Gradient Accumulation**: Copying gradients between buffers
6. **Data Parallelism**: All-reduce operations (GPU-to-GPU during gradient sync)

### Common Training Patterns with D2D Copies

**Pattern 1: Explicit `.clone()` or `.copy_()`**
```python
# Explicit copy
tensor_copy = tensor.clone()  # D2D copy
tensor_dst.copy_(tensor_src)  # D2D copy
```

**Pattern 2: Non-contiguous to Contiguous**
```python
# Transpose creates a non-contiguous view
x = tensor.transpose(0, 1)
# Next operation may trigger implicit copy to make contiguous
y = model(x)  # May trigger D2D copy
```

**Pattern 3: Multi-GPU Model Parallelism**
```python
# Model split across GPUs
x = x.to('cuda:0')
x = layer1(x)  # On GPU 0
x = x.to('cuda:1')  # D2D copy: GPU 0 → GPU 1
x = layer2(x)  # On GPU 1
```

**Pattern 4: Gradient Accumulation**
```python
# Accumulating gradients from multiple batches
for micro_batch in batches:
    loss = forward(micro_batch)
    loss.backward()
    # Gradients accumulated (may involve D2D copies)
```

### How to Detect D2D Copies

**In Nsight Systems:**
- Look for "Memcpy DtoD" or "Device to Device" in memory operations
- Check "CUDA Memory Operation Statistics"
- Timeline shows transfers as separate events

**In Nsight Compute:**
```bash
# Profile memory operations
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
    --kernel-regex "memcpy" \\
    python train.py
```

**In PyTorch Profiler:**
```python
with torch.profiler.profile() as prof:
    # Training code
    pass

# Look for "aten::copy_" or "Memcpy DtoD" in output
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Overview: Nsight Systems vs Nsight Compute

### Nsight Systems (nsys)
- **Purpose**: System-wide profiling, timeline view
- **Use Case**: Understand overall application behavior, find bottlenecks, CPU-GPU interaction
- **Metrics**: Kernel execution time, memory transfers, CPU activity, GPU utilization
- **Level**: High-level overview
- **Speed**: Fast, low overhead (~5-10%)

### Nsight Compute (ncu)
- **Purpose**: Detailed kernel-level analysis
- **Use Case**: Deep dive into specific kernels, optimize individual operations
- **Metrics**: FLOPS, memory bandwidth, SM efficiency, warp occupancy, memory throughput
- **Level**: Kernel-level detail
- **Speed**: Slow, high overhead (10-100x slower)

**Recommended Workflow:**
1. Use **Nsight Systems** first to identify slow kernels
2. Use **Nsight Compute** to analyze specific slow kernels in detail

---

## Device-to-Device Copies in Training

Yes! Device-to-device copies happen in several scenarios:

### When D2D Copies Occur:

1. **Multi-GPU Training (Most Common)**
   - Gradient all-reduce operations (synchronizing gradients across GPUs)
   - Model parameter broadcasting
   - P2P transfers between GPUs during pipeline parallelism

2. **Single GPU Scenarios**
   - Tensor copies during gradient checkpointing
   - Moving tensors between different memory pools
   - `.clone()` operations on GPU tensors
   - Reshaping operations that can't be done in-place
   - Some autograd operations that create copies

3. **Mixed Precision Training**
   - Copying between FP16 and FP32 tensors
   - Master weight updates in AMP (Automatic Mixed Precision)

4. **Data Loading with Pinned Memory**
   - Async copies from staging buffers
   - Multi-stream data loading

### How to Measure D2D Bandwidth:

**In Nsight Systems:**
- Look for "Memcpy DtoD" or "cudaMemcpy" operations in the timeline
- Filter for CUDA memory operations
- Check the "CUDA Memory Operation Statistics" section

**In Nsight Compute:**
- Profile kernels with names containing "memcpy" or "copy"
- Check global memory load/store throughput

**In PyTorch Profiler:**
- Look for operations like "cudaMemcpyAsync", "clone", "copy_"

### PyTorch Operations That Trigger D2D Copies

```python
import torch

# 1. Explicit tensor cloning
a = torch.randn(1000, 1000, device='cuda')
b = a.clone()  # D2D copy

# 2. Explicit copy
a = torch.randn(1000, 1000, device='cuda')
b = torch.empty_like(a)
b.copy_(a)  # D2D copy

# 3. Type conversion requiring copy
a = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
b = a.half()  # D2D copy (FP32 to FP16)

# 4. Non-contiguous to contiguous
a = torch.randn(1000, 1000, device='cuda')
b = a.transpose(0, 1)  # View, no copy
c = b.contiguous()  # D2D copy if b is non-contiguous

# 5. Gradient checkpointing (implicit)
# torch.utils.checkpoint.checkpoint() creates D2D copies

# 6. Multi-GPU data parallel
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
# Automatically creates D2D copies during scatter/gather

# 7. Moving between GPUs
a = torch.randn(1000, 1000, device='cuda:0')
b = a.to('cuda:1')  # D2D copy between GPUs (P2P or via host)
```

### Multi-GPU Training Code with D2D Tracking

```python
import torch
import torch.cuda.nvtx as nvtx
from torch.nn.parallel import DistributedDataParallel as DDP

# Example: Gradient synchronization triggers D2D
model = DDP(model, device_ids=[rank])

for data, target in dataloader:
    nvtx.range_push("Forward")
    output = model(data)
    loss = criterion(output, target)
    nvtx.range_pop()
    
    nvtx.range_push("Backward")
    optimizer.zero_grad()
    loss.backward()  # Triggers all-reduce D2D copies here
    nvtx.range_pop()
    
    nvtx.range_push("Optimizer")
    optimizer.step()
    nvtx.range_pop()
```

When you profile this with Nsight Systems, you'll see D2D copies during the backward pass corresponding to gradient synchronization across GPUs.

## Part 1: Nsight Systems (nsys)

### Installation

Nsight Systems comes with CUDA Toolkit:
```bash
# Check if installed
which nsys

# If not installed, install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
```

### Basic Usage

#### 1. Profile PyTorch Training Script

```bash
# Basic profiling
nsys profile -o pytorch_profile python train.py

# With more options
nsys profile \
    -o pytorch_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    python train.py

# Profile for specific duration (e.g., 30 seconds)
nsys profile \
    -o pytorch_profile \
    --duration=30 \
    --trace=cuda,nvtx \
    python train.py
```

#### 2. Add NVTX Markers to Your PyTorch Code

Adding markers helps identify specific sections:

```python
import torch
import torch.cuda.nvtx as nvtx

# Training loop with markers
for epoch in range(num_epochs):
    nvtx.range_push(f"Epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        nvtx.range_push(f"Batch {batch_idx}")
        
        # Forward pass
        nvtx.range_push("Forward")
        output = model(data)
        loss = criterion(output, target)
        nvtx.range_pop()  # Forward
        
        # Backward pass
        nvtx.range_push("Backward")
        optimizer.zero_grad()
        loss.backward()
        nvtx.range_pop()  # Backward
        
        # Optimizer step
        nvtx.range_push("Optimizer")
        optimizer.step()
        nvtx.range_pop()  # Optimizer
        
        nvtx.range_pop()  # Batch
    
    nvtx.range_pop()  # Epoch
```

#### 3. Analyze the Profile

```bash
# View in GUI (recommended)
nsys-ui pytorch_profile.nsys-rep

# Or export to text for analysis
nsys stats pytorch_profile.nsys-rep

# Export kernel statistics to CSV
nsys stats --report cuda_gpu_kern_sum pytorch_profile.nsys-rep -o . --format csv

# Export memory operations
nsys stats --report cuda_gpu_mem_time_sum pytorch_profile.nsys-rep -o . --format csv
```

#### 4. Key Metrics from Nsight Systems

The GUI and stats output provide:

**From CUDA GPU Kernel Summary:**
- Kernel name (e.g., `volta_sgemm_128x128_nn`, `cudnn_convolution`)
- Total execution time
- Number of calls
- Average/Min/Max duration
- Grid/block dimensions

**From Memory Operations:**
- HtoD (Host-to-Device) transfers: time and size
- DtoH (Device-to-Host) transfers: time and size
- DtoD (Device-to-Device) copies: time and size **← These are GPU memory copies**
- Memory bandwidth = Size / Time

**Identifying D2D Copies:**
In the Nsight Systems GUI or stats output, look for:
- Operation name: "cudaMemcpy" with "Device to Device"
- Operation name: "cudaMemcpyAsync DtoD"
- CUDA kernel names containing "memcpy"

**Example output:**
```
CUDA Kernel Statistics:

Name                                  | Total Time (ns) | Calls | Avg (ns)  | Grid      | Block
--------------------------------------------------------------------------------------------------
volta_fp16_s884gemm_fp16_128x128_ldg  | 1,234,567,890  | 1000  | 1,234,567 | 256,1,1   | 128,1,1
cudnn::winograd::generateWinogradTiles| 567,890,123    | 500   | 1,135,780 | 128,4,1   | 32,8,1
void at::native::elementwise_kernel   | 123,456,789    | 2000  | 61,728    | 1024,1,1  | 256,1,1

Memory Operations:

Type  | Size (MB) | Time (ms) | Bandwidth (GB/s) | Count
-----------------------------------------------------------
HtoD  | 512.00    | 20.5      | 24.98            | 100
DtoH  | 256.00    | 10.2      | 25.10            | 50
DtoD  | 1024.00   | 0.8       | 1280.00          | 200   ← GPU-to-GPU or intra-GPU copies
```

**What D2D Bandwidth Tells You:**
- **Multi-GPU**: D2D bandwidth reveals NVLink vs PCIe performance
- **Single GPU**: D2D bandwidth shows HBM/GDDR memory speed
- **High D2D time**: May indicate unnecessary tensor copies in your code

---

## Part 2: Nsight Compute (ncu)

### Installation

Comes with CUDA Toolkit 10.0+:
```bash
# Check if installed
which ncu

# Or use older name
which nv-nsight-cu-cli
```

### Basic Usage

#### 1. Profile Specific Kernels

**WARNING**: Nsight Compute is VERY slow. Don't profile entire training runs!

```bash
# Profile all kernels (SLOW!)
ncu -o pytorch_kernel_profile python train.py

# Profile only the first 10 kernel launches
ncu --launch-count 10 -o profile python train.py

# Profile specific kernel by name
ncu --kernel-name "volta_sgemm" -o profile python train.py

# Profile with all metrics (VERY SLOW!)
ncu --set full -o profile python train.py

# Profile with specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
-o profile python train.py
```

#### 2. Key Metrics to Extract

**Memory Bandwidth:**
```bash
# Get memory bandwidth metrics
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
duration \
--kernel-name "your_kernel_name" \
python train.py
```

**FLOPS (Floating Point Operations):**
```bash
# Get FLOP metrics
ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
duration \
--kernel-name "your_kernel_name" \
python train.py
```

**Compute TFLOPS from metrics:**
```
FP32 FLOPS = (FADD + FMUL + 2*FFMA)
FP16 FLOPS = (HADD + HMUL + 2*HFMA)

TFLOPS = Total_FLOPS / (Duration_in_seconds * 1e12)
```

#### 3. Interactive Analysis

```bash
# Launch GUI
ncu-ui pytorch_kernel_profile.ncu-rep

# Export to CSV
ncu --csv -i pytorch_kernel_profile.ncu-rep

# Print summary
ncu --print-summary per-kernel -i pytorch_kernel_profile.ncu-rep
```

#### 4. Example: Analyzing a MatMul Kernel

```bash
# Profile matmul kernels with key metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
duration \
--kernel-name "gemm" \
-o matmul_profile \
python -c "
import torch
a = torch.randn(8192, 8192, device='cuda')
b = torch.randn(8192, 8192, device='cuda')
torch.cuda.synchronize()
c = torch.matmul(a, b)
torch.cuda.synchronize()
"

# View results
ncu --print-summary per-kernel -i matmul_profile.ncu-rep
```

Expected output:
```
Kernel: volta_sgemm_128x128_nn
  Duration: 1.234 ms
  SM Throughput: 87.5%
  DRAM Throughput: 12.3%
  FFMA Operations: 549,755,813,888
  
  Calculated FLOPS:
  Total FP Operations = 2 * FFMA = 1,099,511,627,776
  TFLOPS = 1,099,511,627,776 / (0.001234 * 1e12) = 891.2 TFLOPS
```

---

## Part 3: PyTorch Built-in Profiler

PyTorch has a built-in profiler that's easier to use but less detailed:

### Basic PyTorch Profiler

```python
import torch
import torch.profiler

# Training code
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,  # Estimate FLOPS
) as prof:
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # Profile only first 10 batches
            break
        
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export to Chrome trace (view in chrome://tracing)
prof.export_chrome_trace("pytorch_trace.json")

# Export to TensorBoard
# prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
```

### PyTorch Profiler with FLOPS

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_flops=True,
) as prof:
    # Your training code
    output = model(input)
    loss.backward()

# Get FLOP counts
for evt in prof.key_averages():
    if evt.flops:
        print(f"{evt.key}: {evt.flops/1e12:.2f} TFLOPS, Time: {evt.cuda_time_total/1e6:.2f} ms")
```

---

---

## Part 4: Device-to-Device (D2D) Memory Copies

### When Do D2D Copies Happen in Training?

Device-to-device copies within GPU memory happen in several scenarios:

#### Common Cases:

1. **Tensor Reshaping/Views (Usually Zero-Copy)**
   - Most reshaping operations are just metadata changes (no copy)
   - Example: `tensor.view()`, `tensor.transpose()` - zero-copy
   
2. **Non-Contiguous to Contiguous Conversion**
   ```python
   # This may trigger a D2D copy
   x = tensor.transpose(0, 1)  # Creates non-contiguous view
   y = x.contiguous()          # Copies data to make contiguous
   ```

3. **Explicit Copies**
   ```python
   # Explicit device-to-device copy
   x = torch.randn(1000, 1000, device='cuda:0')
   y = x.clone()  # D2D copy on same device
   ```

4. **Multi-GPU Training (Most Common)**
   ```python
   # Copy tensors between GPUs
   x = torch.randn(1000, 1000, device='cuda:0')
   y = x.to('cuda:1')  # GPU-to-GPU copy via PCIe/NVLink
   ```

### How to Detect D2D Copies in Nsight Systems

```bash
# Profile with detailed memory tracking
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    -o profile \
    python train.py

# Export memory operations
nsys stats --report cuda_gpu_mem_time_sum profile.nsys-rep --format csv
```

Look for **"Memcpy DtoD"** in the output.

### Measuring D2D Bandwidth

**In Nsight Systems CSV output:**
```csv
Operation,Count,Total Size (MB),Total Time (ns)
[CUDA memcpy DtoD],500,25600.0,128000000
```

Calculate: `Bandwidth = 25600 MB / 0.128 s = 195.3 GB/s`

**In Code:**
```python
import torch
import time

size_mb = 1024
size_elements = (size_mb * 1024 * 1024) // 4

src = torch.randn(size_elements, dtype=torch.float32, device='cuda')
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    dst = src.clone()
torch.cuda.synchronize()
end = time.perf_counter()

time_ms = (end - start) * 1000 / 100
bandwidth_gbs = (size_mb / 1024) / (time_ms / 1000)

print(f"D2D Copy: {bandwidth_gbs:.2f} GB/s")
```

### Multi-GPU D2D Bandwidth

```python
# GPU-to-GPU copy
src = torch.randn(size_elements, device='cuda:0')
dst = src.to('cuda:1')  # Triggers P2P or via CPU
```

**Expected Results:**
- **PCIe Gen4**: 20-25 GB/s (one direction)
- **NVLink 3.0 (A100)**: 400-500 GB/s
- **NVLink 4.0 (H100)**: 600-700 GB/s

### D2D Bandwidth by GPU (Single Device)

| GPU | Memory Type | D2D Bandwidth |
|-----|-------------|---------------|
| Quadro GP100 | HBM2 | 600-700 GB/s |
| RTX 3090 | GDDR6X | 900-950 GB/s |
| A100 80GB | HBM2e | 1,800-1,935 GB/s |
| H100 | HBM3 | 2,800-3,200 GB/s |
| L40S | GDDR6 | 750-850 GB/s |

**Note:** D2D bandwidth should match the GPU's memory bandwidth.

### When D2D Copies Matter

D2D copies become significant when:
- Frequent `.contiguous()` calls
- Excessive `.clone()` operations
- Multi-GPU communication dominates
- Model/pipeline parallelism

**In most training**: D2D copies are <5% of time. Compute (GEMM/Conv) and data loading (H2D) dominate.

---

## Part 5: Practical Example - Profile User's Training Code

### Step 1: Quick Overview with Nsight Systems

```bash
# Add NVTX markers to user's code (if you have access)
# Then profile for 30 seconds
nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --duration=30 \
    --output=user_training_profile \
    python user_train.py

# Analyze in GUI
nsys-ui user_training_profile.nsys-rep

# Export kernel statistics
nsys stats --report cuda_gpu_kern_sum user_training_profile.nsys-rep \
    --format csv -o user_kernels.csv
```

### Step 2: Identify Slow Kernels

From the CSV or GUI, find:
1. Kernels with highest total time
2. Kernels called frequently
3. Memory transfers taking significant time

### Step 3: Deep Dive with Nsight Compute

```bash
# Profile the top 3 slowest kernels
ncu --kernel-name "slow_kernel_name_1|slow_kernel_name_2|slow_kernel_name_3" \
    --set full \
    --launch-count 5 \
    --output=detailed_kernel_profile \
    python user_train.py

# Analyze in GUI
ncu-ui detailed_kernel_profile.ncu-rep
```

### Step 4: Extract Metrics

In Nsight Compute GUI, look for:

**Memory Section:**
- Memory Throughput (% of peak)
- Actual bandwidth in GB/s
- L1/L2 cache hit rates

**Compute Section:**
- SM Throughput (% of peak)
- Achieved FLOPS
- Warp execution efficiency

**Example workflow:**
1. Open `.ncu-rep` file in ncu-ui
2. Select kernel from list
3. Go to "Details" page
4. Check "Memory Workload Analysis" for bandwidth
5. Check "Compute Workload Analysis" for FLOPS
6. Check "SOL (Speed of Light)" for bottleneck identification

### Step 5: Measure Device-to-Device (D2D) Bandwidth

**In Nsight Systems:**

D2D copies appear as "Memcpy DtoD" operations in the timeline.

```bash
# Profile with memory operations
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    python model_parallel_train.py

# Export memory statistics
nsys stats --report cuda_gpu_mem_time_sum profile.nsys-rep --format csv
```

In the GUI or CSV output, look for:
- Operation Type: "Device to Device" or "DtoD"
- Size (bytes/MB)
- Duration
- Bandwidth = Size / Duration

**D2D Bandwidth Calculation:**
```
Bandwidth (GB/s) = Size (GB) / Duration (seconds)
```

**Expected D2D Bandwidth:**
- **Same GPU (on-device copy)**: ~HBM bandwidth (1,500-3,000 GB/s)
- **PCIe Gen4 between GPUs**: 20-25 GB/s
- **NVLink 3.0 (A100)**: 300-600 GB/s per direction
- **NVLink 4.0 (H100)**: 450-900 GB/s per direction
- **NVLink 5.0 (B200)**: 900-1,800 GB/s per direction (estimated)

**In PyTorch Profiler:**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Model parallelism code with .to() calls
    x = x.to('cuda:1')  # D2D copy
    
# Look for copy operations
for evt in prof.key_averages():
    if 'copy' in evt.key.lower() or 'memcpy' in evt.key.lower():
        print(f"{evt.key}: {evt.cuda_time_total/1000:.2f} ms")
```

**Common D2D Copy Triggers in PyTorch:**
- `tensor.to(device)` - Explicit device transfer
- `tensor.clone()` - Creates a copy
- `tensor.contiguous()` - May copy if tensor is non-contiguous
- `.reshape()` vs `.view()` - reshape may copy, view won't
- Mixed precision casting - FP32 ↔ FP16 conversions

**Nsight Systems Example Output:**
```
Memory Operation Statistics:

Type            | Count | Total Size (MB) | Total Time (ms) | Bandwidth (GB/s)
-----------------------------------------------------------------------------
Host to Device  | 100   | 1024.00        | 41.2           | 24.9
Device to Host  | 50    | 512.00         | 20.6           | 24.9
Device to Device| 200   | 2048.00        | 4.1            | 500.0  ← D2D copies
```

The high D2D bandwidth (500 GB/s) indicates NVLink connection between GPUs.

---

---

## Part 6: Quick Reference Commands

### Nsight Systems Cheat Sheet

```bash
# Basic profile
nsys profile -o output python train.py

# With NVTX and memory tracking
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true -o output python train.py

# Export kernel stats to CSV
nsys stats --report cuda_gpu_kern_sum output.nsys-rep --format csv

# View in GUI
nsys-ui output.nsys-rep
```

### Nsight Compute Cheat Sheet

```bash
# Profile first 10 kernels
ncu --launch-count 10 -o output python train.py

# Profile specific kernel
ncu --kernel-name "kernel_name" -o output python train.py

# Get memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed python train.py

# Get FLOPS metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python train.py

# Full metrics (slow!)
ncu --set full -o output python train.py

# View in GUI
ncu-ui output.ncu-rep
```

### PyTorch Profiler Cheat Sheet

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    # Training code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

---

## Part 7: Device-to-Device (D2D) Copies in Training

### When D2D Copies Occur

Device-to-device copies between GPUs are common in distributed training:

1. **Model Parallel Training**
   - Layer outputs copied from one GPU to another
   - Example: GPT-3 style pipeline parallelism

2. **Pipeline Parallel Training**
   - Activations passed between pipeline stages
   - Microbatch transfers between GPUs

3. **Tensor Parallel Training**
   - Split/gather operations for tensor model parallelism
   - Example: Megatron-LM splits attention heads across GPUs

4. **Mixed Strategies**
   - Combinations of data, pipeline, and tensor parallelism
   - ZeRO optimizer stages with parameter/gradient gathering

### Detecting D2D Copies with Nsight Systems

```bash
# Profile with memory trace
nsys profile --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    -o profile \
    python train.py
```

In the Nsight Systems GUI:
1. Look for "CUDA Memory Operations" timeline
2. Filter for "Device-to-Device" or "Peer-to-Peer" transfers
3. These appear as green bars in the timeline

**Common D2D patterns:**
- `cudaMemcpyPeerAsync` - Direct P2P copy between GPUs
- Repeated small copies during forward/backward pass
- Large copies during weight updates or gradient synchronization

### Example: Finding D2D Copies in PyTorch Code

```python
import torch
import torch.cuda.nvtx as nvtx

# Model parallel example
class ModelParallelNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on GPU 0
        self.layer1 = nn.Linear(1024, 1024).cuda(0)
        self.layer2 = nn.Linear(1024, 1024).cuda(0)
        
        # Second half on GPU 1
        self.layer3 = nn.Linear(1024, 1024).cuda(1)
        self.layer4 = nn.Linear(1024, 10).cuda(1)
    
    def forward(self, x):
        # Compute on GPU 0
        x = x.cuda(0)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        # D2D COPY HAPPENS HERE
        nvtx.range_push("D2D Copy GPU0->GPU1")
        x = x.cuda(1)  # This triggers a D2D copy
        nvtx.range_pop()
        
        # Compute on GPU 1
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
```

Profile this:
```bash
nsys profile --trace=cuda,nvtx -o model_parallel_profile python train.py
```

In the timeline, you'll see:
- Compute kernels on GPU 0
- **D2D copy from GPU 0 to GPU 1** (marked by NVTX)
- Compute kernels on GPU 1

### Measuring D2D Bandwidth with Nsight Systems

From the exported memory statistics:

```bash
# Export memory operations
nsys stats --report cuda_gpu_mem_time_sum profile.nsys-rep --format csv

# Look for rows with "DtoD" or "Peer" in operation column
# Calculate: Bandwidth (GB/s) = Size (MB) / 1024 / Time (s)
```

Example output:
```
Operation           | Size (MB) | Time (ms) | Bandwidth (GB/s)
-----------------------------------------------------------------
DtoD (GPU 0->1)     | 256.0     | 0.8       | 320.0
DtoD (GPU 1->0)     | 256.0     | 0.8       | 320.0
```

### Measuring D2D Bandwidth with Nsight Compute

Nsight Compute doesn't directly measure memory copies, but you can:

1. **Use CUDA profiler APIs in code:**

```python
import torch

# Measure D2D copy bandwidth manually
def measure_d2d_bandwidth(size_mb=100, iterations=100):
    size_elements = (size_mb * 1024 * 1024) // 4
    
    src = torch.randn(size_elements, device='cuda:0')
    dst = torch.empty(size_elements, device='cuda:1')
    
    # Warmup
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    # Measure
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        dst.copy_(src)
    end.record()
    
    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end) / iterations
    bandwidth_gbs = (size_mb / 1024) / (time_ms / 1000)
    
    print(f"D2D Bandwidth: {bandwidth_gbs:.2f} GB/s")
    return bandwidth_gbs

# Run it
measure_d2d_bandwidth(100)
```

2. **Look for memory copy operations in training:**

```bash
# Profile and look for copy operations
nsys profile --trace=cuda,nvtx python measure_d2d.py
nsys stats --report cuda_gpu_mem_time_sum --format csv
```

### Expected D2D Bandwidth

D2D copies use the same interconnect as P2P operations:

**NVLink 3.0 (A100):**
- Unidirectional: 300-400 GB/s
- Bidirectional: 500-600 GB/s

**NVLink 4.0 (H100):**
- Unidirectional: 400-500 GB/s  
- Bidirectional: 700-800 GB/s

**PCIe Gen4 x16:**
- Unidirectional: 12-15 GB/s
- Bidirectional: 20-25 GB/s

**PCIe Gen5 x16:**
- Unidirectional: 25-30 GB/s
- Bidirectional: 40-50 GB/s

### Common D2D Bottlenecks

1. **Too Many Small Copies**
   - Symptom: Many small D2D transfers in timeline
   - Solution: Batch copies together, use pipelining

2. **Sequential Copies**
   - Symptom: Copies happen one after another
   - Solution: Use async copies with streams

3. **PCIe Bottleneck**
   - Symptom: D2D bandwidth < 30 GB/s
   - Solution: Use NVLink-connected systems, or reduce cross-GPU communication

4. **Unbalanced Pipeline**
   - Symptom: One GPU idle while waiting for D2D
   - Solution: Overlap computation with communication

### Optimizing D2D Transfers

```python
# BAD: Synchronous copies
x = x.cuda(1)  # Blocks until copy complete

# BETTER: Async with streams
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    x = x.cuda(1, non_blocking=True)
# Overlap with other work

# BEST: Batch multiple small copies
# Instead of copying layer-by-layer, copy entire activations at once
```

---

## Part 8: Comparing Benchmark with Real Training

## Part 8: Comparing Benchmark with Real Training

### Mapping Benchmark Tests to Real Kernels

| Benchmark Test | Real PyTorch Kernels | How to Find |
|----------------|---------------------|-------------|
| GEMM (MatMul) | `volta_sgemm`, `turing_sgemm`, `ampere_sgemm` | Look for `gemm` or `matmul` in kernel names |
| Conv2D | `cudnn::convolution`, `implicit_gemm_conv` | Look for `conv` in kernel names |
| Attention | `fmha_*`, custom kernels | Look for `attention` or multiple matmuls + softmax |
| Memory Bandwidth | HtoD, DtoH transfers | Check "CUDA Memory Operation Statistics" |

### Extract Same Metrics

**From Nsight Systems:**
- Kernel execution time → Compare with benchmark time_ms
- Memory transfer bandwidth → Compare with H2D/D2H bandwidth

**From Nsight Compute:**
- Achieved FLOPS → Compare with benchmark throughput_tflops
- Memory bandwidth → Compare with benchmark memory_bandwidth_gb_s
- SM occupancy → Indicates efficiency

---

## Troubleshooting

### Nsight Systems Issues

**Problem**: No CUDA traces showing
```bash
# Make sure CUDA tracing is enabled
nsys profile --trace=cuda python train.py
```

**Problem**: File too large
```bash
# Limit profiling duration
nsys profile --duration=10 python train.py
```

### Nsight Compute Issues

**Problem**: Takes forever to run
```bash
# Profile only first few kernels
ncu --launch-count 5 python train.py

# Or specific kernels
ncu --kernel-name "kernel_name" python train.py
```

**Problem**: Permission denied
```bash
# Run with sudo if needed (not recommended)
# Or adjust permissions: sudo chmod 666 /dev/nvidia*
```

### PyTorch Profiler Issues

**Problem**: Out of memory
```python
# Profile fewer batches
if batch_idx >= 5:
    break
```

---

## Summary

1. **For overall performance**: Use **Nsight Systems** (`nsys`)
   - Shows timeline, kernel execution, memory transfers
   - Low overhead, fast

2. **For kernel details**: Use **Nsight Compute** (`ncu`)
   - Shows FLOPS, bandwidth, efficiency
   - High overhead, slow

3. **For quick analysis**: Use **PyTorch Profiler**
   - Built-in, easy to use
   - Good for identifying hot spots

4. **Workflow**:
   - Start with PyTorch profiler or Nsight Systems
   - Identify slow kernels
   - Deep dive with Nsight Compute on specific kernels
   - Compare metrics with our benchmark to identify architecture

The key advantage of profiling real training code is that you see actual performance in production workloads, not just synthetic benchmarks!
