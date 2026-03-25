# FP32 → INT8 Weight Quantization: PyTorch vs CUDA C vs ONNX Runtime

A side-by-side comparison of FP32-to-INT8 weight quantization and matrix multiplication across **PyTorch**, **raw CUDA C**, and **ONNX Runtime**, with GPU profiling to show exactly which GPU cores (CUDA Cores vs Tensor Cores) each operation uses.

**Tested on:** NVIDIA RTX A3000 Laptop GPU (Ampere, sm_86, 6GB VRAM, 32 SMs)

---

## Files

| File | Language | What It Does |
|------|----------|-------------|
| `fp32_to_int8_pytorch.py` | Python/PyTorch | FP32 linear layer → INT8 quantization (manual + torchao) → error analysis → LLM-scale benchmarks |
| `fp32_to_int8_profiled.py` | Python/PyTorch | GPU kernel profiling with `torch.profiler` — identifies which GPU cores each operation uses |
| `fp32_to_int8_cuda.cu` | CUDA C | Same quantization + matmul in raw CUDA — cuBLAS SGEMM (FP32) + cuBLASLt IGEMM (INT8) + custom kernels |
| `fp32_to_int8_onnx.py` | Python/ONNX | ONNX Runtime FP32/INT8 (CPU + GPU) vs PyTorch — export, quantize, benchmark, error comparison |
| `Makefile` | Make | Build system for the CUDA binary |

---

## Quick Start

### 1. PyTorch Version (quantization + benchmarks)

```bash
# Requires: PyTorch with CUDA, torchao
pip install torch torchao

python fp32_to_int8_pytorch.py
```

**What it does:**
- Step 1: FP32 linear layer forward pass (`y = xW^T + b`)
- Step 2: Symmetric INT8 quantization (`scale = max(|tensor|) / 127`)
- Step 3: INT8 matmul with INT32 accumulation + dequantize
- Step 4: Error analysis (FP32 vs INT8 output)
- Step 5: Weight reconstruction quality check
- Step 6: torchao official INT8 quantization (weight-only + dynamic activation)
- Step 7: Timing benchmarks at LLM-scale (4K×4K, 4K×11K, 8K×8K) with `torch.compile`

### 2. PyTorch Profiling (which GPU cores?)

```bash
python fp32_to_int8_profiled.py
```

**What it does:**
- Profiles 6 sessions: FP32 and INT8 at batch=1 (GEMV) and batch=32 (GEMM)
- Tests TF32 ON vs OFF to show CUDA Core vs Tensor Core routing
- Prints actual CUDA kernel names with core-type classification
- Exports Chrome trace `.json` files (open in `chrome://tracing`)
- Runs clean timing benchmarks (no profiler overhead)

### 3. ONNX Runtime Version (cross-platform inference)

```bash
# Requires: onnx, onnxruntime-gpu
pip install onnx onnxruntime-gpu

python fp32_to_int8_onnx.py
```

**What it does:**
- Step 1-2: Create FP32 model, export to ONNX format
- Step 3: Run FP32 inference via ONNX Runtime (CPU + GPU)
- Step 4: Dynamic INT8 quantization (`quantize_dynamic`)
- Step 5: Error analysis — PyTorch vs ONNX (FP32 & INT8)
- Step 6-7: LLM-scale timing benchmark across all runtimes with model size comparison

### 4. CUDA C Version (raw GPU programming)

```bash
# Install CUDA toolkit (one-time setup)
conda create -n cuda_build -c nvidia/label/cuda-12.8.0 cuda-toolkit

# Build
make

# Run
make run

# Clean
make clean
```

**What it does:**
- Same math as the PyTorch version but in raw CUDA C
- FP32 matmul via cuBLAS `cublasSgemm` (with TF32 ON/OFF control)
- INT8 matmul via cuBLASLt `cublasLtMatmul` — true `INT8 × INT8 → INT32` on Tensor Cores
- Custom CUDA kernels for: quantization, dequantization, bias-add, abs-max reduction
- Per-kernel timing with CUDA Events
- LLM-scale benchmarks matching the PyTorch configs

---

## Which GPU Core Is Each Operation Using?

```
┌─────────────────────────┬──────────────────────────┬────────────────┐
│ Operation               │ Kernel Dispatched        │ GPU Core Type  │
├─────────────────────────┼──────────────────────────┼────────────────┤
│ FP32 GEMV (batch=1)     │ internal::gemvx          │ CUDA Cores     │
│ FP32 GEMM TF32=OFF      │ ampere_sgemm_64x32       │ CUDA Cores     │
│ FP32 GEMM TF32=ON       │ cutlass_80_tensorop_s1688│ Tensor Cores   │
│ INT8 IGEMM (CUDA C)     │ imma / s8_tensorop       │ Tensor Cores   │
│ INT8 torchao (PyTorch)   │ dequant → gemvx/sgemm    │ CUDA Cores *   │
│ Quantize FP32→INT8       │ custom elementwise       │ CUDA Cores     │
│ Dequantize INT32→FP32    │ custom elementwise       │ CUDA Cores     │
│ ONNX FP32 GPU            │ cuBLAS sgemm (via ORT)   │ CUDA/Tensor ** │
│ ONNX FP32 CPU            │ MLAS sgemm               │ CPU (AVX)      │
│ ONNX INT8 CPU            │ MLAS int8 gemm           │ CPU (VNNI)     │
└─────────────────────────┴──────────────────────────┴────────────────┘

*  torchao weight-only quantization dequantizes INT8→FP32 FIRST, then does
   FP32 matmul. It does NOT do native INT8 matmul on Tensor Cores.
** ONNX Runtime GPU uses cuBLAS under the hood — same CUDA/Tensor Core
   routing as PyTorch (TF32 default on Ampere for GEMM).
```

### CUDA Cores vs Tensor Cores — What Are They?

```
  SM (Streaming Multiprocessor) — 32 SMs on RTX A3000
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │   CUDA Cores (128 per SM = 4096 total)             │
  │   ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  ...          │
  │   │FP││FP││FP││FP││FP││FP││FP││FP│               │
  │   │32││32││32││32││32││32││32││32│               │
  │   └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘               │
  │   General-purpose: 1 FMA per core per clock        │
  │   Used for: SGEMM, GEMV, elementwise, reductions   │
  │                                                    │
  │   Tensor Cores (4 per SM = 128 total)              │
  │   ┌──────────┐ ┌──────────┐                        │
  │   │ 4×4 MMA  │ │ 4×4 MMA  │  ...                  │
  │   │ FP16/BF16│ │ INT8/INT4│                        │
  │   │ TF32     │ │ IMMA     │                        │
  │   └──────────┘ └──────────┘                        │
  │   Specialized: 4×4 matrix multiply per clock       │
  │   Used for: GEMM (batch≥2), NOT GEMV (batch=1)     │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

**Key insight:** Tensor Cores only activate for matrix-matrix multiply (GEMM, batch≥2). At batch=1, cuBLAS dispatches GEMV (matrix-vector), which always runs on CUDA Cores. This matters for LLM inference where single-token generation is batch=1.

---

## Results

All benchmarks on **NVIDIA RTX A3000 Laptop GPU** (sm_86, 6GB VRAM, 32 SMs).

### Accuracy (Small 2×4 Demo)

Both PyTorch and CUDA C produce the same quantization:

| Metric | PyTorch | CUDA C |
|--------|---------|--------|
| Mean absolute error | ~0.007 | 0.0075 |
| Mean relative error | ~1.3% | 1.27% |
| Compression ratio | 4× | 4× |

### PyTorch Profiling Results — Kernel Identification

**batch=1 (GEMV):**

| Session | Top Kernel | Core Type | GPU Time % |
|---------|-----------|-----------|------------|
| FP32 TF32-OFF | `internal::gemvx::kernel` | **CUDA Cores** | 95.2% |
| FP32 TF32-ON | `internal::gemvx::kernel` | **CUDA Cores** | 95.2% |
| INT8 torchao | `unrolled_elementwise` (dequant) + `gemvx` | **CUDA Cores** | 41% compute, 56% dequant |

> TF32 ON/OFF makes **zero difference** at batch=1 because GEMV has no Tensor Core path.

**batch=32 (GEMM):**

| Session | Top Kernel | Core Type | GPU Time % |
|---------|-----------|-----------|------------|
| FP32 TF32-OFF | `ampere_sgemm_64x32_sliced1x4_tn` | **CUDA Cores** | 95.1% |
| FP32 TF32-ON | `cutlass_80_tensorop_s1688gemm_128x64` | **Tensor Cores** | 95.2% |
| INT8 torchao | `cutlass_80_tensorop_s1688gemm_64x64` + dequant | **Tensor Cores** | 42% compute, 55% dequant |

> At batch=32, TF32=ON routes through Tensor Cores (`cutlass_tensorop`), giving 1.17× speedup over CUDA Cores.

### Benchmark Results — PyTorch (torch.profiler OFF)

| Config | FP32 TF32-OFF | FP32 TF32-ON | INT8 torchao | Best |
|--------|--------------|-------------|-------------|------|
| GEMV 4K×4K (b=1) | 0.283 ms | 0.283 ms | 0.666 ms | FP32 |
| GEMV 4K×11K (b=1) | 0.730 ms | 0.728 ms | 1.721 ms | FP32 |
| GEMV 8K×8K (b=1) | 1.074 ms | 1.075 ms | 2.528 ms | FP32 |
| GEMM 4K×11K (b=32) | 0.888 ms | 0.756 ms | 1.750 ms | TF32 |
| GEMM 8K×8K (b=32) | 1.210 ms | 1.114 ms | 2.561 ms | TF32 |

> **torchao INT8 weight-only is SLOWER than FP32** because it dequantizes weights to FP32 at runtime. The dequant overhead (55% of GPU time) outweighs the memory savings.

### Benchmark Results — CUDA C (cuBLASLt IGEMM)

| Config | FP32 TF32-OFF | FP32 TF32-ON | INT8 IGEMM | Best |
|--------|--------------|-------------|-----------|------|
| GEMV 4K×4K (b=1) | 0.269 ms | 0.269 ms | **0.084 ms** | **INT8 (3.2×)** |
| GEMV 4K×11K (b=1) | 0.714 ms | 0.714 ms | **0.204 ms** | **INT8 (3.5×)** |
| GEMV 8K×8K (b=1) | 1.061 ms | 1.060 ms | **0.311 ms** | **INT8 (3.4×)** |
| GEMM 4K×11K (b=32) | 0.841 ms | 0.740 ms | **0.194 ms** | **INT8 (4.3×)** |
| GEMM 8K×8K (b=32) | 1.181 ms | 1.088 ms | **0.281 ms** | **INT8 (4.2×)** |

> **CUDA C INT8 IGEMM is 3-4× faster than FP32** across ALL sizes. True INT8×INT8→INT32 matmul, no dequant overhead.

---

### Benchmark Results — ONNX Runtime

| Config | PT FP32 GPU | ORT FP32 GPU | ORT FP32 CPU | ORT INT8 CPU | PT INT8 torchao |
|--------|------------|-------------|-------------|-------------|----------------|
| Small (4K×4K, b=1) | 0.280 ms | 0.285 ms | 1.496 ms | **0.107 ms** | 0.667 ms |
| Medium (4K×11K, b=1) | 0.725 ms | 0.729 ms | 4.520 ms | 0.783 ms | 1.711 ms |
| Large (8K×8K, b=1) | 1.074 ms | 1.079 ms | 7.307 ms | 1.560 ms | 2.525 ms |
| Batched (32×4K×11K) | 0.752 ms | 0.774 ms | 4.780 ms | 1.114 ms | 1.753 ms |

> **ONNX INT8 CPU beats GPU for small matrices** (0.107 ms vs 0.280 ms at 4K×4K) because MLAS has native INT8 GEMM (AVX-512/VNNI), there's no GPU launch overhead, and the 16 MB INT8 weight fits in L3 cache.

### ONNX Model Size (4× Compression)

| Config | FP32 | INT8 | Compression |
|--------|------|------|-------------|
| Small (4K×4K) | 65,552 KB | 16,401 KB | 4.00× |
| Medium (4K×11K) | 176,171 KB | 44,076 KB | 4.00× |
| Large (8K×8K) | 262,176 KB | 65,569 KB | 4.00× |

### ONNX Quantization Error (vs PyTorch FP32 GPU)

| Config | ONNX FP32 | ONNX INT8 | PT INT8 torchao |
|--------|-----------|-----------|-----------------|
| Small (4K×4K) | 0.000018 | 0.749 | 0.432 |
| Medium (4K×11K) | 0.000018 | 0.732 | 0.440 |
| Large (8K×8K) | 0.000029 | 1.106 | 0.635 |

> ONNX INT8 has higher error than torchao because it quantizes **both weights AND activations** (dynamic quantization), while torchao only quantizes weights and keeps activations in FP32.

---

## Full Cross-Runtime Comparison — Head-to-Head

### Master Inference Time Table (ms)

| Config | CUDA C FP32 | CUDA C TF32 | CUDA C INT8 | PT FP32 GPU | PT INT8 torchao | ORT FP32 GPU | ORT INT8 CPU |
|--------|------------|------------|------------|------------|----------------|-------------|-------------|
| GEMV 4K×4K (b=1) | 0.270 | 0.269 | **0.082** | 0.280 | 0.668 | 0.285 | 0.114 |
| GEMV 4K×11K (b=1) | 0.714 | 0.714 | **0.204** | 0.725 | 1.713 | 0.729 | 0.616 |
| GEMV 8K×8K (b=1) | 1.061 | 1.060 | **0.312** | 1.074 | 2.524 | 1.079 | 1.263 |
| GEMM 4K×11K (b=32) | 0.840 | 0.740 | **0.194** | 0.752 | 1.751 | 0.774 | 1.114 |
| GEMM 8K×8K (b=32) | 1.181 | 1.088 | **0.281** | — | — | — | — |

**Winner at every size: CUDA C INT8 IGEMM** (3-4× faster than any FP32 variant).

### PyTorch INT8 vs CUDA C INT8 vs ONNX INT8

| Config | PyTorch INT8 | CUDA C INT8 | ONNX INT8 CPU | Fastest |
|--------|-------------|------------|--------------|---------|
| 4K×4K (b=1) | 0.668 ms | **0.082 ms** | 0.114 ms | CUDA C (8.1×) |
| 4K×11K (b=1) | 1.713 ms | **0.204 ms** | 0.616 ms | CUDA C (8.4×) |
| 8K×8K (b=1) | 2.524 ms | **0.312 ms** | 1.263 ms | CUDA C (8.1×) |
| 4K×11K (b=32) | 1.751 ms | **0.194 ms** | 1.114 ms | CUDA C (9.0×) |

### Why Is CUDA C 8-9× Faster Than PyTorch INT8?

```
  PyTorch torchao (weight-only INT8):
  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────┐
  │ INT8     │ →  │ Dequantize   │ →  │ FP32     │ →  │ FP32 │
  │ Weights  │    │ INT8 → FP32  │    │ SGEMM    │    │ Out  │
  └──────────┘    └──────────────┘    └──────────┘    └──────┘
                    55% of time!        CUDA Cores

  CUDA C (cuBLASLt IGEMM):
  ┌──────────┐    ┌──────────────┐    ┌──────────┐
  │ INT8     │ →  │ INT8 × INT8  │ →  │ INT32    │
  │ Weights  │    │ IGEMM        │    │ Out      │
  └──────────┘    └──────────────┘    └──────────┘
                    Tensor Cores        No dequant!
```

The fundamental difference:
1. **PyTorch torchao weight-only**: stores weights as INT8, but **dequantizes to FP32 at inference time**, then does FP32 matmul. The INT8 storage saves memory but the compute is still FP32.
2. **CUDA C cuBLASLt**: does **true INT8 × INT8 → INT32** matrix multiplication directly on Tensor Cores. 4× less data to move AND 4× more throughput per Tensor Core cycle.

### Why Is ONNX INT8 CPU So Fast for Small Matrices?

```
  ONNX INT8 CPU at 4K×4K:  0.107 ms  (2.6× FASTER than PyTorch FP32 GPU!)
  PyTorch FP32 GPU:         0.280 ms

  Three reasons:
  1. Native INT8 GEMM: ONNX Runtime's MLAS library has hand-tuned
     AVX-512/VNNI INT8 kernels → true INT8×INT8→INT32 on CPU
  2. No GPU launch overhead: ~5-15 µs kernel launch + CPU↔GPU
     transfer is significant for small matrices
  3. Fits in L3 cache: 16 MB INT8 weight matrix stays in CPU cache
     → compute-bound, not memory-bound

  This advantage vanishes at larger sizes:
  8K×8K: ORT INT8 CPU = 1.263 ms > PT FP32 GPU = 1.074 ms
  (256 MB weight exceeds L3 cache → memory-bound again)
```

### Why Does ONNX FP32 GPU ≈ PyTorch FP32 GPU?

Both call the same cuBLAS library under the hood. ONNX Runtime's `CUDAExecutionProvider` dispatches `sgemm`/`cublasGemmEx` — identical to what PyTorch calls. The ~1% difference is ORT's session/graph optimization overhead, which is negligible at LLM scale.

---

## The Big Picture: When to Use What

| Scenario | Best Runtime | Why |
|----------|-------------|-----|
| Maximum inference speed (GPU) | CUDA C cuBLASLt INT8 | True INT8 IGEMM on Tensor Cores, no overhead |
| Small model on CPU | ONNX INT8 CPU | Native VNNI kernels, no GPU launch cost |
| Fit larger model in VRAM | PyTorch torchao INT8 | 4× weight compression, framework integration |
| Cross-platform deployment | ONNX Runtime | Runs on CPU/GPU/NPU/edge with same model |
| Production GPU serving | CUDA C / TensorRT | Lowest latency, most control |
| Quick prototyping | PyTorch FP32 | Simplest, no quantization complexity |

---

## Prerequisites

| Tool | Version | How to Install | Used By |
|------|---------|---------------|---------|
| Python | 3.10+ | `conda` or system | All Python scripts |
| PyTorch | 2.0+ with CUDA | `pip install torch` | pytorch, profiled |
| torchao | 0.10+ | `pip install torchao` | pytorch, profiled, onnx |
| ONNX | 1.14+ | `pip install onnx` | onnx |
| ONNX Runtime GPU | 1.16+ | `pip install onnxruntime-gpu` | onnx |
| CUDA Toolkit | 12.x | `conda create -n cuda_build -c nvidia/label/cuda-12.8.0 cuda-toolkit` | cuda.cu |
| NVIDIA GPU | Ampere+ (sm_80+) | Required for INT8 Tensor Cores | All GPU benchmarks |

### GPU Compatibility

| GPU Architecture | Compute Cap | INT8 Tensor Cores? | TF32? |
|-----------------|-------------|-------------------|-------|
| Volta (V100) | 7.0 | No | No |
| Turing (RTX 2080) | 7.5 | INT8 IMMA (limited) | No |
| **Ampere (RTX 3090, A100, A3000)** | **8.0-8.6** | **Yes** | **Yes** |
| Ada Lovelace (RTX 4090) | 8.9 | Yes | Yes |
| Hopper (H100) | 9.0 | Yes + FP8 | Yes |

---

## Chrome Trace Visualization

The profiling script exports `.json` trace files. To visualize:

1. Run `python fp32_to_int8_profiled.py`
2. Open `chrome://tracing` in Chrome or Edge
3. Click **Load** and select a trace file (e.g., `profile_fp32_b32_tf32on.json`)
4. Zoom into the **GPU** row to see individual kernel launches
5. Click on a kernel to see its name, duration, and arguments

Trace files generated:
- `profile_fp32_b1_tf32off.json` — FP32 batch=1, CUDA Cores
- `profile_fp32_b1_tf32on.json` — FP32 batch=1, CUDA Cores (GEMV ignores TF32)
- `profile_int8_b1.json` — INT8 batch=1, dequant + CUDA Cores
- `profile_fp32_b32_tf32off.json` — FP32 batch=32, CUDA Cores (sgemm)
- `profile_fp32_b32_tf32on.json` — FP32 batch=32, **Tensor Cores** (cutlass_tensorop)
- `profile_int8_b32.json` — INT8 batch=32, dequant + **Tensor Cores**

---

## Modifying for Your GPU

If you have a different GPU, change the architecture flag in the `Makefile`:

```makefile
# RTX 3090 / A100
ARCH := sm_80

# RTX 4090
ARCH := sm_89

# H100
ARCH := sm_90
```

---

## Key Concepts

### Symmetric Quantization
```
scale = max(|tensor|) / 127
int8_value = round(float_value / scale)
float_reconstructed = int8_value × scale
```

### Why INT8 Helps (Memory-Bound Regime)
At batch=1 (LLM token generation), the GPU spends most time **loading weights from VRAM**, not computing. INT8 weights are 4× smaller → 4× less memory to load → up to 4× faster (if the matmul is truly memory-bound).

### TF32 — The Hidden Default
PyTorch enables TF32 by default on Ampere+ GPUs. This silently routes FP32 GEMM operations through Tensor Cores using 10-bit mantissa (instead of FP32's 23-bit). Disable with:
```python
torch.backends.cuda.matmul.allow_tf32 = False
```

---

## License

MIT
