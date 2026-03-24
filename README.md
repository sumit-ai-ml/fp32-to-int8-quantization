# FP32 → INT8 Weight Quantization: PyTorch vs CUDA C

A side-by-side comparison of FP32-to-INT8 weight quantization and matrix multiplication in **PyTorch** and **raw CUDA C**, with GPU profiling to show exactly which GPU cores (CUDA Cores vs Tensor Cores) each operation uses.

**Tested on:** NVIDIA RTX A3000 Laptop GPU (Ampere, sm_86, 6GB VRAM, 32 SMs)

---

## Files

| File | Language | What It Does |
|------|----------|-------------|
| `fp32_to_int8_pytorch.py` | Python/PyTorch | FP32 linear layer → INT8 quantization (manual + torchao) → error analysis → LLM-scale benchmarks |
| `fp32_to_int8_profiled.py` | Python/PyTorch | GPU kernel profiling with `torch.profiler` — identifies which GPU cores each operation uses |
| `fp32_to_int8_cuda.cu` | CUDA C | Same quantization + matmul in raw CUDA — cuBLAS SGEMM (FP32) + cuBLASLt IGEMM (INT8) + custom kernels |
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

### 3. CUDA C Version (raw GPU programming)

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
└─────────────────────────┴──────────────────────────┴────────────────┘

* torchao weight-only quantization dequantizes INT8→FP32 FIRST, then does
  FP32 matmul. It does NOT do native INT8 matmul on Tensor Cores.
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

## PyTorch vs CUDA C — Head-to-Head

| Config | PyTorch INT8 | CUDA C INT8 | CUDA C speedup |
|--------|-------------|------------|----------------|
| GEMV 4K×4K (b=1) | 0.666 ms | 0.084 ms | **7.9×** |
| GEMV 4K×11K (b=1) | 1.721 ms | 0.204 ms | **8.4×** |
| GEMV 8K×8K (b=1) | 2.528 ms | 0.311 ms | **8.1×** |
| GEMM 4K×11K (b=32) | 1.750 ms | 0.194 ms | **9.0×** |
| GEMM 8K×8K (b=32) | 2.561 ms | 0.281 ms | **9.1×** |

### Why Is CUDA C 8-9× Faster?

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

---

## Prerequisites

| Tool | Version | How to Install |
|------|---------|---------------|
| Python | 3.10+ | `conda` or system |
| PyTorch | 2.0+ with CUDA | `pip install torch` |
| torchao | 0.10+ | `pip install torchao` |
| CUDA Toolkit | 12.x | `conda create -n cuda_build -c nvidia/label/cuda-12.8.0 cuda-toolkit` |
| NVIDIA GPU | Ampere+ (sm_80+) | Required for INT8 Tensor Cores |

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
