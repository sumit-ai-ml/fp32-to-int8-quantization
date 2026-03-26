"""
Optimized INT8 inference benchmark — closing the gap with CUDA C.

Compares 5 approaches:
  1. FP32 baseline (TF32 default on Ampere)
  2. torchao Int8WeightOnly (CURRENT — dequant then FP32 GEMM = SLOW)
  3. torchao Int8DynamicActivationInt8Weight (native INT8×INT8 via torch._int_mm)
  4. torch._int_mm direct (raw INT8×INT8→INT32 on Tensor Cores)
  5. torch.compile on the best INT8 method (fused kernels)

Goal: match CUDA C INT8 IGEMM performance (~0.10 ms for 4Kx4K)
"""

import torch
import torch.nn as nn
import time
import csv
import os
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print()


def benchmark(fn, warmup=100, iters=500):
    """Precise GPU timing with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def quantize_symmetric(tensor, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = tensor.abs().max()
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)
    tensor_int8 = torch.round(tensor / scale).to(torch.int8)
    return tensor_int8, scale


# ──────────────────────────────────────────────
# Benchmark configs
# ──────────────────────────────────────────────
configs = [
    {"name": "4Kx4K (b=1)",   "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "4Kx11K (b=1)",  "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "8Kx8K (b=1)",   "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "4Kx11K (b=32)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]

from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
timing_rows = []

for cfg in configs:
    nm = cfg["name"]
    bsz, inf, outf = cfg["batch"], cfg["in_f"], cfg["out_f"]
    wmb = inf * outf * 4 / 1024 / 1024
    print(f"{'='*70}")
    print(f"Config: {nm}  (weight={wmb:.0f}MB)")
    print(f"{'='*70}")

    W = torch.randn(outf, inf, dtype=torch.float32, device=device)
    b = torch.randn(outf, dtype=torch.float32, device=device)
    x = torch.randn(bsz, inf, dtype=torch.float32, device=device)

    base = {"config": nm, "batch": bsz, "in_features": inf,
            "out_features": outf, "weight_mb": f"{wmb:.0f}", "gpu": gpu_name}

    # ── 1. FP32 baseline ──
    lin = nn.Linear(inf, outf, bias=True, device=device)
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(b)
    lin.eval()
    fp32_ms, fp32_std = benchmark(lambda: lin(x))
    print(f"  FP32 baseline:              {fp32_ms:.3f} ms ± {fp32_std:.3f}")
    timing_rows.append({**base, "runtime": "FP32 baseline",
                        "time_ms": f"{fp32_ms:.4f}", "std_ms": f"{fp32_std:.4f}"})

    # ── 2. torchao Int8WeightOnly (CURRENT — SLOW) ──
    lin_wo = nn.Linear(inf, outf, bias=True, device=device)
    with torch.no_grad():
        lin_wo.weight.copy_(W)
        lin_wo.bias.copy_(b)
    lin_wo.eval()
    quantize_(lin_wo, Int8WeightOnlyConfig())
    wo_ms, wo_std = benchmark(lambda: lin_wo(x))
    print(f"  INT8 Weight-Only (current): {wo_ms:.3f} ms ± {wo_std:.3f}  "
          f"({wo_ms/fp32_ms:.1f}x vs FP32)")
    timing_rows.append({**base, "runtime": "INT8 Weight-Only (current)",
                        "time_ms": f"{wo_ms:.4f}", "std_ms": f"{wo_std:.4f}"})

    # ── 3. torchao Int8DynamicActivationInt8Weight (native INT8×INT8) ──
    lin_dyn = nn.Linear(inf, outf, bias=True, device=device)
    with torch.no_grad():
        lin_dyn.weight.copy_(W)
        lin_dyn.bias.copy_(b)
    lin_dyn.eval()
    quantize_(lin_dyn, Int8DynamicActivationInt8WeightConfig())
    dyn_ms, dyn_std = benchmark(lambda: lin_dyn(x))
    print(f"  INT8 Dynamic Act+Weight:    {dyn_ms:.3f} ms ± {dyn_std:.3f}  "
          f"({dyn_ms/fp32_ms:.1f}x vs FP32)")
    timing_rows.append({**base, "runtime": "INT8 Dynamic Act+Weight",
                        "time_ms": f"{dyn_ms:.4f}", "std_ms": f"{dyn_std:.4f}"})

    # ── 4. torch._int_mm direct (raw Tensor Core INT8) ──
    # _int_mm requires M >= 16 (batch dimension), only works for batched
    raw_ms = None
    if bsz >= 16:
        W_int8, scale_W = quantize_symmetric(W)
        x_int8, scale_x = quantize_symmetric(x)
        W_int8_t = W_int8.t().contiguous()

        def raw_int_mm():
            out_i32 = torch._int_mm(x_int8, W_int8_t)
            return out_i32.to(torch.float32) * (scale_x * scale_W) + b

        try:
            raw_int_mm()
            raw_ms, raw_std = benchmark(raw_int_mm)
            print(f"  torch._int_mm (raw TC):     {raw_ms:.3f} ms ± {raw_std:.3f}  "
                  f"({raw_ms/fp32_ms:.1f}x vs FP32)")
            timing_rows.append({**base, "runtime": "torch._int_mm (raw Tensor Core)",
                                "time_ms": f"{raw_ms:.4f}", "std_ms": f"{raw_std:.4f}"})
        except Exception as e:
            print(f"  torch._int_mm: FAILED — {e}")
    else:
        print(f"  torch._int_mm: SKIPPED (batch={bsz} < 16, Tensor Core INT8 needs M>=16)")

    # ── 5. torch.compile on FP32 (fused kernels) ──
    lin_compiled = nn.Linear(inf, outf, bias=True, device=device)
    with torch.no_grad():
        lin_compiled.weight.copy_(W)
        lin_compiled.bias.copy_(b)
    lin_compiled.eval()
    lin_compiled = torch.compile(lin_compiled, mode="reduce-overhead")

    for _ in range(5):
        with torch.no_grad():
            _ = lin_compiled(x)
    torch.cuda.synchronize()

    comp_ms, comp_std = benchmark(lambda: lin_compiled(x))
    print(f"  FP32 + torch.compile:       {comp_ms:.3f} ms ± {comp_std:.3f}  "
          f"({comp_ms/fp32_ms:.1f}x vs FP32)")
    timing_rows.append({**base, "runtime": "FP32 + torch.compile",
                        "time_ms": f"{comp_ms:.4f}", "std_ms": f"{comp_std:.4f}"})

    # ── 6. FP16 (Tensor Core HMMA) ──
    lin_fp16 = lin.half()
    x_fp16 = x.half()
    fp16_ms, fp16_std = benchmark(lambda: lin_fp16(x_fp16))
    print(f"  FP16 (Tensor Core HMMA):    {fp16_ms:.3f} ms ± {fp16_std:.3f}  "
          f"({fp16_ms/fp32_ms:.1f}x vs FP32)")
    timing_rows.append({**base, "runtime": "FP16 Tensor Core",
                        "time_ms": f"{fp16_ms:.4f}", "std_ms": f"{fp16_std:.4f}"})

    # Summary
    best_ms = min(t for t in [fp32_ms, wo_ms, dyn_ms, raw_ms, comp_ms, fp16_ms] if t is not None)
    print(f"\n  BEST: {best_ms:.3f} ms → {fp32_ms/best_ms:.1f}x faster than FP32")
    print()

    # Cleanup
    del W, b, x, lin, lin_wo, lin_dyn, lin_compiled, lin_fp16
    torch.cuda.empty_cache()

# Save results
csv_path = os.path.join(results_dir, "optimized_timing.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "batch", "in_features", "out_features",
                                       "weight_mb", "runtime", "time_ms", "std_ms", "gpu"])
    w.writeheader()
    w.writerows(timing_rows)

print(f"Results saved to: {csv_path}")
