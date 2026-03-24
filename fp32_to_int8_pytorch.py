"""
FP32 Linear Multiplication -> INT8 Quantization (PyTorch)
=========================================================
Step 1: Normal FP32 linear layer (y = Wx + b)
Step 2: Quantize weights & activations to INT8 (manual symmetric)
Step 3: INT8 matmul via INT32 accumulation (educational)
Step 4: Compare outputs and measure error
Step 5: Weight reconstruction quality
Step 6: torchao official INT8 quantization
Step 7: Timing benchmark (FP32 vs manual INT8 vs torchao INT8)
"""

import torch
import torch.nn as nn
import time
import warnings

warnings.filterwarnings("ignore", message=".*cpp extensions.*")
warnings.filterwarnings("ignore", message=".*Config Deprecation.*")

torch.manual_seed(42)

# Device auto-detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────
# STEP 1: FP32 Linear Layer
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: FP32 Linear Multiplication (y = Wx + b)")
print("=" * 60)

# Dimensions (small for readable output)
in_features = 4
out_features = 3
batch_size = 2

# Create random FP32 weight, bias, input
W_fp32 = torch.randn(out_features, in_features, dtype=torch.float32, device=device)
b_fp32 = torch.randn(out_features, dtype=torch.float32, device=device)
x_fp32 = torch.randn(batch_size, in_features, dtype=torch.float32, device=device)

# FP32 forward pass: y = xW^T + b
y_fp32 = x_fp32 @ W_fp32.T + b_fp32

print(f"\nInput x ({x_fp32.dtype}):\n{x_fp32}")
print(f"\nWeight W ({W_fp32.dtype}):\n{W_fp32}")
print(f"\nBias b ({b_fp32.dtype}):\n{b_fp32}")
print(f"\nOutput y_fp32 = xW^T + b:\n{y_fp32}")


# ──────────────────────────────────────────────
# STEP 2: Quantize to INT8
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: INT8 Quantization (Symmetric, Per-Tensor)")
print("=" * 60)


def quantize_symmetric(tensor_fp32, num_bits=8):
    """
    Symmetric quantization: map FP32 range to [-127, 127]

    scale = max(|tensor|) / 127
    tensor_int8 = round(tensor / scale)
    """
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8

    abs_max = tensor_fp32.abs().max()  # find the range
    scale = abs_max / qmax  # FP32 per unit of INT8
    scale = torch.clamp(scale, min=1e-8)  # guard against zero

    # Quantize: float -> int
    tensor_int8 = torch.round(tensor_fp32 / scale).to(torch.int8)

    return tensor_int8, scale


def dequantize(tensor_int8, scale):
    """Dequantize: int -> float"""
    return tensor_int8.to(torch.float32) * scale


# Quantize weights and inputs separately
W_int8, scale_W = quantize_symmetric(W_fp32)
x_int8, scale_x = quantize_symmetric(x_fp32)

print(f"\n--- Weight Quantization ---")
print(f"W_fp32 range: [{W_fp32.min().item():.4f}, {W_fp32.max().item():.4f}]")
print(f"Scale (W):    {scale_W.item():.6f}")
print(f"W_int8:\n{W_int8}")

print(f"\n--- Input Quantization ---")
print(f"x_fp32 range: [{x_fp32.min().item():.4f}, {x_fp32.max().item():.4f}]")
print(f"Scale (x):    {scale_x.item():.6f}")
print(f"x_int8:\n{x_int8}")


# ──────────────────────────────────────────────
# STEP 3: INT8 Matmul -> Dequantize -> Add Bias
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: INT8 Matmul + Dequantize + Bias")
print("=" * 60)

# Integer matmul (accumulate in INT32 to avoid overflow)
# This is what hardware actually does — multiply int8 x int8, accumulate in int32
# Note: PyTorch CUDA doesn't support int32 matmul, so we run this on CPU
y_int32 = (x_int8.cpu().to(torch.int32) @ W_int8.cpu().to(torch.int32).T).to(device)

print(f"\nINT32 accumulator (x_int8 @ W_int8^T):\n{y_int32}")

# Dequantize the result back to FP32
# scale_output = scale_x * scale_W  (the combined scale)
scale_output = scale_x * scale_W
y_quantized = y_int32.to(torch.float32) * scale_output + b_fp32  # bias stays FP32

print(f"\nCombined scale (scale_x x scale_W): {scale_output.item():.8f}")
print(f"\nDequantized output y_int8:\n{y_quantized}")


# ──────────────────────────────────────────────
# STEP 4: Compare FP32 vs INT8
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Error Analysis - FP32 vs INT8")
print("=" * 60)

abs_error = torch.abs(y_fp32 - y_quantized)
rel_error = abs_error / (torch.abs(y_fp32) + 1e-8) * 100  # percentage

print(f"\nFP32 output:\n{y_fp32}")
print(f"\nINT8 output:\n{y_quantized}")
print(f"\nAbsolute error:\n{abs_error}")
print(f"\nRelative error (%):\n{rel_error}")
print(f"\nMax absolute error:  {abs_error.max().item():.6f}")
print(f"Mean absolute error: {abs_error.mean().item():.6f}")
print(f"Mean relative error: {rel_error.mean().item():.2f}%")


# ──────────────────────────────────────────────
# STEP 5: Verify weight reconstruction
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Weight Reconstruction Quality")
print("=" * 60)

W_reconstructed = dequantize(W_int8, scale_W)
w_error = torch.abs(W_fp32 - W_reconstructed)

print(f"\nOriginal W:\n{W_fp32}")
print(f"\nReconstructed W (int8 -> fp32):\n{W_reconstructed}")
print(f"\nWeight reconstruction error:\n{w_error}")
print(f"Max weight error: {w_error.max().item():.6f}")


# ──────────────────────────────────────────────
# STEP 6: torchao Official INT8 Quantization
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: torchao Official INT8 Quantization")
print("=" * 60)

try:
    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    TORCHAO_AVAILABLE = True
    print("torchao loaded successfully")
except ImportError:
    TORCHAO_AVAILABLE = False
    print("torchao not installed — skipping official quantization")
    print("Install with: pip install torchao")

if TORCHAO_AVAILABLE:
    # Create an nn.Linear layer with the SAME weights for fair comparison
    linear_fp32 = nn.Linear(in_features, out_features, bias=True, device=device)
    with torch.no_grad():
        linear_fp32.weight.copy_(W_fp32)
        linear_fp32.bias.copy_(b_fp32)

    # FP32 reference output
    with torch.no_grad():
        y_linear_fp32 = linear_fp32(x_fp32)
    print(f"\nnn.Linear FP32 output:\n{y_linear_fp32}")

    # --- INT8 Weight-Only quantization ---
    linear_w8 = nn.Linear(in_features, out_features, bias=True, device=device)
    with torch.no_grad():
        linear_w8.weight.copy_(W_fp32)
        linear_w8.bias.copy_(b_fp32)

    quantize_(linear_w8, Int8WeightOnlyConfig())
    with torch.no_grad():
        y_w8 = linear_w8(x_fp32)

    w8_error = torch.abs(y_linear_fp32 - y_w8)
    print(f"\n--- INT8 Weight-Only (torchao) ---")
    print(f"Output:\n{y_w8}")
    print(f"Mean abs error vs FP32: {w8_error.mean().item():.6f}")
    print(f"Max  abs error vs FP32: {w8_error.max().item():.6f}")

    # --- INT8 Dynamic Activation + Weight quantization ---
    linear_dyn8 = nn.Linear(in_features, out_features, bias=True, device=device)
    with torch.no_grad():
        linear_dyn8.weight.copy_(W_fp32)
        linear_dyn8.bias.copy_(b_fp32)

    quantize_(linear_dyn8, Int8DynamicActivationInt8WeightConfig())
    with torch.no_grad():
        y_dyn8 = linear_dyn8(x_fp32)

    dyn8_error = torch.abs(y_linear_fp32 - y_dyn8)
    print(f"\n--- INT8 Dynamic Activation + Weight (torchao) ---")
    print(f"Output:\n{y_dyn8}")
    print(f"Mean abs error vs FP32: {dyn8_error.mean().item():.6f}")
    print(f"Max  abs error vs FP32: {dyn8_error.max().item():.6f}")

    # Compare manual vs torchao
    manual_vs_torchao = torch.abs(y_quantized - y_dyn8)
    print(f"\n--- Manual INT8 vs torchao Dynamic INT8 ---")
    print(f"Mean abs difference: {manual_vs_torchao.mean().item():.6f}")
    print(f"Max  abs difference: {manual_vs_torchao.max().item():.6f}")


# ──────────────────────────────────────────────
# STEP 7: Timing Benchmark (LLM-scale, memory-bound)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Timing Benchmark — LLM-Scale (Memory-Bound)")
print("=" * 60)

if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"L2 cache: {props.L2_cache_size / 1024**2:.1f} MB")

warmup_iters = 50
bench_iters = 200


def benchmark(fn, warmup=warmup_iters, iters=bench_iters):
    """Benchmark a function with proper GPU synchronization."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times_t = torch.tensor(times)
    return times_t.mean().item(), times_t.std().item()


# ── Sweep multiple sizes to show crossover point ──
# batch=1 simulates LLM token generation (always memory-bound)
# Sizes modeled after real LLM layers:
#   4096x4096   = 64 MB FP32   (small, may still be compute-bound)
#   4096x11008  = 172 MB FP32  (LLaMA-7B FFN size)
#   8192x8192   = 256 MB FP32  (large, clearly memory-bound)

bench_configs = [
    {"name": "Small (4K×4K)",       "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "Medium (4K×11K)",     "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "Large (8K×8K)",       "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "Batched (32×4K×11K)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]

print(f"\nWarmup: {warmup_iters} iters, Benchmark: {bench_iters} iters")
print(f"batch=1 simulates LLM single-token generation (memory-bound)")
print()

all_results = []

for cfg in bench_configs:
    name = cfg["name"]
    bs, in_f, out_f = cfg["batch"], cfg["in_f"], cfg["out_f"]
    weight_mb_fp32 = in_f * out_f * 4 / 1024 / 1024
    weight_mb_int8 = in_f * out_f * 1 / 1024 / 1024

    print(f"{'='*60}")
    print(f"  {name}")
    print(f"  Weight: {in_f}x{out_f} = {weight_mb_fp32:.0f} MB FP32 / {weight_mb_int8:.0f} MB INT8")
    print(f"  Input:  {bs}x{in_f}")
    print(f"{'='*60}")

    # Create tensors
    x_b = torch.randn(bs, in_f, dtype=torch.float32, device=device)
    W_b = torch.randn(out_f, in_f, dtype=torch.float32, device=device)
    b_b = torch.randn(out_f, dtype=torch.float32, device=device)

    # --- FP32 nn.Linear ---
    linear_fp = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        linear_fp.weight.copy_(W_b)
        linear_fp.bias.copy_(b_b)
    linear_fp.eval()

    def fp32_fn(model=linear_fp, x=x_b):
        with torch.no_grad():
            return model(x)

    fp32_ms, fp32_s = benchmark(fp32_fn)
    print(f"  FP32 nn.Linear:        {fp32_ms:.3f} +/- {fp32_s:.3f} ms")

    # --- torchao INT8 weight-only ---
    torchao_w8_ms = None
    if TORCHAO_AVAILABLE:
        linear_w8 = nn.Linear(in_f, out_f, bias=True, device=device)
        with torch.no_grad():
            linear_w8.weight.copy_(W_b)
            linear_w8.bias.copy_(b_b)
        linear_w8.eval()
        quantize_(linear_w8, Int8WeightOnlyConfig())

        def w8_fn(model=linear_w8, x=x_b):
            with torch.no_grad():
                return model(x)

        torchao_w8_ms, torchao_w8_s = benchmark(w8_fn)
        speedup_w8 = fp32_ms / torchao_w8_ms
        print(f"  torchao INT8 W-only:   {torchao_w8_ms:.3f} +/- {torchao_w8_s:.3f} ms  ({speedup_w8:.2f}x)")

    # --- torchao INT8 dynamic (activation + weight) ---
    torchao_dyn_ms = None
    if TORCHAO_AVAILABLE:
        linear_dyn = nn.Linear(in_f, out_f, bias=True, device=device)
        with torch.no_grad():
            linear_dyn.weight.copy_(W_b)
            linear_dyn.bias.copy_(b_b)
        linear_dyn.eval()
        quantize_(linear_dyn, Int8DynamicActivationInt8WeightConfig())

        def dyn_fn(model=linear_dyn, x=x_b):
            with torch.no_grad():
                return model(x)

        torchao_dyn_ms, torchao_dyn_s = benchmark(dyn_fn)
        speedup_dyn = fp32_ms / torchao_dyn_ms
        print(f"  torchao INT8 dynamic:  {torchao_dyn_ms:.3f} +/- {torchao_dyn_s:.3f} ms  ({speedup_dyn:.2f}x)")

    # --- torchao INT8 weight-only + torch.compile ---
    torchao_compiled_ms = None
    if TORCHAO_AVAILABLE:
        linear_comp = nn.Linear(in_f, out_f, bias=True, device=device)
        with torch.no_grad():
            linear_comp.weight.copy_(W_b)
            linear_comp.bias.copy_(b_b)
        linear_comp.eval()
        quantize_(linear_comp, Int8WeightOnlyConfig())
        compiled_model = torch.compile(linear_comp, mode="max-autotune")

        def comp_fn(model=compiled_model, x=x_b):
            with torch.no_grad():
                return model(x)

        # Extra warmup for compilation
        print(f"  [compiling...] ", end="", flush=True)
        torchao_compiled_ms, torchao_compiled_s = benchmark(comp_fn, warmup=100)
        speedup_comp = fp32_ms / torchao_compiled_ms
        print(f"torchao INT8 W-only + compile: {torchao_compiled_ms:.3f} +/- {torchao_compiled_s:.3f} ms  ({speedup_comp:.2f}x)")

    result = {
        "name": name, "weight_mb": weight_mb_fp32,
        "fp32": fp32_ms,
        "w8": torchao_w8_ms,
        "dyn": torchao_dyn_ms,
        "compiled": torchao_compiled_ms,
    }
    all_results.append(result)

    # Free GPU memory between configs
    del x_b, W_b, b_b, linear_fp
    if TORCHAO_AVAILABLE:
        del linear_w8, linear_dyn, linear_comp, compiled_model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print()


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# --- Accuracy (from small demo) ---
compression = W_fp32.element_size() / W_int8.element_size()
mean_abs = abs_error.mean().item()
mean_rel = rel_error.mean().item()
max_rel = rel_error.max().item()

worst_flat = rel_error.argmax().item()
worst_idx = (worst_flat // rel_error.shape[1], worst_flat % rel_error.shape[1])
worst_fp32_val = y_fp32[worst_idx[0], worst_idx[1]].item()

print(f"""
--- Accuracy (small {batch_size}x{in_features} demo) ---
Mean absolute error: {mean_abs:.6f}
Mean relative error: {mean_rel:.2f}%
Max  relative error: {max_rel:.2f}% at position {worst_idx}
Compression:         {compression:.0f}x for weights
""")

# --- Timing summary table: inference times ---
print("--- Inference Time (ms) ---")
print(f"{'Config':<25} {'Weight':>8} {'FP32':>12} {'INT8 W-only':>12} {'INT8 Dynamic':>13} {'INT8 Compiled':>14}")
print("-" * 86)
for r in all_results:
    def fmt_ms(ms):
        if ms is None:
            return "N/A"
        return f"{ms:.3f} ms"

    print(f"{r['name']:<25} {r['weight_mb']:>6.0f}MB "
          f"{fmt_ms(r['fp32']):>12} "
          f"{fmt_ms(r['w8']):>12} "
          f"{fmt_ms(r['dyn']):>13} "
          f"{fmt_ms(r['compiled']):>14}")

# --- Speedup table ---
print()
print("--- Speedup vs FP32 (>1.0 = INT8 faster) ---")
print(f"{'Config':<25} {'Weight':>8} {'INT8 W-only':>12} {'INT8 Dynamic':>13} {'INT8 Compiled':>14}")
print("-" * 74)
for r in all_results:
    def fmt_speedup(int8_ms, fp32_ms):
        if int8_ms is None:
            return "N/A"
        s = fp32_ms / int8_ms
        marker = " <<" if s > 1.0 else ""
        return f"{s:.2f}x{marker}"

    print(f"{r['name']:<25} {r['weight_mb']:>6.0f}MB "
          f"{fmt_speedup(r['w8'], r['fp32']):>12} "
          f"{fmt_speedup(r['dyn'], r['fp32']):>13} "
          f"{fmt_speedup(r['compiled'], r['fp32']):>14}")

# --- Key insights ---
print("\n\nKEY INSIGHTS:")
print("-" * 40)

print(f"1. {compression:.0f}x memory reduction (FP32 -> INT8).")

if mean_rel < 1.0:
    print(f"2. Quantization is very accurate: {mean_rel:.2f}% mean error.")
elif mean_rel < 5.0:
    print(f"2. Moderate quantization error: {mean_rel:.2f}% (per-channel quantization helps).")
else:
    print(f"2. High quantization error: {mean_rel:.2f}% (needs asymmetric/per-channel).")

# Find the config where INT8 was fastest relative to FP32
best_speedup = 0
best_config = ""
best_method = ""
for r in all_results:
    for method, key in [("W8-only", "w8"), ("Dynamic", "dyn"), ("Compiled", "compiled")]:
        if r[key] and r["fp32"] / r[key] > best_speedup:
            best_speedup = r["fp32"] / r[key]
            best_config = r["name"]
            best_method = method

if best_speedup > 1.0:
    print(f"3. Best INT8 speedup: {best_speedup:.2f}x at {best_config} ({best_method}).")
    print(f"   Larger weights = more memory-bound = bigger INT8 advantage.")
else:
    print(f"3. INT8 did not outperform FP32 at these sizes on this GPU.")
    print(f"   This can happen on laptop GPUs with smaller memory bandwidth gaps.")

print(f"4. torch.compile() fuses quantize/dequantize kernels, reducing overhead.")

# Crossover analysis
print(f"5. Crossover analysis:")
for r in all_results:
    if r["w8"] is not None:
        ratio = r["fp32"] / r["w8"]
        status = "INT8 FASTER" if ratio > 1.0 else "FP32 faster"
        print(f"   {r['name']:<25} {r['weight_mb']:>4.0f} MB  → {ratio:.2f}x  ({status})")
