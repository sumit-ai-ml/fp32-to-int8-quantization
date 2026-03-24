"""
FP32 vs INT8 GPU Profiling — Which Cores Are Running?
======================================================
Uses torch.profiler to identify which GPU cores (CUDA Cores vs Tensor Cores)
are used for FP32 and INT8 matrix multiplication.

Profiling sessions (run at two batch sizes):
  batch=1   → GEMV (matrix-vector), always uses CUDA Cores
  batch=32  → GEMM (matrix-matrix), can use Tensor Cores

For each batch size, we profile:
  A) FP32 matmul, TF32 DISABLED  → CUDA Cores (sgemm)
  B) FP32 matmul, TF32 ENABLED   → Tensor Cores (TF32 mode) [batch≥2 only]
  C) INT8 matmul (torchao)        → Tensor Cores (INT8 mode) [batch≥2 only]

Then a clean timing benchmark (no profiler overhead) for accurate speed comparison.

Requires: PyTorch with CUDA, torchao
GPU:      Ampere+ (sm80/sm86) recommended for INT8 Tensor Core support
"""

import torch
import torch.nn as nn
import time
import warnings

warnings.filterwarnings("ignore", message=".*cpp extensions.*")
warnings.filterwarnings("ignore", message=".*Config Deprecation.*")

from torch.profiler import profile, ProfilerActivity

torch.manual_seed(42)

# ──────────────────────────────────────────────
# Device check
# ──────────────────────────────────────────────
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU.")
    print("Run fp32_to_int8_pytorch.py for CPU-compatible version.")
    exit(1)

device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU:             {props.name}")
print(f"Compute cap:     {props.major}.{props.minor}")
print(f"VRAM:            {props.total_memory / 1024**3:.1f} GB")
print(f"SM count:        {props.multi_processor_count}")

if props.major < 8:
    print(f"WARNING: Compute capability {props.major}.{props.minor} < 8.0")
    print("         INT8 Tensor Cores require Ampere (sm80) or newer.")
    print("         FP32 will use CUDA Cores regardless of TF32 setting.")

# ──────────────────────────────────────────────
# torchao check
# ──────────────────────────────────────────────
try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    TORCHAO_AVAILABLE = True
    print(f"torchao:         loaded")
except ImportError:
    TORCHAO_AVAILABLE = False
    print(f"torchao:         NOT installed (INT8 profiling will be skipped)")
    print(f"                 Install with: pip install torchao")


# ──────────────────────────────────────────────
# Kernel name → Core type classifier
# ──────────────────────────────────────────────
def classify_kernel(kernel_name: str) -> str:
    """
    Heuristic: classify a CUDA kernel name into the GPU core type it uses.

    Returns: 'TENSOR CORE', 'CUDA CORE', 'MEMORY/ELEM', or 'OTHER'
    """
    name = kernel_name.lower()

    # Memory / elementwise operations (not matrix compute)
    if any(kw in name for kw in ['memcpy', 'memset', 'fill', 'copy_', '_to_copy',
                                  'elementwise', 'vectorized', 'reduce']):
        return 'MEMORY/ELEM'

    # Tensor Core patterns — INT8 and TF32
    tensor_core_patterns = [
        'imma',          # Integer Matrix Multiply Accumulate (INT8 Tensor Core)
        'hmma',          # Half Matrix Multiply Accumulate (FP16 Tensor Core)
        's8_',           # Signed 8-bit CUTLASS
        '_s8',           # Signed 8-bit suffix
        'i8816',         # INT8 tile size
        'tensorop',      # Explicit tensor op marker
        's16816',        # Tensor Core tile size (INT8)
        '16816',         # Tensor Core tile size (INT8)
        'h884',          # Tensor Core tile size (FP16)
        'h1688',         # Tensor Core tile size (FP16)
        'tf32',          # TF32 mode (uses Tensor Cores)
        'xmma',          # Extended Matrix Multiply Accumulate
    ]
    if any(pattern in name for pattern in tensor_core_patterns):
        return 'TENSOR CORE'

    # CUDA Core GEMM/GEMV patterns
    cuda_core_patterns = [
        'sgemm',         # Single-precision GEMM (CUDA Cores)
        'dgemm',         # Double-precision GEMM
        'gemvx',         # GEMV (matrix-vector) — always CUDA Cores
        'gemv2',         # GEMV variant 2
        'gemv_',         # GEMV generic
        'cublaslt',      # cuBLASLt can be either, but without TC markers → CUDA
    ]
    if any(pattern in name for pattern in cuda_core_patterns):
        return 'CUDA CORE'

    # CUTLASS kernels without explicit TC markers are likely Tensor Core on Ampere+
    if 'cutlass' in name:
        return 'TENSOR CORE'

    return 'OTHER'


def extract_cuda_kernels(prof):
    """
    Extract raw CUDA kernel names from profiler trace events.
    This gives us the actual GPU kernel names (e.g., 'internal::gemvx_kernel<...>')
    instead of PyTorch operator names (e.g., 'aten::addmm').
    """
    kernels = {}  # kernel_name -> {total_time, count}

    for event in prof.events():
        # We want CUDA kernel events (device_time > 0, and actual kernel names)
        if event.device_time_total > 0:
            name = event.name
            # Skip high-level PyTorch ops — we want the actual CUDA kernel launches
            if name.startswith("aten::") or name.startswith("torch::"):
                continue
            # Skip profiler overhead events
            if "Activity Buffer" in name or "Memcpy" in name.title():
                pass  # keep memcpy, it's informative

            if name not in kernels:
                kernels[name] = {"total_time": 0, "count": 0}
            kernels[name]["total_time"] += event.device_time_total
            kernels[name]["count"] += 1

    # Sort by total time descending
    sorted_kernels = sorted(kernels.items(), key=lambda x: -x[1]["total_time"])
    return sorted_kernels


def print_kernel_table(prof, title: str, top_n: int = 15):
    """Print CUDA kernel names with core type classification."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    kernels = extract_cuda_kernels(prof)

    if not kernels:
        # Fallback to key_averages if no raw kernels found
        events = prof.key_averages()
        cuda_events = [e for e in events if e.device_time_total > 0]
        cuda_events.sort(key=lambda e: e.device_time_total, reverse=True)
        kernels = [(e.key, {"total_time": e.device_time_total, "count": e.count})
                    for e in cuda_events]

    total_time = sum(info["total_time"] for _, info in kernels)

    print(f"\n{'CUDA Kernel':<60} {'Core Type':<14} {'GPU Time':>10} {'Calls':>6} {'%':>6}")
    print("-" * 100)

    core_type_time = {}

    for name, info in kernels[:top_n]:
        display_name = name[:57] + "..." if len(name) > 60 else name
        core_type = classify_kernel(name)
        gpu_us = info["total_time"]
        pct = gpu_us / total_time * 100 if total_time > 0 else 0

        core_type_time[core_type] = core_type_time.get(core_type, 0) + gpu_us

        print(f"{display_name:<60} {core_type:<14} {gpu_us:>8.0f}us {info['count']:>6} {pct:>5.1f}%")

    # Include remaining kernels in breakdown even if not printed
    for name, info in kernels[top_n:]:
        core_type = classify_kernel(name)
        core_type_time[core_type] = core_type_time.get(core_type, 0) + info["total_time"]

    # Summary
    print(f"\n--- Core Type Breakdown ---")
    for ct, t in sorted(core_type_time.items(), key=lambda x: -x[1]):
        pct = t / total_time * 100 if total_time > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {ct:<20} {t:>10.0f}us  ({pct:>5.1f}%)  {bar}")

    return core_type_time


# ──────────────────────────────────────────────
# Profiling config
# ──────────────────────────────────────────────
PROFILE_IN = 4096
PROFILE_OUT = 11008  # LLaMA-7B FFN layer size
PROFILE_ITERS = 20
WARMUP_ITERS = 10


def run_profiling_session(label, linear, x_input):
    """Run a single profiling session: warmup + profile + print."""
    # Warmup outside profiler
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = linear(x_input)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            with torch.no_grad():
                _ = linear(x_input)
        torch.cuda.synchronize()

    breakdown = print_kernel_table(prof, label)
    return prof, breakdown


# Create shared weights
W = torch.randn(PROFILE_OUT, PROFILE_IN, dtype=torch.float32, device=device)
b = torch.randn(PROFILE_OUT, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════
# PART 1: PROFILING (kernel identification)
# ══════════════════════════════════════════════
print("\n" + "█" * 100)
print("  PART 1: GPU KERNEL PROFILING")
print("  Identifies which GPU core type each operation uses")
print("  NOTE: Profiler adds overhead — see Part 2 for accurate timing")
print("█" * 100)

all_profs = {}          # name -> profiler object (for trace export)
all_breakdowns = {}     # name -> core_type_time dict

# ────────────────────────────────────────────────────
# GROUP 1: batch=1 (GEMV — matrix-vector multiply)
# GEMV always uses CUDA Cores (no Tensor Core path)
# ────────────────────────────────────────────────────
print("\n\n" + "─" * 100)
print("  GROUP 1: batch=1 (GEMV — always CUDA Cores)")
print("  At batch=1, cuBLAS dispatches GEMV (matrix-vector), which has no Tensor Core path.")
print("  Both TF32=ON and TF32=OFF will use CUDA Cores here.")
print("─" * 100)

x_b1 = torch.randn(1, PROFILE_IN, dtype=torch.float32, device=device)

# Session A: FP32, batch=1, TF32 OFF
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

lin_a = nn.Linear(PROFILE_IN, PROFILE_OUT, bias=True, device=device)
with torch.no_grad():
    lin_a.weight.copy_(W)
    lin_a.bias.copy_(b)
lin_a.eval()

print("\n▶ SESSION A: FP32, batch=1, TF32 OFF")
prof_a, bd_a = run_profiling_session(
    "FP32 batch=1 TF32-OFF → Expect CUDA Cores (gemvx)", lin_a, x_b1
)
all_profs["fp32_b1_tf32off"] = prof_a
all_breakdowns["FP32 b=1 TF32off"] = bd_a

# Session B: FP32, batch=1, TF32 ON
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("\n▶ SESSION B: FP32, batch=1, TF32 ON")
prof_b, bd_b = run_profiling_session(
    "FP32 batch=1 TF32-ON → Still CUDA Cores (GEMV ignores TF32)", lin_a, x_b1
)
all_profs["fp32_b1_tf32on"] = prof_b
all_breakdowns["FP32 b=1 TF32on"] = bd_b

# Session C: INT8, batch=1
if TORCHAO_AVAILABLE:
    lin_c = nn.Linear(PROFILE_IN, PROFILE_OUT, bias=True, device=device)
    with torch.no_grad():
        lin_c.weight.copy_(W)
        lin_c.bias.copy_(b)
    lin_c.eval()
    quantize_(lin_c, Int8WeightOnlyConfig())

    print("\n▶ SESSION C: INT8 torchao, batch=1")
    prof_c, bd_c = run_profiling_session(
        "INT8 batch=1 → Weight dequant + GEMV (CUDA Cores)", lin_c, x_b1
    )
    all_profs["int8_b1"] = prof_c
    all_breakdowns["INT8 b=1"] = bd_c

del x_b1
torch.cuda.empty_cache()

# ────────────────────────────────────────────────────
# GROUP 2: batch=32 (GEMM — matrix-matrix multiply)
# GEMM CAN use Tensor Cores when tiles are large enough
# ────────────────────────────────────────────────────
print("\n\n" + "─" * 100)
print("  GROUP 2: batch=32 (GEMM — Tensor Cores activate here)")
print("  At batch≥2, cuBLAS dispatches GEMM, which CAN use Tensor Cores.")
print("  TF32=ON should route through Tensor Cores. TF32=OFF stays on CUDA Cores.")
print("─" * 100)

x_b32 = torch.randn(32, PROFILE_IN, dtype=torch.float32, device=device)

# Session D: FP32, batch=32, TF32 OFF
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

lin_d = nn.Linear(PROFILE_IN, PROFILE_OUT, bias=True, device=device)
with torch.no_grad():
    lin_d.weight.copy_(W)
    lin_d.bias.copy_(b)
lin_d.eval()

print("\n▶ SESSION D: FP32, batch=32, TF32 OFF")
prof_d, bd_d = run_profiling_session(
    "FP32 batch=32 TF32-OFF → Expect CUDA Cores (sgemm)", lin_d, x_b32
)
all_profs["fp32_b32_tf32off"] = prof_d
all_breakdowns["FP32 b=32 TF32off"] = bd_d

# Session E: FP32, batch=32, TF32 ON
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("\n▶ SESSION E: FP32, batch=32, TF32 ON")
prof_e, bd_e = run_profiling_session(
    "FP32 batch=32 TF32-ON → Expect TENSOR CORES (TF32 mode)", lin_d, x_b32
)
all_profs["fp32_b32_tf32on"] = prof_e
all_breakdowns["FP32 b=32 TF32on"] = bd_e

# Session F: INT8, batch=32
if TORCHAO_AVAILABLE:
    lin_f = nn.Linear(PROFILE_IN, PROFILE_OUT, bias=True, device=device)
    with torch.no_grad():
        lin_f.weight.copy_(W)
        lin_f.bias.copy_(b)
    lin_f.eval()
    quantize_(lin_f, Int8WeightOnlyConfig())

    print("\n▶ SESSION F: INT8 torchao, batch=32")
    prof_f, bd_f = run_profiling_session(
        "INT8 batch=32 → Expect TENSOR CORES (INT8 IMMA)", lin_f, x_b32
    )
    all_profs["int8_b32"] = prof_f
    all_breakdowns["INT8 b=32"] = bd_f

del x_b32
torch.cuda.empty_cache()


# ── Side-by-side summary ──
print("\n\n" + "=" * 100)
print("  CORE TYPE SUMMARY — Side by Side")
print("=" * 100)

# Print Group 1 (batch=1) and Group 2 (batch=32) separately
for group_label, group_keys in [
    ("batch=1 (GEMV — memory-bound, CUDA Cores only)",
     [k for k in all_breakdowns if "b=1" in k]),
    ("batch=32 (GEMM — compute-bound, Tensor Cores possible)",
     [k for k in all_breakdowns if "b=32" in k]),
]:
    if not group_keys:
        continue

    print(f"\n--- {group_label} ---")

    all_core_types = set()
    for k in group_keys:
        all_core_types.update(all_breakdowns[k].keys())

    compute_types = sorted(ct for ct in all_core_types
                           if ct not in ('MEMORY/ELEM', 'OTHER'))

    print(f"\n{'Core Type':<18}", end="")
    for k in group_keys:
        short = k[:18]
        print(f" {short:>18}", end="")
    print()
    print("-" * (18 + 19 * len(group_keys)))

    for ct in compute_types:
        print(f"{ct:<18}", end="")
        for k in group_keys:
            breakdown = all_breakdowns[k]
            total = sum(breakdown.values())
            val = breakdown.get(ct, 0)
            pct = val / total * 100 if total > 0 else 0
            print(f" {pct:>16.1f}%", end="")
        print()

    # Verdict row
    print(f"\n{'VERDICT':<18}", end="")
    for k in group_keys:
        breakdown = all_breakdowns[k]
        tc = breakdown.get('TENSOR CORE', 0)
        cc = breakdown.get('CUDA CORE', 0)
        if tc > cc and tc > 0:
            verdict = "TENSOR CORES"
        elif cc > tc and cc > 0:
            verdict = "CUDA CORES"
        elif tc == 0 and cc == 0:
            verdict = "N/A"
        else:
            verdict = "MIXED"
        print(f" {verdict:>18}", end="")
    print()


# ── Export Chrome traces ──
print(f"\n\n--- Chrome Trace Export ---")

import os
trace_dir = "/home/sumit-pandey/distil_weight"
for name, prof_obj in all_profs.items():
    filepath = os.path.join(trace_dir, f"profile_{name}.json")
    try:
        prof_obj.export_chrome_trace(filepath)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Failed: {filepath}: {e}")

print("\n  To view traces:")
print("    1. Open chrome://tracing in Chrome/Edge")
print("    2. Click 'Load' and select a .json file")
print("    3. Zoom into GPU row to see individual kernel launches")


# ══════════════════════════════════════════════
# PART 2: TIMING BENCHMARK (no profiler overhead)
# ══════════════════════════════════════════════
print("\n\n" + "█" * 100)
print("  PART 2: ACCURATE TIMING BENCHMARK")
print("  No profiler attached — these are real execution times")
print("█" * 100)

BENCH_WARMUP = 50
BENCH_ITERS = 200


def benchmark(fn, warmup=BENCH_WARMUP, iters=BENCH_ITERS):
    """Benchmark with proper GPU synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item(), t.median().item()


bench_configs = [
    # batch=1: GEMV (memory-bound, CUDA Cores)
    {"name": "GEMV  4K×4K  (b=1)",     "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "GEMV  4K×11K (b=1)",     "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "GEMV  8K×8K  (b=1)",     "batch": 1,  "in_f": 8192,  "out_f": 8192},
    # batch=32: GEMM (compute-bound, Tensor Cores)
    {"name": "GEMM  4K×11K (b=32)",    "batch": 32, "in_f": 4096,  "out_f": 11008},
    {"name": "GEMM  8K×8K  (b=32)",    "batch": 32, "in_f": 8192,  "out_f": 8192},
]

print(f"\nWarmup: {BENCH_WARMUP} iters, Benchmark: {BENCH_ITERS} iters\n")

all_results = []

for cfg in bench_configs:
    name = cfg["name"]
    bs, in_f, out_f = cfg["batch"], cfg["in_f"], cfg["out_f"]
    weight_mb_fp32 = in_f * out_f * 4 / 1024 / 1024
    weight_mb_int8 = in_f * out_f * 1 / 1024 / 1024
    op_type = "GEMV" if bs == 1 else "GEMM"

    print(f"{'─' * 80}")
    print(f"  {name}  (weight: {weight_mb_fp32:.0f}MB FP32 / {weight_mb_int8:.0f}MB INT8)  [{op_type}]")
    print(f"{'─' * 80}")

    x_b = torch.randn(bs, in_f, dtype=torch.float32, device=device)
    W_b = torch.randn(out_f, in_f, dtype=torch.float32, device=device)
    b_b = torch.randn(out_f, dtype=torch.float32, device=device)

    # ── FP32 TF32-OFF ──
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    lin_1 = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        lin_1.weight.copy_(W_b)
        lin_1.bias.copy_(b_b)
    lin_1.eval()

    def fn1(m=lin_1, inp=x_b):
        with torch.no_grad():
            return m(inp)

    ms_a, std_a, med_a = benchmark(fn1)
    core_a = "CUDA Cores" if bs == 1 else "CUDA Cores"
    print(f"  FP32 TF32-OFF ({core_a:12}): {ms_a:.3f} ± {std_a:.3f} ms")

    # ── FP32 TF32-ON ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    lin_2 = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        lin_2.weight.copy_(W_b)
        lin_2.bias.copy_(b_b)
    lin_2.eval()

    def fn2(m=lin_2, inp=x_b):
        with torch.no_grad():
            return m(inp)

    ms_b, std_b, med_b = benchmark(fn2)
    core_b = "CUDA Cores" if bs == 1 else "Tensor Cores"
    tf32_speedup = ms_a / ms_b if ms_b > 0 else 0
    print(f"  FP32 TF32-ON  ({core_b:12}): {ms_b:.3f} ± {std_b:.3f} ms  ({tf32_speedup:.2f}x vs TF32-OFF)")

    # ── INT8 torchao ──
    int8_ms = None
    if TORCHAO_AVAILABLE:
        lin_3 = nn.Linear(in_f, out_f, bias=True, device=device)
        with torch.no_grad():
            lin_3.weight.copy_(W_b)
            lin_3.bias.copy_(b_b)
        lin_3.eval()
        quantize_(lin_3, Int8WeightOnlyConfig())

        def fn3(m=lin_3, inp=x_b):
            with torch.no_grad():
                return m(inp)

        ms_c, std_c, med_c = benchmark(fn3)
        int8_speedup = ms_a / ms_c if ms_c > 0 else 0
        int8_ms = ms_c
        core_c = "CUDA+deq" if bs == 1 else "Tensor Cores"
        print(f"  INT8 torchao  ({core_c:12}): {ms_c:.3f} ± {std_c:.3f} ms  ({int8_speedup:.2f}x vs FP32)")

    all_results.append({
        "name": name, "op": op_type, "weight_mb": weight_mb_fp32,
        "fp32_notf32": ms_a, "fp32_tf32": ms_b, "int8": int8_ms,
    })

    # Free memory
    del x_b, W_b, b_b, lin_1, lin_2
    if TORCHAO_AVAILABLE:
        del lin_3
    torch.cuda.empty_cache()
    print()


# ── Final summary table ──
print("\n" + "=" * 100)
print("  FINAL SUMMARY")
print("=" * 100)

print(f"\n{'Config':<27} {'Op':>5} {'Weight':>7} {'FP32 CUDA':>11} {'FP32 TF32':>11} {'INT8':>11} {'Best':>14}")
print("-" * 90)

for r in all_results:
    def fmt(ms):
        return f"{ms:.3f}ms" if ms else "N/A"

    times = [("CUDA", r["fp32_notf32"]), ("TF32", r["fp32_tf32"])]
    if r["int8"]:
        times.append(("INT8", r["int8"]))

    best_name, best_ms = min(times, key=lambda x: x[1] if x[1] else float('inf'))

    print(f"{r['name']:<27} {r['op']:>5} {r['weight_mb']:>5.0f}MB "
          f"{fmt(r['fp32_notf32']):>11} "
          f"{fmt(r['fp32_tf32']):>11} "
          f"{fmt(r['int8']):>11} "
          f"{best_name:>14}")


print(f"""

KEY TAKEAWAYS:
{'─' * 60}
1. GEMV (batch=1) ALWAYS uses CUDA Cores.
   Matrix-vector multiply has no Tensor Core path in cuBLAS.
   TF32 ON/OFF makes no difference here.
   This is the LLM single-token generation regime.

2. GEMM (batch≥2) CAN use Tensor Cores.
   - TF32 OFF → sgemm on CUDA Cores
   - TF32 ON  → Tensor Cores in TF32 mode (10-bit mantissa)
   - INT8     → Tensor Cores in INT8 IMMA mode

3. PyTorch enables TF32 BY DEFAULT on Ampere+ GPUs.
   This means FP32 GEMM silently uses Tensor Cores
   with reduced precision unless you explicitly disable it.

4. INT8 weight-only quantization (torchao):
   - Stores weights as INT8 (4x smaller)
   - Dequantizes to FP32 at runtime, then does FP32 matmul
   - At batch=1: dequant overhead can make it SLOWER
   - At batch=32: memory savings + Tensor Cores help

5. Your {props.name} (sm{props.major}.{props.minor}):
   - {props.multi_processor_count} SMs, each with CUDA Cores + Tensor Cores
   - CUDA Cores: general-purpose FP32/INT32 ALUs
   - Tensor Cores: specialized 4x4 matrix multiply units
     supporting FP16, BF16, TF32, INT8, INT4

6. To see exact kernel names per operation:
   Open the Chrome trace .json files in chrome://tracing
   and zoom into the GPU timeline row.

7. For deeper profiling (exact core utilization %):
   Install NVIDIA Nsight Compute (ncu) or Nsight Systems (nsys):
     conda install -c nvidia cuda-toolkit nsight-systems nsight-compute
   Then run: nsys profile python fp32_to_int8_profiled.py
""")
