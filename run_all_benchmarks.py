"""
Run all benchmark scripts and save results to CSV files.

This is the single entry point. It runs:
  1. fp32_to_int8_pytorch.py  → results/pytorch_timing.csv, results/pytorch_error.csv
  2. fp32_to_int8_onnx.py     → results/onnx_timing.csv, results/onnx_error.csv
  3. fp32_to_int8_cuda (binary)→ results/cuda_timing.csv
  4. fp32_to_int8_profiled.py  → results/profiling.csv

Usage:
  python run_all_benchmarks.py          # run all
  python run_all_benchmarks.py pytorch  # run only pytorch
  python run_all_benchmarks.py onnx     # run only onnx
  python run_all_benchmarks.py cuda     # run only cuda
  python run_all_benchmarks.py profile  # run only profiling
"""

import subprocess
import csv
import os
import sys
import re
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, timeout=600):
    """Run a command and return stdout."""
    print(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=SCRIPT_DIR, shell=isinstance(cmd, str)
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
    return result.stdout, result.stderr


# ──────────────────────────────────────────────
# 1. PyTorch benchmark
# ──────────────────────────────────────────────
def run_pytorch():
    print("\n[1/4] Running PyTorch benchmarks...")
    # We run a small inline script that imports the benchmark logic
    # and writes CSV directly, to avoid modifying the original script
    code = '''
import torch
import torch.nn as nn
import time
import csv
import warnings
import os

warnings.filterwarnings("ignore")
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

# Quantize helper
def quantize_symmetric(tensor_fp32, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = tensor_fp32.abs().max()
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)
    tensor_int8 = torch.round(tensor_fp32 / scale).to(torch.int8)
    return tensor_int8, scale

# Benchmark helper
def benchmark(fn, warmup=50, iters=200):
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
        times.append((t1 - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()

# ── Small demo for error measurement ──
in_f, out_f, bs = 4, 3, 2
W = torch.randn(out_f, in_f, dtype=torch.float32, device=device)
b = torch.randn(out_f, dtype=torch.float32, device=device)
x = torch.randn(bs, in_f, dtype=torch.float32, device=device)

y_fp32 = x @ W.T + b
W_int8, scale_W = quantize_symmetric(W)
x_int8, scale_x = quantize_symmetric(x)
y_int32 = (x_int8.cpu().to(torch.int32) @ W_int8.cpu().to(torch.int32).T).to(device)
y_quant = y_int32.to(torch.float32) * (scale_x * scale_W) + b

abs_error = torch.abs(y_fp32 - y_quant)
rel_error = abs_error / (torch.abs(y_fp32) + 1e-8) * 100

# torchao
try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
    TORCHAO = True
except ImportError:
    TORCHAO = False

# Error CSV
error_rows = [
    {"method": "Manual INT8", "mean_abs_error": abs_error.mean().item(),
     "max_abs_error": abs_error.max().item(), "mean_rel_error_pct": rel_error.mean().item()},
]

if TORCHAO:
    linear_fp = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        linear_fp.weight.copy_(W); linear_fp.bias.copy_(b)
    with torch.no_grad():
        y_ref = linear_fp(x)

    # Weight-only
    lw = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        lw.weight.copy_(W); lw.bias.copy_(b)
    quantize_(lw, Int8WeightOnlyConfig())
    with torch.no_grad():
        yw = lw(x)
    ew = torch.abs(y_ref - yw)
    error_rows.append({"method": "torchao Weight-Only", "mean_abs_error": ew.mean().item(),
                        "max_abs_error": ew.max().item(), "mean_rel_error_pct": (ew / (y_ref.abs() + 1e-8) * 100).mean().item()})

    # Dynamic
    ld = nn.Linear(in_f, out_f, bias=True, device=device)
    with torch.no_grad():
        ld.weight.copy_(W); ld.bias.copy_(b)
    quantize_(ld, Int8DynamicActivationInt8WeightConfig())
    with torch.no_grad():
        yd = ld(x)
    ed = torch.abs(y_ref - yd)
    error_rows.append({"method": "torchao Dynamic", "mean_abs_error": ed.mean().item(),
                        "max_abs_error": ed.max().item(), "mean_rel_error_pct": (ed / (y_ref.abs() + 1e-8) * 100).mean().item()})

results_dir = os.environ.get("RESULTS_DIR", "results")
with open(os.path.join(results_dir, "pytorch_error.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "mean_abs_error", "max_abs_error", "mean_rel_error_pct"])
    w.writeheader()
    w.writerows(error_rows)

# ── Timing benchmarks ──
configs = [
    {"name": "4Kx4K (b=1)",   "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "4Kx11K (b=1)",  "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "8Kx8K (b=1)",   "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "4Kx11K (b=32)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]

timing_rows = []
for cfg in configs:
    nm, bsz, inf, outf = cfg["name"], cfg["batch"], cfg["in_f"], cfg["out_f"]
    wmb = inf * outf * 4 / 1024 / 1024
    xb = torch.randn(bsz, inf, dtype=torch.float32, device=device)
    Wb = torch.randn(outf, inf, dtype=torch.float32, device=device)
    bb = torch.randn(outf, dtype=torch.float32, device=device)

    # FP32
    lin = nn.Linear(inf, outf, bias=True, device=device)
    with torch.no_grad():
        lin.weight.copy_(Wb); lin.bias.copy_(bb)
    lin.eval()
    fp32_ms, fp32_std = benchmark(lambda m=lin, xi=xb: m(xi))

    row = {"config": nm, "batch": bsz, "in_features": inf, "out_features": outf,
           "weight_mb": f"{wmb:.0f}", "runtime": "PyTorch FP32 GPU",
           "time_ms": f"{fp32_ms:.4f}", "std_ms": f"{fp32_std:.4f}", "gpu": gpu_name}
    timing_rows.append(row)

    if TORCHAO:
        lw8 = nn.Linear(inf, outf, bias=True, device=device)
        with torch.no_grad():
            lw8.weight.copy_(Wb); lw8.bias.copy_(bb)
        lw8.eval()
        quantize_(lw8, Int8WeightOnlyConfig())
        w8_ms, w8_std = benchmark(lambda m=lw8, xi=xb: m(xi))
        timing_rows.append({**row, "runtime": "PyTorch INT8 torchao", "time_ms": f"{w8_ms:.4f}", "std_ms": f"{w8_std:.4f}"})
        del lw8

    del xb, Wb, bb, lin
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"  {nm} done")

with open(os.path.join(results_dir, "pytorch_timing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "batch", "in_features", "out_features",
                                       "weight_mb", "runtime", "time_ms", "std_ms", "gpu"])
    w.writeheader()
    w.writerows(timing_rows)

print("PyTorch CSVs written.")
'''
    env = os.environ.copy()
    env["RESULTS_DIR"] = RESULTS_DIR
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:1000]}")
    return result.returncode == 0


# ──────────────────────────────────────────────
# 2. ONNX Runtime benchmark
# ──────────────────────────────────────────────
def run_onnx():
    print("\n[2/4] Running ONNX Runtime benchmarks...")
    code = '''
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import time
import csv
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
USE_CUDA_EP = "CUDAExecutionProvider" in ort.get_available_providers()

results_dir = os.environ.get("RESULTS_DIR", "results")
tmpdir = tempfile.mkdtemp(prefix="onnx_bench_")

def export_onnx(linear, bs, inf, path):
    dummy = torch.randn(bs, inf, dtype=torch.float32, device="cpu")
    torch.onnx.export(linear.cpu().eval(), dummy, path, input_names=["input"],
                       output_names=["output"], opset_version=17,
                       dynamic_axes={"input":{0:"b"}, "output":{0:"b"}})
    onnx.checker.check_model(onnx.load(path))

def benchmark_ort(sess, inp, warmup=50, iters=200):
    for _ in range(warmup):
        sess.run(None, inp)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, inp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    t = np.array(times)
    return t.mean(), t.std()

from onnxruntime.quantization import quantize_dynamic, QuantType

configs = [
    {"name": "4Kx4K (b=1)",   "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "4Kx11K (b=1)",  "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "8Kx8K (b=1)",   "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "4Kx11K (b=32)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]

timing_rows = []
error_rows = []

for cfg in configs:
    nm, bsz, inf, outf = cfg["name"], cfg["batch"], cfg["in_f"], cfg["out_f"]
    wmb = inf * outf * 4 / 1024 / 1024

    W = torch.randn(outf, inf); b = torch.randn(outf)
    x_np = np.random.randn(bsz, inf).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)

    lin = nn.Linear(inf, outf, bias=True)
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    lin.eval()

    # PyTorch FP32 reference
    lin_gpu = lin.to(device)
    with torch.no_grad():
        y_ref = lin_gpu(x_torch).cpu().numpy()

    base = {"config": nm, "batch": bsz, "in_features": inf, "out_features": outf,
            "weight_mb": f"{wmb:.0f}", "gpu": gpu_name}

    # Export ONNX
    fp32_path = os.path.join(tmpdir, f"fp32_{nm.replace(' ','_')}.onnx")
    export_onnx(lin, bsz, inf, fp32_path)

    fp32_size_kb = os.path.getsize(fp32_path) / 1024

    # ONNX FP32 CPU
    sess_cpu = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    ms, std = benchmark_ort(sess_cpu, {"input": x_np})
    y_cpu = sess_cpu.run(None, {"input": x_np})[0]
    err_cpu = np.abs(y_ref - y_cpu).mean()
    timing_rows.append({**base, "runtime": "ONNX FP32 CPU", "time_ms": f"{ms:.4f}", "std_ms": f"{std:.4f}"})
    error_rows.append({**base, "runtime": "ONNX FP32 CPU", "mean_abs_error": f"{err_cpu:.8f}"})

    # ONNX FP32 GPU
    if USE_CUDA_EP:
        sess_gpu = ort.InferenceSession(fp32_path, providers=[("CUDAExecutionProvider",{"device_id":0}), "CPUExecutionProvider"])
        ms, std = benchmark_ort(sess_gpu, {"input": x_np})
        y_gpu = sess_gpu.run(None, {"input": x_np})[0]
        err_gpu = np.abs(y_ref - y_gpu).mean()
        timing_rows.append({**base, "runtime": "ONNX FP32 GPU", "time_ms": f"{ms:.4f}", "std_ms": f"{std:.4f}"})
        error_rows.append({**base, "runtime": "ONNX FP32 GPU", "mean_abs_error": f"{err_gpu:.8f}"})

    # ONNX INT8 CPU
    int8_path = os.path.join(tmpdir, f"int8_{nm.replace(' ','_')}.onnx")
    quantize_dynamic(model_input=fp32_path, model_output=int8_path, weight_type=QuantType.QInt8)
    int8_size_kb = os.path.getsize(int8_path) / 1024

    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    ms, std = benchmark_ort(sess_int8, {"input": x_np})
    y_int8 = sess_int8.run(None, {"input": x_np})[0]
    err_int8 = np.abs(y_ref - y_int8).mean()
    timing_rows.append({**base, "runtime": "ONNX INT8 CPU", "time_ms": f"{ms:.4f}", "std_ms": f"{std:.4f}"})
    error_rows.append({**base, "runtime": "ONNX INT8 CPU", "mean_abs_error": f"{err_int8:.8f}",
                       "fp32_size_kb": f"{fp32_size_kb:.1f}", "int8_size_kb": f"{int8_size_kb:.1f}"})

    del W, b, x_np, x_torch, lin, lin_gpu
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"  {nm} done")

with open(os.path.join(results_dir, "onnx_timing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config","batch","in_features","out_features",
                                       "weight_mb","runtime","time_ms","std_ms","gpu"])
    w.writeheader()
    w.writerows(timing_rows)

err_fields = ["config","batch","in_features","out_features","weight_mb","runtime",
              "mean_abs_error","fp32_size_kb","int8_size_kb","gpu"]
with open(os.path.join(results_dir, "onnx_error.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=err_fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(error_rows)

import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
print("ONNX CSVs written.")
'''
    env = os.environ.copy()
    env["RESULTS_DIR"] = RESULTS_DIR
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:1000]}")
    return result.returncode == 0


# ──────────────────────────────────────────────
# 3. CUDA C benchmark
# ──────────────────────────────────────────────
def run_cuda():
    print("\n[3/4] Running CUDA C benchmarks...")
    binary = os.path.join(SCRIPT_DIR, "fp32_to_int8_cuda")
    if not os.path.isfile(binary):
        print("  Binary not found, building...")
        subprocess.run(["make"], cwd=SCRIPT_DIR, capture_output=True)

    # Run with LD_LIBRARY_PATH
    home = os.path.expanduser("~")
    cuda_lib = f"{home}/miniconda3/envs/cuda_build/targets/x86_64-linux/lib"
    cuda_lib2 = f"{home}/miniconda3/envs/cuda_build/lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{cuda_lib2}:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run([binary], capture_output=True, text=True, timeout=600, env=env, cwd=SCRIPT_DIR)
    stdout = result.stdout

    # Parse the benchmark table from CUDA C output
    # Format: "GEMV  4Kx4K   (b=1)           GEMV    64MB    0.270 ms     0.269 ms     0.082 ms     INT8"
    timing_rows = []
    gpu_name = ""

    for line in stdout.split("\n"):
        if line.startswith("GPU:"):
            gpu_name = line.split(":", 1)[1].strip()

        # Match benchmark result lines
        m = re.match(
            r'(\S+\s+\S+\s+\(b=\d+\))\s+(GEMV|GEMM)\s+(\d+)MB\s+([\d.]+)\s+ms\s+([\d.]+)\s+ms\s+([\d.]+)\s+ms\s+(\w+)',
            line.strip()
        )
        if m:
            config_name = m.group(1).strip()
            op = m.group(2)
            weight_mb = m.group(3)
            fp32_ms = m.group(4)
            tf32_ms = m.group(5)
            int8_ms = m.group(6)
            batch = 1 if op == "GEMV" else 32

            # Parse in/out features from config name
            size_m = re.search(r'(\d+)Kx(\d+)K', config_name)
            if size_m:
                in_f = int(size_m.group(1)) * 1024
                out_f = int(size_m.group(2)) * 1024
            else:
                in_f = out_f = 0

            # Normalize config name to match other CSVs
            short_name = f"{size_m.group(1)}Kx{size_m.group(2)}K (b={batch})" if size_m else config_name

            base = {"config": short_name, "batch": batch, "in_features": in_f,
                    "out_features": out_f, "weight_mb": weight_mb, "gpu": gpu_name}

            timing_rows.append({**base, "runtime": "CUDA C FP32 (TF32=OFF)", "time_ms": fp32_ms, "std_ms": ""})
            timing_rows.append({**base, "runtime": "CUDA C FP32 (TF32=ON)", "time_ms": tf32_ms, "std_ms": ""})
            timing_rows.append({**base, "runtime": "CUDA C INT8 IGEMM", "time_ms": int8_ms, "std_ms": ""})

    # Also parse error from step 4
    error_m = re.search(r'Mean absolute error:\s+([\d.]+)', stdout)
    rel_m = re.search(r'Mean relative error:\s+([\d.]+)%', stdout)

    csv_path = os.path.join(RESULTS_DIR, "cuda_timing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config","batch","in_features","out_features",
                                           "weight_mb","runtime","time_ms","std_ms","gpu"])
        w.writeheader()
        w.writerows(timing_rows)

    # Error CSV
    if error_m:
        err_path = os.path.join(RESULTS_DIR, "cuda_error.csv")
        with open(err_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "mean_abs_error", "mean_rel_error_pct"])
            w.writeheader()
            w.writerow({"method": "CUDA C INT8",
                        "mean_abs_error": error_m.group(1),
                        "mean_rel_error_pct": rel_m.group(1) if rel_m else ""})

    print(f"  Parsed {len(timing_rows)} timing entries")
    print("  CUDA C CSVs written.")
    return len(timing_rows) > 0


# ──────────────────────────────────────────────
# 4. Profiling (core type breakdown)
# ──────────────────────────────────────────────
def run_profiling():
    print("\n[4/4] Running GPU core profiling...")
    code = '''
import torch
import torch.nn as nn
import csv
import os
import warnings

warnings.filterwarnings("ignore")
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(42)
device = torch.device("cuda")

try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    TORCHAO = True
except ImportError:
    TORCHAO = False

def classify_kernel(name):
    n = name.lower()
    if any(kw in n for kw in ['memcpy','memset','fill','copy_','elementwise','vectorized','reduce']):
        return 'Memory/Elem'
    tc = ['imma','hmma','s8_','_s8','i8816','tensorop','s16816','16816','h884','h1688','tf32','xmma']
    if any(p in n for p in tc):
        return 'Tensor Core'
    cc = ['sgemm','dgemm','gemvx','gemv2','gemv_','cublaslt']
    if any(p in n for p in cc):
        return 'CUDA Core'
    if 'cutlass' in n:
        return 'Tensor Core'
    return 'Other'

def profile_session(label, linear, x_input, warmup=10, iters=20):
    for _ in range(warmup):
        with torch.no_grad():
            _ = linear(x_input)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(iters):
            with torch.no_grad():
                _ = linear(x_input)
        torch.cuda.synchronize()

    kernels = {}
    for event in prof.events():
        if event.device_time_total > 0:
            name = event.name
            if name.startswith("aten::") or name.startswith("torch::"):
                continue
            ct = classify_kernel(name)
            kernels[ct] = kernels.get(ct, 0) + event.device_time_total

    total = sum(kernels.values()) or 1
    return {ct: round(v / total * 100, 1) for ct, v in kernels.items()}

IN, OUT = 4096, 11008
W = torch.randn(OUT, IN, dtype=torch.float32, device=device)
b = torch.randn(OUT, dtype=torch.float32, device=device)

results_dir = os.environ.get("RESULTS_DIR", "results")
rows = []

for batch_size in [1, 32]:
    x = torch.randn(batch_size, IN, dtype=torch.float32, device=device)
    op = "GEMV" if batch_size == 1 else "GEMM"

    # FP32 TF32=OFF
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lin = nn.Linear(IN, OUT, bias=True, device=device)
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    lin.eval()
    bd = profile_session(f"FP32 b={batch_size} TF32=OFF", lin, x)
    rows.append({"session": f"FP32 TF32=OFF (b={batch_size})", "operation": op, "batch": batch_size, **bd})

    # FP32 TF32=ON
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bd = profile_session(f"FP32 b={batch_size} TF32=ON", lin, x)
    rows.append({"session": f"FP32 TF32=ON (b={batch_size})", "operation": op, "batch": batch_size, **bd})

    # INT8 torchao
    if TORCHAO:
        lin8 = nn.Linear(IN, OUT, bias=True, device=device)
        with torch.no_grad():
            lin8.weight.copy_(W); lin8.bias.copy_(b)
        lin8.eval()
        quantize_(lin8, Int8WeightOnlyConfig())
        bd = profile_session(f"INT8 torchao b={batch_size}", lin8, x)
        rows.append({"session": f"INT8 torchao (b={batch_size})", "operation": op, "batch": batch_size, **bd})
        del lin8

    del x
    torch.cuda.empty_cache()
    print(f"  batch={batch_size} done")

# Write CSV with all core types as columns
all_types = set()
for r in rows:
    all_types.update(k for k in r if k not in ("session","operation","batch"))
all_types = sorted(all_types)

with open(os.path.join(results_dir, "profiling.csv"), "w", newline="") as f:
    fields = ["session", "operation", "batch"] + all_types
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        # Fill missing core types with 0
        for ct in all_types:
            r.setdefault(ct, 0.0)
        w.writerow(r)

print("Profiling CSV written.")
'''
    env = os.environ.copy()
    env["RESULTS_DIR"] = RESULTS_DIR
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:1000]}")
    return result.returncode == 0


# ──────────────────────────────────────────────
# 5. Ollama benchmark (real model inference)
# ──────────────────────────────────────────────
def run_ollama():
    print("\n[5/5] Running Ollama benchmarks...")
    try:
        import ollama
    except ImportError:
        print("  ERROR: ollama package not installed. Run: pip install ollama")
        return False

    # Check if Ollama server is running
    try:
        model_list = ollama.list()
    except Exception as e:
        print(f"  ERROR: Ollama server not reachable: {e}")
        print("  Start it with: systemctl start ollama")
        return False

    if not model_list.models:
        print("  ERROR: No models available. Pull one with: ollama pull llama3.2:3b")
        return False

    timing_rows = []
    import statistics

    for model_entry in model_list.models:
        model_name = model_entry.model
        details = model_entry.details
        quant = details.quantization_level if details else "unknown"
        param_size = details.parameter_size if details else "unknown"
        family = details.family if details else "unknown"
        size_mb = model_entry.size / (1024 * 1024)

        print(f"  Benchmarking: {model_name} ({quant}, {param_size})")

        # Run inference benchmark: measure time for a short prompt
        prompt = "What is 2+2? Answer in one word."
        warmup_iters = 3
        bench_iters = 10

        # Warmup
        warmup_ok = True
        for _ in range(warmup_iters):
            try:
                ollama.generate(model=model_name, prompt=prompt,
                                options={"num_predict": 20})
            except Exception as e:
                print(f"    SKIP: Inference failed: {e}")
                warmup_ok = False
                break

        if not warmup_ok:
            continue

        # Benchmark
        times = []
        total_tokens = 0
        for _ in range(bench_iters):
            t0 = time.time()
            resp = ollama.generate(model=model_name, prompt=prompt,
                                   options={"num_predict": 20})
            t1 = time.time()
            times.append((t1 - t0) * 1000)  # ms
            total_tokens += getattr(resp, "eval_count", 0) or 0

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        avg_tokens = total_tokens / bench_iters if bench_iters > 0 else 0

        # Check GPU allocation after inference
        gpu_pct = "unknown"
        vram_mb = 0
        try:
            ps_info = ollama.ps()
            for pm in ps_info.models:
                if pm.model == model_name:
                    total_size = pm.size or 1
                    vram_size = pm.size_vram or 0
                    gpu_pct = f"{vram_size / total_size * 100:.0f}%"
                    vram_mb = vram_size / (1024 * 1024)
                    break
        except Exception:
            pass

        timing_rows.append({
            "model": model_name,
            "family": family,
            "parameters": param_size,
            "quantization": quant,
            "model_size_mb": f"{size_mb:.0f}",
            "inference_ms": f"{mean_ms:.1f}",
            "std_ms": f"{std_ms:.1f}",
            "avg_tokens": f"{avg_tokens:.1f}",
            "gpu_percent": gpu_pct,
            "vram_mb": f"{vram_mb:.0f}",
        })
        print(f"    {model_name}: {mean_ms:.1f}ms ± {std_ms:.1f}ms, "
              f"{avg_tokens:.0f} tokens, GPU: {gpu_pct}")

    if not timing_rows:
        print("  No Ollama results collected")
        return False

    csv_path = os.path.join(RESULTS_DIR, "ollama_timing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "family", "parameters", "quantization", "model_size_mb",
            "inference_ms", "std_ms", "avg_tokens", "gpu_percent", "vram_mb"
        ])
        w.writeheader()
        w.writerows(timing_rows)

    print("  Ollama CSV written.")
    return True


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["pytorch", "onnx", "cuda", "profile", "ollama"]

    t0 = time.time()
    results = {}

    if "pytorch" in targets:
        results["pytorch"] = run_pytorch()
    if "onnx" in targets:
        results["onnx"] = run_onnx()
    if "cuda" in targets:
        results["cuda"] = run_cuda()
    if "profile" in targets:
        results["profile"] = run_profiling()
    if "ollama" in targets:
        results["ollama"] = run_ollama()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All benchmarks complete in {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")

    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:10s}: {status}")

    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    print(f"\nCSV files generated:")
    for f in sorted(csv_files):
        path = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f} ({size} bytes)")
