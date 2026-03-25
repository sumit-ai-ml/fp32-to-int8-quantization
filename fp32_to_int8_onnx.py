"""
FP32 vs INT8 Quantization: ONNX Runtime vs PyTorch Comparison
=============================================================
Step 1: Create FP32 linear model (same weights as PyTorch version)
Step 2: Export to ONNX format
Step 3: Run FP32 inference via ONNX Runtime (CPU & GPU)
Step 4: Quantize ONNX model to INT8 (dynamic quantization)
Step 5: Run INT8 inference via ONNX Runtime
Step 6: Compare error and inference time: PyTorch vs ONNX (FP32 & INT8)
Step 7: LLM-scale benchmark across multiple sizes
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import time
import os
import tempfile
import warnings

warnings.filterwarnings("ignore", message=".*cpp extensions.*")
warnings.filterwarnings("ignore", message=".*Config Deprecation.*")

torch.manual_seed(42)

# Device auto-detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ONNX Runtime providers
available_providers = ort.get_available_providers()
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {available_providers}")

USE_CUDA_EP = "CUDAExecutionProvider" in available_providers

# ──────────────────────────────────────────────
# Helper: export nn.Linear to ONNX
# ──────────────────────────────────────────────
def export_linear_to_onnx(linear_layer, batch_size, in_features, onnx_path):
    """Export a PyTorch nn.Linear to ONNX format."""
    dummy_input = torch.randn(batch_size, in_features, dtype=torch.float32, device="cpu")
    linear_cpu = linear_layer.cpu().eval()

    torch.onnx.export(
        linear_cpu,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    # Validate
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return onnx_path


def create_ort_session(onnx_path, use_cuda=False):
    """Create an ONNX Runtime inference session."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if use_cuda and USE_CUDA_EP:
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)


# ──────────────────────────────────────────────
# Symmetric quantization (same as PyTorch version)
# ──────────────────────────────────────────────
def quantize_symmetric(tensor_fp32, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = tensor_fp32.abs().max()
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)
    tensor_int8 = torch.round(tensor_fp32 / scale).to(torch.int8)
    return tensor_int8, scale


# ══════════════════════════════════════════════
# STEP 1-4: Small demo — accuracy comparison
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: FP32 Linear Model (Small Demo)")
print("=" * 60)

in_features = 4
out_features = 3
batch_size = 2

W_fp32 = torch.randn(out_features, in_features, dtype=torch.float32)
b_fp32 = torch.randn(out_features, dtype=torch.float32)
x_fp32 = torch.randn(batch_size, in_features, dtype=torch.float32)

# Create nn.Linear with fixed weights
linear_demo = nn.Linear(in_features, out_features, bias=True)
with torch.no_grad():
    linear_demo.weight.copy_(W_fp32)
    linear_demo.bias.copy_(b_fp32)
linear_demo.eval()

# PyTorch FP32 output
with torch.no_grad():
    y_pytorch_fp32 = linear_demo(x_fp32)

print(f"Input x: {x_fp32.shape}")
print(f"Weight W: {W_fp32.shape}")
print(f"PyTorch FP32 output:\n{y_pytorch_fp32}")


# ──────────────────────────────────────────────
# STEP 2: Export to ONNX
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Export to ONNX Format")
print("=" * 60)

tmpdir = tempfile.mkdtemp(prefix="onnx_quant_")
onnx_fp32_path = os.path.join(tmpdir, "linear_fp32.onnx")

export_linear_to_onnx(linear_demo, batch_size, in_features, onnx_fp32_path)
model_size = os.path.getsize(onnx_fp32_path)
print(f"ONNX FP32 model saved: {onnx_fp32_path}")
print(f"Model size: {model_size} bytes")

# Print ONNX graph info
onnx_model = onnx.load(onnx_fp32_path)
print(f"ONNX opset version: {onnx_model.opset_import[0].version}")
print(f"Graph nodes: {len(onnx_model.graph.node)}")
for node in onnx_model.graph.node:
    print(f"  {node.op_type}: {list(node.input)} -> {list(node.output)}")


# ──────────────────────────────────────────────
# STEP 3: ONNX Runtime FP32 Inference
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: ONNX Runtime FP32 Inference")
print("=" * 60)

x_np = x_fp32.numpy()

# CPU session
sess_cpu = create_ort_session(onnx_fp32_path, use_cuda=False)
y_onnx_fp32_cpu = sess_cpu.run(None, {"input": x_np})[0]
print(f"\nONNX FP32 (CPU) output:\n{y_onnx_fp32_cpu}")

# GPU session
if USE_CUDA_EP:
    sess_gpu = create_ort_session(onnx_fp32_path, use_cuda=True)
    y_onnx_fp32_gpu = sess_gpu.run(None, {"input": x_np})[0]
    print(f"\nONNX FP32 (GPU) output:\n{y_onnx_fp32_gpu}")


# ──────────────────────────────────────────────
# STEP 4: ONNX INT8 Dynamic Quantization
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: ONNX INT8 Dynamic Quantization")
print("=" * 60)

from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_int8_path = os.path.join(tmpdir, "linear_int8.onnx")
quantize_dynamic(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8,
)

int8_size = os.path.getsize(onnx_int8_path)
compression = model_size / int8_size if int8_size > 0 else 0
print(f"ONNX INT8 model saved: {onnx_int8_path}")
print(f"FP32 model size: {model_size} bytes")
print(f"INT8 model size: {int8_size} bytes")
print(f"Size ratio:      {compression:.2f}x")

# INT8 inference
sess_int8_cpu = create_ort_session(onnx_int8_path, use_cuda=False)
y_onnx_int8_cpu = sess_int8_cpu.run(None, {"input": x_np})[0]
print(f"\nONNX INT8 (CPU) output:\n{y_onnx_int8_cpu}")


# ──────────────────────────────────────────────
# STEP 5: Error Analysis — All Methods
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Error Analysis — PyTorch vs ONNX (FP32 & INT8)")
print("=" * 60)

y_ref = y_pytorch_fp32.numpy()

# Manual INT8 (same as PyTorch script)
W_int8, scale_W = quantize_symmetric(W_fp32)
x_int8, scale_x = quantize_symmetric(x_fp32)
y_int32 = x_int8.to(torch.int32) @ W_int8.to(torch.int32).T
scale_output = scale_x * scale_W
y_manual_int8 = (y_int32.to(torch.float32) * scale_output + b_fp32).numpy()

methods = {
    "ONNX FP32 (CPU)": y_onnx_fp32_cpu,
    "ONNX INT8 (CPU)": y_onnx_int8_cpu,
    "Manual INT8 (PyTorch)": y_manual_int8,
}

if USE_CUDA_EP:
    methods["ONNX FP32 (GPU)"] = y_onnx_fp32_gpu

print(f"\nReference: PyTorch FP32 nn.Linear output")
print(f"{'Method':<25} {'Mean Abs Err':>12} {'Max Abs Err':>12} {'Mean Rel Err %':>15}")
print("-" * 66)

for name, y_pred in methods.items():
    abs_err = np.abs(y_ref - y_pred)
    rel_err = abs_err / (np.abs(y_ref) + 1e-8) * 100
    print(f"{name:<25} {abs_err.mean():>12.6f} {abs_err.max():>12.6f} {rel_err.mean():>14.2f}%")


# ══════════════════════════════════════════════
# STEP 6-7: LLM-Scale Benchmark
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: LLM-Scale Timing Benchmark — PyTorch vs ONNX")
print("=" * 60)

if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")

warmup_iters = 50
bench_iters = 200


def benchmark_pytorch(fn, warmup=warmup_iters, iters=bench_iters):
    """Benchmark PyTorch with GPU sync."""
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

    t = np.array(times)
    return t.mean(), t.std()


def benchmark_ort(session, input_dict, warmup=warmup_iters, iters=bench_iters):
    """Benchmark ONNX Runtime inference."""
    for _ in range(warmup):
        session.run(None, input_dict)

    # IO binding for GPU sessions to avoid CPU<->GPU copies
    use_io_binding = False
    if hasattr(session, '_providers') or True:
        try:
            providers = session.get_providers()
            if "CUDAExecutionProvider" in providers:
                use_io_binding = True
        except Exception:
            pass

    times = []
    if use_io_binding and "CUDAExecutionProvider" in session.get_providers():
        # Use IO binding for GPU to get accurate timing
        io_binding = session.io_binding()
        x_ort = ort.OrtValue.ortvalue_from_numpy(input_dict["input"], "cuda", 0)
        io_binding.bind_ortvalue_input("input", x_ort)
        io_binding.bind_output("output", "cuda")

        for _ in range(warmup):
            session.run_with_iobinding(io_binding)

        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            session.run_with_iobinding(io_binding)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            session.run(None, input_dict)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    t = np.array(times)
    return t.mean(), t.std()


bench_configs = [
    {"name": "Small (4K×4K)",       "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "Medium (4K×11K)",     "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "Large (8K×8K)",       "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "Batched (32×4K×11K)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]

print(f"\nWarmup: {warmup_iters} iters, Benchmark: {bench_iters} iters\n")

all_results = []

for cfg in bench_configs:
    name = cfg["name"]
    bs, in_f, out_f = cfg["batch"], cfg["in_f"], cfg["out_f"]
    weight_mb_fp32 = in_f * out_f * 4 / 1024 / 1024
    weight_mb_int8 = in_f * out_f * 1 / 1024 / 1024

    print(f"{'=' * 80}")
    print(f"  {name}")
    print(f"  Weight: {in_f}x{out_f} = {weight_mb_fp32:.0f} MB FP32 / {weight_mb_int8:.0f} MB INT8")
    print(f"  Input:  {bs}x{in_f}")
    print(f"{'=' * 80}")

    # Create PyTorch model with shared weights
    W_b = torch.randn(out_f, in_f, dtype=torch.float32)
    b_b = torch.randn(out_f, dtype=torch.float32)
    x_b_np = np.random.randn(bs, in_f).astype(np.float32)
    x_b_torch = torch.from_numpy(x_b_np).to(device)

    linear_bench = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        linear_bench.weight.copy_(W_b)
        linear_bench.bias.copy_(b_b)
    linear_bench.eval()

    # ── PyTorch FP32 (GPU) ──
    linear_gpu = linear_bench.to(device)

    def pt_fp32_fn(m=linear_gpu, x=x_b_torch):
        with torch.no_grad():
            return m(x)

    pt_fp32_ms, pt_fp32_std = benchmark_pytorch(pt_fp32_fn)
    print(f"  PyTorch FP32 (GPU):     {pt_fp32_ms:.3f} ± {pt_fp32_std:.3f} ms")

    # Get PyTorch FP32 reference output for error comparison
    with torch.no_grad():
        y_ref_bench = linear_gpu(x_b_torch).cpu().numpy()

    # ── Export to ONNX ──
    onnx_bench_fp32 = os.path.join(tmpdir, f"bench_fp32_{name.replace(' ', '_')}.onnx")
    export_linear_to_onnx(linear_bench, bs, in_f, onnx_bench_fp32)

    # ── ONNX FP32 (CPU) ──
    sess_bench_cpu = create_ort_session(onnx_bench_fp32, use_cuda=False)
    ort_fp32_cpu_ms, ort_fp32_cpu_std = benchmark_ort(
        sess_bench_cpu, {"input": x_b_np}
    )
    y_ort_cpu = sess_bench_cpu.run(None, {"input": x_b_np})[0]
    err_ort_cpu = np.abs(y_ref_bench - y_ort_cpu).mean()
    print(f"  ONNX FP32 (CPU):        {ort_fp32_cpu_ms:.3f} ± {ort_fp32_cpu_std:.3f} ms  (err={err_ort_cpu:.6f})")

    # ── ONNX FP32 (GPU) ──
    ort_fp32_gpu_ms = None
    if USE_CUDA_EP:
        sess_bench_gpu = create_ort_session(onnx_bench_fp32, use_cuda=True)
        ort_fp32_gpu_ms, ort_fp32_gpu_std = benchmark_ort(
            sess_bench_gpu, {"input": x_b_np}
        )
        y_ort_gpu = sess_bench_gpu.run(None, {"input": x_b_np})[0]
        err_ort_gpu = np.abs(y_ref_bench - y_ort_gpu).mean()
        print(f"  ONNX FP32 (GPU):        {ort_fp32_gpu_ms:.3f} ± {ort_fp32_gpu_std:.3f} ms  (err={err_ort_gpu:.6f})")

    # ── ONNX INT8 Dynamic (CPU) ──
    onnx_bench_int8 = os.path.join(tmpdir, f"bench_int8_{name.replace(' ', '_')}.onnx")
    quantize_dynamic(
        model_input=onnx_bench_fp32,
        model_output=onnx_bench_int8,
        weight_type=QuantType.QInt8,
    )

    sess_bench_int8 = create_ort_session(onnx_bench_int8, use_cuda=False)
    ort_int8_cpu_ms, ort_int8_cpu_std = benchmark_ort(
        sess_bench_int8, {"input": x_b_np}
    )
    y_ort_int8 = sess_bench_int8.run(None, {"input": x_b_np})[0]
    err_ort_int8 = np.abs(y_ref_bench - y_ort_int8).mean()
    print(f"  ONNX INT8 (CPU):        {ort_int8_cpu_ms:.3f} ± {ort_int8_cpu_std:.3f} ms  (err={err_ort_int8:.6f})")

    # ── PyTorch torchao INT8 (GPU) ──
    pt_int8_ms = None
    err_pt_int8 = None
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        TORCHAO_AVAILABLE = True
    except ImportError:
        TORCHAO_AVAILABLE = False

    if TORCHAO_AVAILABLE:
        linear_int8 = nn.Linear(in_f, out_f, bias=True, device=device)
        with torch.no_grad():
            linear_int8.weight.copy_(W_b.to(device))
            linear_int8.bias.copy_(b_b.to(device))
        linear_int8.eval()
        quantize_(linear_int8, Int8WeightOnlyConfig())

        def pt_int8_fn(m=linear_int8, x=x_b_torch):
            with torch.no_grad():
                return m(x)

        pt_int8_ms, pt_int8_std = benchmark_pytorch(pt_int8_fn)
        with torch.no_grad():
            y_pt_int8 = linear_int8(x_b_torch).cpu().numpy()
        err_pt_int8 = np.abs(y_ref_bench - y_pt_int8).mean()
        print(f"  PyTorch INT8 torchao:   {pt_int8_ms:.3f} ± {pt_int8_std:.3f} ms  (err={err_pt_int8:.6f})")

    # Model sizes
    fp32_sz = os.path.getsize(onnx_bench_fp32)
    int8_sz = os.path.getsize(onnx_bench_int8)
    print(f"  ONNX model size: FP32={fp32_sz/1024:.1f}KB  INT8={int8_sz/1024:.1f}KB  ({fp32_sz/int8_sz:.2f}x)")

    result = {
        "name": name, "weight_mb": weight_mb_fp32,
        "pt_fp32_gpu": pt_fp32_ms,
        "ort_fp32_cpu": ort_fp32_cpu_ms,
        "ort_fp32_gpu": ort_fp32_gpu_ms,
        "ort_int8_cpu": ort_int8_cpu_ms,
        "pt_int8_gpu": pt_int8_ms,
        "err_ort_fp32_cpu": err_ort_cpu,
        "err_ort_int8_cpu": err_ort_int8,
        "err_pt_int8": err_pt_int8,
        "onnx_fp32_kb": fp32_sz / 1024,
        "onnx_int8_kb": int8_sz / 1024,
    }
    all_results.append(result)

    # Free memory
    del W_b, b_b, x_b_np, x_b_torch, linear_bench, linear_gpu
    if TORCHAO_AVAILABLE:
        del linear_int8
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print()


# ══════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY: Inference Time (ms)")
print("=" * 80)

def fmt(ms):
    return f"{ms:.3f}" if ms is not None else "N/A"

header = (f"{'Config':<25} {'Weight':>7} {'PT FP32':>9} {'ORT FP32':>9} "
          f"{'ORT FP32':>9} {'ORT INT8':>9} {'PT INT8':>9}")
subhdr = (f"{'':25} {'':>7} {'GPU':>9} {'CPU':>9} "
          f"{'GPU':>9} {'CPU':>9} {'GPU':>9}")
print(header)
print(subhdr)
print("-" * 80)

for r in all_results:
    print(f"{r['name']:<25} {r['weight_mb']:>5.0f}MB "
          f"{fmt(r['pt_fp32_gpu']):>9} "
          f"{fmt(r['ort_fp32_cpu']):>9} "
          f"{fmt(r['ort_fp32_gpu']):>9} "
          f"{fmt(r['ort_int8_cpu']):>9} "
          f"{fmt(r['pt_int8_gpu']):>9}")


# ── Error summary ──
print(f"\n{'=' * 80}")
print("SUMMARY: Mean Absolute Error (vs PyTorch FP32 GPU)")
print("=" * 80)

print(f"{'Config':<25} {'ONNX FP32 CPU':>14} {'ONNX INT8 CPU':>14} {'PT INT8 GPU':>14}")
print("-" * 70)

for r in all_results:
    print(f"{r['name']:<25} "
          f"{r['err_ort_fp32_cpu']:>14.6f} "
          f"{r['err_ort_int8_cpu']:>14.6f} "
          f"{fmt(r['err_pt_int8']):>14}")


# ── Speedup table ──
print(f"\n{'=' * 80}")
print("SUMMARY: Speedup vs PyTorch FP32 GPU (>1.0 = faster)")
print("=" * 80)

print(f"{'Config':<25} {'ORT FP32 CPU':>13} {'ORT FP32 GPU':>13} {'ORT INT8 CPU':>13} {'PT INT8 GPU':>13}")
print("-" * 80)

for r in all_results:
    def spd(ms):
        if ms is None:
            return "N/A"
        s = r["pt_fp32_gpu"] / ms
        marker = " <<" if s > 1.0 else ""
        return f"{s:.2f}x{marker}"

    print(f"{r['name']:<25} "
          f"{spd(r['ort_fp32_cpu']):>13} "
          f"{spd(r['ort_fp32_gpu']):>13} "
          f"{spd(r['ort_int8_cpu']):>13} "
          f"{spd(r['pt_int8_gpu']):>13}")


# ── Model size summary ──
print(f"\n{'=' * 80}")
print("SUMMARY: ONNX Model Size")
print("=" * 80)

print(f"{'Config':<25} {'FP32 (KB)':>10} {'INT8 (KB)':>10} {'Compression':>12}")
print("-" * 60)

for r in all_results:
    comp = r["onnx_fp32_kb"] / r["onnx_int8_kb"] if r["onnx_int8_kb"] > 0 else 0
    print(f"{r['name']:<25} {r['onnx_fp32_kb']:>10.1f} {r['onnx_int8_kb']:>10.1f} {comp:>11.2f}x")


# ── Key insights ──
print(f"""

KEY INSIGHTS:
{'─' * 60}
1. ONNX Runtime uses optimized graph-level transformations
   (operator fusion, constant folding) that PyTorch eager mode doesn't.

2. ONNX INT8 dynamic quantization quantizes weights offline and
   activations at runtime — similar to torchao's dynamic INT8.

3. ONNX FP32 should match PyTorch FP32 output exactly (or within
   floating-point epsilon) since it's the same computation graph.

4. ONNX INT8 error vs PyTorch INT8 error comparison shows whether
   ONNX Runtime's quantization strategy differs from torchao's.

5. For deployment:
   - ONNX is portable across hardware (CPU/GPU/NPU/edge)
   - PyTorch + torchao is better for GPU-heavy server inference
   - ONNX INT8 on CPU can rival GPU inference for small batch sizes
""")

# Cleanup temp files
import shutil
print(f"Temp ONNX models in: {tmpdir}")
print(f"To clean up: rm -rf {tmpdir}")
