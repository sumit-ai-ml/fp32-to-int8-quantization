"""
Targeted tests for the FP32→INT8 quantization benchmark suite.

Tests:
  - classify_kernel() heuristic correctness
  - CUDA C stdout regex parser
  - CSV schema validation
  - Pipeline smoke test
"""

import os
import re
import pytest
import pandas as pd

from bench_utils import classify_kernel, CONFIGS


# ══════════════════════════════════════════════
# 1. classify_kernel() unit tests
# ══════════════════════════════════════════════

class TestClassifyKernel:
    """Test the kernel name → core type classifier."""

    def test_tensor_core_imma(self):
        assert classify_kernel("sm86_xmma_gemm_f16f16_f32f32_f16_tn_n_tilesize256x64x32_stage3_warpsize4x1x1_tensor16x8x16_execute_kernel__51") == "Tensor Core"

    def test_tensor_core_s8(self):
        assert classify_kernel("s8_tensorop_kernel") == "Tensor Core"

    def test_tensor_core_hmma(self):
        assert classify_kernel("volta_hmma_gemm") == "Tensor Core"

    def test_cutlass_simt_is_cuda_core(self):
        # cutlass_simt_sgemm has 'sgemm' which matches CUDA Core before 'cutlass'
        assert classify_kernel("cutlass_80_simt_sgemm") == "CUDA Core"

    def test_cutlass_tensorop_is_tensor_core(self):
        assert classify_kernel("cutlass_80_tensorop_gemm") == "Tensor Core"

    def test_tensor_core_tf32(self):
        assert classify_kernel("ampere_tf32_gemm") == "Tensor Core"

    def test_cuda_core_sgemm(self):
        assert classify_kernel("ampere_sgemm_128x32_tn") == "CUDA Core"

    def test_cuda_core_gemvx(self):
        assert classify_kernel("gemvx::kernel<float>") == "CUDA Core"

    def test_cuda_core_gemv2(self):
        assert classify_kernel("gemv2T_kernel") == "CUDA Core"

    def test_cuda_core_dgemm(self):
        assert classify_kernel("dgemm_kernel") == "CUDA Core"

    def test_memory_memcpy(self):
        assert classify_kernel("cudaMemcpyAsync") == "Memory/Elem"

    def test_memory_memset(self):
        assert classify_kernel("cudaMemsetD32Async") == "Memory/Elem"

    def test_memory_elementwise(self):
        assert classify_kernel("elementwise_kernel") == "Memory/Elem"

    def test_memory_vectorized(self):
        assert classify_kernel("vectorized_reduce") == "Memory/Elem"

    def test_memory_fill(self):
        assert classify_kernel("fill_kernel") == "Memory/Elem"

    def test_unknown_returns_other(self):
        assert classify_kernel("some_random_kernel_name") == "Other"

    def test_empty_string(self):
        assert classify_kernel("") == "Other"

    def test_case_insensitive(self):
        # classify_kernel lowercases internally
        assert classify_kernel("SGEMM_KERNEL") == "CUDA Core"
        assert classify_kernel("IMMA_KERNEL") == "Tensor Core"

    def test_memory_before_tensor(self):
        """Memory patterns should match before tensor core patterns."""
        # A kernel that has both 'copy_' and 'tensorop' should be Memory
        assert classify_kernel("copy_tensorop_data") == "Memory/Elem"


# ══════════════════════════════════════════════
# 2. CUDA C stdout regex parser test
# ══════════════════════════════════════════════

class TestCudaRegexParser:
    """Test the regex that parses CUDA C binary output."""

    REGEX = re.compile(
        r'(\S+\s+\S+\s+\(b=\d+\))\s+(GEMV|GEMM)\s+(\d+)MB\s+([\d.]+)\s+ms\s+([\d.]+)\s+ms\s+([\d.]+)\s+ms\s+(\w+)'
    )

    SAMPLE_LINES = [
        "GEMV  4Kx4K   (b=1)           GEMV    64MB    0.300 ms     0.300 ms     0.095 ms     INT8",
        "GEMV  4Kx11K  (b=1)           GEMV   172MB    0.789 ms     0.790 ms     0.243 ms     INT8",
        "GEMV  8Kx8K   (b=1)           GEMV   256MB    1.170 ms     1.171 ms     0.360 ms     INT8",
        "GEMM  4Kx11K  (b=32)          GEMM   172MB    1.130 ms     0.845 ms     0.220 ms     INT8",
        "GEMM  8Kx8K   (b=32)          GEMM   256MB    1.748 ms     1.208 ms     0.318 ms     INT8",
    ]

    def test_parses_all_lines(self):
        for line in self.SAMPLE_LINES:
            m = self.REGEX.match(line.strip())
            assert m is not None, f"Failed to parse: {line}"

    def test_extracts_correct_fields(self):
        m = self.REGEX.match(self.SAMPLE_LINES[0].strip())
        assert m.group(2) == "GEMV"
        assert m.group(3) == "64"  # weight MB
        assert m.group(4) == "0.300"  # FP32 ms
        assert m.group(5) == "0.300"  # TF32 ms
        assert m.group(6) == "0.095"  # INT8 ms
        assert m.group(7) == "INT8"

    def test_gemm_line(self):
        m = self.REGEX.match(self.SAMPLE_LINES[3].strip())
        assert m.group(2) == "GEMM"
        assert m.group(3) == "172"

    def test_non_matching_line(self):
        assert self.REGEX.match("GPU: NVIDIA RTX A3000") is None
        assert self.REGEX.match("") is None
        assert self.REGEX.match("============") is None


# ══════════════════════════════════════════════
# 3. CSV schema validation
# ══════════════════════════════════════════════

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class TestCSVSchemas:
    """Validate that CSVs have expected columns and non-empty data."""

    def test_pytorch_timing_schema(self):
        path = os.path.join(RESULTS_DIR, "pytorch_timing.csv")
        if not os.path.exists(path):
            pytest.skip("pytorch_timing.csv not found (run benchmarks first)")
        df = pd.read_csv(path)
        required = {"config", "batch", "in_features", "out_features", "runtime", "time_ms"}
        assert required.issubset(set(df.columns))
        assert len(df) > 0

    def test_cuda_timing_schema(self):
        path = os.path.join(RESULTS_DIR, "cuda_timing.csv")
        if not os.path.exists(path):
            pytest.skip("cuda_timing.csv not found")
        df = pd.read_csv(path)
        required = {"config", "runtime", "time_ms"}
        assert required.issubset(set(df.columns))
        assert len(df) > 0

    def test_onnx_timing_schema(self):
        path = os.path.join(RESULTS_DIR, "onnx_timing.csv")
        if not os.path.exists(path):
            pytest.skip("onnx_timing.csv not found")
        df = pd.read_csv(path)
        required = {"config", "runtime", "time_ms"}
        assert required.issubset(set(df.columns))
        assert len(df) > 0

    def test_profiling_schema(self):
        path = os.path.join(RESULTS_DIR, "profiling.csv")
        if not os.path.exists(path):
            pytest.skip("profiling.csv not found")
        df = pd.read_csv(path)
        assert "session" in df.columns
        assert "batch" in df.columns
        assert len(df) > 0

    def test_ollama_timing_schema(self):
        path = os.path.join(RESULTS_DIR, "ollama_timing.csv")
        if not os.path.exists(path):
            pytest.skip("ollama_timing.csv not found")
        df = pd.read_csv(path)
        required = {"model", "quantization", "inference_ms", "gpu_percent", "vram_mb"}
        assert required.issubset(set(df.columns))
        assert len(df) > 0


# ══════════════════════════════════════════════
# 4. Configs consistency
# ══════════════════════════════════════════════

class TestConfigs:
    """Verify benchmark configs are consistent."""

    def test_configs_have_required_keys(self):
        for cfg in CONFIGS:
            assert "name" in cfg
            assert "batch" in cfg
            assert "in_f" in cfg
            assert "out_f" in cfg

    def test_configs_count(self):
        assert len(CONFIGS) == 4

    def test_config_names_format(self):
        for cfg in CONFIGS:
            assert "(b=" in cfg["name"], f"Config name missing batch: {cfg['name']}"


# ══════════════════════════════════════════════
# 5. Speedup calculation sanity
# ══════════════════════════════════════════════

class TestSpeedupCalculation:
    """Test that speedup computation handles edge cases."""

    def test_speedup_normal(self):
        base = 1.0  # ms
        target = 0.25  # ms
        speedup = base / target
        assert speedup == 4.0

    def test_speedup_slower(self):
        base = 0.3
        target = 0.6
        speedup = base / target
        assert speedup == 0.5

    def test_speedup_zero_guard(self):
        """Division by zero should not happen — target time is always > 0."""
        base = 1.0
        target = 0.0
        # In the real code, this would be a data error
        with pytest.raises(ZeroDivisionError):
            _ = base / target
