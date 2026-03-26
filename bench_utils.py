"""
Shared utilities for the FP32→INT8 quantization benchmark suite.

Provides:
  - CONFIGS: standard matrix configurations for benchmarking
  - classify_kernel(): maps CUDA kernel names to GPU core types
  - benchmark_torch(): timing helper for PyTorch functions
"""

# Standard benchmark configurations (LLM-scale matrix dimensions)
CONFIGS = [
    {"name": "4Kx4K (b=1)",   "batch": 1,  "in_f": 4096,  "out_f": 4096},
    {"name": "4Kx11K (b=1)",  "batch": 1,  "in_f": 4096,  "out_f": 11008},
    {"name": "8Kx8K (b=1)",   "batch": 1,  "in_f": 8192,  "out_f": 8192},
    {"name": "4Kx11K (b=32)", "batch": 32, "in_f": 4096,  "out_f": 11008},
]


def classify_kernel(name):
    """Classify a CUDA kernel name into its GPU core type.

    Heuristic-based: pattern matches known kernel name substrings
    to determine whether the kernel runs on Tensor Cores, CUDA Cores,
    or is a memory/elementwise operation.

    Returns one of: 'Tensor Core', 'CUDA Core', 'Memory/Elem', 'Other'
    """
    n = name.lower()

    # Memory and elementwise operations (check first — most specific)
    mem_patterns = ['memcpy', 'memset', 'fill', 'copy_', 'elementwise',
                    'vectorized', 'reduce']
    if any(kw in n for kw in mem_patterns):
        return 'Memory/Elem'

    # Tensor Core kernels (IMMA, HMMA, CUTLASS tensorop, etc.)
    tc_patterns = ['imma', 'hmma', 's8_', '_s8', 'i8816', 'tensorop',
                   's16816', '16816', 'h884', 'h1688', 'tf32', 'xmma']
    if any(p in n for p in tc_patterns):
        return 'Tensor Core'

    # CUDA Core kernels (SGEMM, GEMV, cuBLASLt non-tensor)
    cc_patterns = ['sgemm', 'dgemm', 'gemvx', 'gemv2', 'gemv_', 'cublaslt']
    if any(p in n for p in cc_patterns):
        return 'CUDA Core'

    # CUTLASS without tensorop → Tensor Core (CUTLASS defaults to TC)
    if 'cutlass' in n:
        return 'Tensor Core'

    return 'Other'
