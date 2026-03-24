/*
 * FP32 → INT8 Quantization & Matmul in CUDA C
 * =============================================
 * Pure CUDA implementation matching fp32_to_int8_pytorch.py:
 *   Step 1: FP32 matmul (y = xW^T + b)             — cuBLAS SGEMM (CUDA Cores)
 *   Step 2: Symmetric INT8 quantization             — custom CUDA kernel
 *   Step 3: INT8 matmul with INT32 accumulation     — cuBLAS IGEMM (Tensor Cores)
 *   Step 4: Dequantize + bias + error analysis      — custom CUDA kernel
 *   Step 5: GPU profiling via CUDA Events           — kernel-level timing
 *   Step 6: Benchmark at LLM scale                  — GEMV vs GEMM comparison
 *
 * Compile:
 *   make            (uses the Makefile)
 *   OR manually:
 *   nvcc -O3 -arch=sm_86 fp32_to_int8_cuda.cu -lcublas -lcudart -o fp32_to_int8_cuda
 *
 * Run:
 *   ./fp32_to_int8_cuda
 *   OR with LD_LIBRARY_PATH set (see Makefile)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ─────────────────────────────────────────────
// Error checking macros
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)status);                                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)


// ─────────────────────────────────────────────
// CUDA Kernels
// ─────────────────────────────────────────────

// Find the max absolute value in a tensor (reduction kernel)
__global__ void abs_max_kernel(const float* data, float* result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? fabsf(data[i]) : 0.0f;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)result, __float_as_int(sdata[0]));
    }
}

// Symmetric quantize: FP32 → INT8
// scale = max(|tensor|) / 127
// int8_val = round(fp32_val / scale)
__global__ void quantize_symmetric_kernel(const float* input, int8_t* output,
                                           float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = input[i] / scale;
        val = fminf(fmaxf(roundf(val), -127.0f), 127.0f);
        output[i] = (int8_t)val;
    }
}

// Dequantize INT32 accumulator → FP32 + add bias
// output[i] = int32_acc[i] * (scale_x * scale_w) + bias[col]
__global__ void dequantize_bias_kernel(const int32_t* acc, float* output,
                                        float combined_scale, const float* bias,
                                        int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = (float)acc[idx] * combined_scale + bias[col];
    }
}

// Compute absolute error between two float arrays
__global__ void abs_error_kernel(const float* a, const float* b, float* err, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        err[i] = fabsf(a[i] - b[i]);
    }
}

// Simple FP32 bias-add kernel (for when cuBLAS GEMM doesn't include bias)
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}


// ─────────────────────────────────────────────
// Helper: find abs max on GPU
// ─────────────────────────────────────────────
float gpu_abs_max(const float* d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    abs_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_result, n);
    CUDA_CHECK(cudaGetLastError());

    float h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return h_result;
}


// ─────────────────────────────────────────────
// Helper: print a small matrix from GPU
// ─────────────────────────────────────────────
void print_matrix_f32(const char* name, const float* d_ptr, int rows, int cols) {
    float* h = (float*)malloc(rows * cols * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h, d_ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n%s (float32):\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) {
            printf("%8.4f", h[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    free(h);
}

void print_matrix_i8(const char* name, const int8_t* d_ptr, int rows, int cols) {
    int8_t* h = (int8_t*)malloc(rows * cols);
    CUDA_CHECK(cudaMemcpy(h, d_ptr, rows * cols, cudaMemcpyDeviceToHost));
    printf("\n%s (int8):\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) {
            printf("%4d", (int)h[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    free(h);
}

void print_matrix_i32(const char* name, const int32_t* d_ptr, int rows, int cols) {
    int32_t* h = (int32_t*)malloc(rows * cols * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(h, d_ptr, rows * cols * sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("\n%s (int32):\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) {
            printf("%6d", h[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    free(h);
}


// ─────────────────────────────────────────────
// Helper: fill with random data (host-side, then upload)
// ─────────────────────────────────────────────
void fill_random(float* d_ptr, int n, unsigned int seed) {
    float* h = (float*)malloc(n * sizeof(float));
    srand(seed);
    for (int i = 0; i < n; i++) {
        // Box-Muller approximation for normal distribution
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        if (u1 < 1e-7f) u1 = 1e-7f;
        h[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}


// ─────────────────────────────────────────────
// Benchmark helper with CUDA events
// ─────────────────────────────────────────────
struct BenchResult {
    float mean_ms;
    float std_ms;
    float median_ms;
};

typedef void (*bench_fn_t)(void* ctx);

BenchResult benchmark_cuda(bench_fn_t fn, void* ctx, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++) fn(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* times = (float*)malloc(iters * sizeof(float));

    for (int i = 0; i < iters; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        fn(ctx);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Compute stats
    float sum = 0, sum2 = 0;
    for (int i = 0; i < iters; i++) sum += times[i];
    float mean = sum / iters;
    for (int i = 0; i < iters; i++) sum2 += (times[i] - mean) * (times[i] - mean);
    float std_dev = sqrtf(sum2 / iters);

    // Median (simple sort)
    for (int i = 0; i < iters - 1; i++)
        for (int j = i + 1; j < iters; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    float median = times[iters / 2];

    free(times);
    return {mean, std_dev, median};
}


// ─────────────────────────────────────────────
// Benchmark context structs
// ─────────────────────────────────────────────
struct FP32GemmCtx {
    cublasHandle_t handle;
    int M, N, K;
    float *d_x, *d_W, *d_y;
    float alpha, beta;
    bool allow_tf32;
};

void fp32_gemm_fn(void* ctx) {
    FP32GemmCtx* c = (FP32GemmCtx*)ctx;

    // Set TF32 mode
    cublasSetMathMode(c->handle, c->allow_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

    // cuBLAS uses column-major. For row-major C = A * B^T:
    //   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N)
    // We want y[M,N] = x[M,K] * W^T[K,N]  where W is [N,K]
    cublasSgemm(c->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                c->N, c->M, c->K,
                &c->alpha,
                c->d_W, c->K,    // W[N,K] → W^T
                c->d_x, c->K,    // x[M,K]
                &c->beta,
                c->d_y, c->N);   // y[M,N]
}

struct INT8GemmCtx {
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    int M, N, K;
    int8_t *d_x_i8, *d_W_i8;
    int32_t *d_y_i32;
    int32_t alpha_i32, beta_i32;
};

void int8_gemm_fn(void* ctx) {
    INT8GemmCtx* c = (INT8GemmCtx*)ctx;
    cublasLtMatmul(c->ltHandle, c->matmulDesc,
                   &c->alpha_i32, c->d_W_i8, c->layoutA,
                   c->d_x_i8, c->layoutB,
                   &c->beta_i32, c->d_y_i32, c->layoutC,
                   c->d_y_i32, c->layoutC,
                   NULL, NULL, 0, 0);
}


// ═══════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════
int main() {
    // ── GPU info ──
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU:             %s\n", prop.name);
    printf("Compute cap:     %d.%d\n", prop.major, prop.minor);
    printf("VRAM:            %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("SM count:        %d\n", prop.multiProcessorCount);
    printf("L2 cache:        %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));

    if (prop.major < 8) {
        printf("WARNING: Compute %d.%d < 8.0 — INT8 Tensor Cores require Ampere+\n",
               prop.major, prop.minor);
    }

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // ══════════════════════════════════════════
    // STEP 1: Small demo — FP32 Linear (y = xW^T + b)
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 1: FP32 Linear Multiplication (y = xW^T + b)\n");
    printf("============================================================\n");

    const int in_features = 4, out_features = 3, batch_size = 2;

    float *d_W, *d_b, *d_x, *d_y_fp32;
    CUDA_CHECK(cudaMalloc(&d_W, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_fp32, batch_size * out_features * sizeof(float)));

    fill_random(d_W, out_features * in_features, 42);
    fill_random(d_b, out_features, 43);
    fill_random(d_x, batch_size * in_features, 44);

    // y = x @ W^T  using cuBLAS
    float alpha = 1.0f, beta = 0.0f;
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             out_features, batch_size, in_features,
                             &alpha,
                             d_W, in_features,
                             d_x, in_features,
                             &beta,
                             d_y_fp32, out_features));

    // Add bias
    int total = batch_size * out_features;
    int threads = 256;
    add_bias_kernel<<<(total + threads - 1) / threads, threads>>>(
        d_y_fp32, d_b, batch_size, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_matrix_f32("Input x", d_x, batch_size, in_features);
    print_matrix_f32("Weight W", d_W, out_features, in_features);
    print_matrix_f32("Output y_fp32 = xW^T + b", d_y_fp32, batch_size, out_features);


    // ══════════════════════════════════════════
    // STEP 2: INT8 Quantization
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 2: INT8 Quantization (Symmetric, Per-Tensor)\n");
    printf("============================================================\n");

    // Quantize W
    float scale_W = gpu_abs_max(d_W, out_features * in_features) / 127.0f;
    if (scale_W < 1e-8f) scale_W = 1e-8f;

    int8_t* d_W_i8;
    CUDA_CHECK(cudaMalloc(&d_W_i8, out_features * in_features));
    int n_W = out_features * in_features;
    quantize_symmetric_kernel<<<(n_W + 255) / 256, 256>>>(d_W, d_W_i8, scale_W, n_W);

    // Quantize x
    float scale_x = gpu_abs_max(d_x, batch_size * in_features) / 127.0f;
    if (scale_x < 1e-8f) scale_x = 1e-8f;

    int8_t* d_x_i8;
    CUDA_CHECK(cudaMalloc(&d_x_i8, batch_size * in_features));
    int n_x = batch_size * in_features;
    quantize_symmetric_kernel<<<(n_x + 255) / 256, 256>>>(d_x, d_x_i8, scale_x, n_x);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nScale (W): %.6f\n", scale_W);
    printf("Scale (x): %.6f\n", scale_x);
    print_matrix_i8("W_int8", d_W_i8, out_features, in_features);
    print_matrix_i8("x_int8", d_x_i8, batch_size, in_features);


    // ══════════════════════════════════════════
    // STEP 3: INT8 Matmul → INT32 accumulator
    //         Uses cuBLASLt IGEMM (Tensor Cores on sm80+)
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 3: INT8 Matmul (cuBLASLt IGEMM → INT32 accumulator)\n");
    printf("============================================================\n");

    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    int M = batch_size, N = out_features, K = in_features;

    // Matmul descriptor: INT8 inputs, INT32 output
    cublasLtMatmulDesc_t matmulDesc;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I));

    // Set transpose for W (op_A) — we want C = W^T * x^T in col-major
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                 &opB, sizeof(opB)));

    // Matrix layouts (column-major for cuBLASLt)
    // A = W[N,K] with op=T → logical [K,N] → input [N,K], ld=K
    // B = x[M,K] with op=N → but col-major: [K,M], ld=K
    // C = y[M,N] → col-major: [N,M], ld=N
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, M, K));

    int32_t* d_y_i32;
    CUDA_CHECK(cudaMalloc(&d_y_i32, M * N * sizeof(int32_t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, N, M, N));

    int32_t alpha_i = 1, beta_i = 0;
    CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                &alpha_i, d_W_i8, layoutA,
                                d_x_i8, layoutB,
                                &beta_i, d_y_i32, layoutC,
                                d_y_i32, layoutC,
                                NULL, NULL, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    print_matrix_i32("INT32 accumulator (x_int8 @ W_int8^T)", d_y_i32, M, N);


    // ══════════════════════════════════════════
    // STEP 4: Dequantize + Bias + Error Analysis
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 4: Dequantize + Bias + Error Analysis\n");
    printf("============================================================\n");

    float combined_scale = scale_x * scale_W;
    printf("\nCombined scale (scale_x * scale_W): %.8f\n", combined_scale);

    float* d_y_int8_deq;
    CUDA_CHECK(cudaMalloc(&d_y_int8_deq, M * N * sizeof(float)));

    dequantize_bias_kernel<<<(M * N + 255) / 256, 256>>>(
        d_y_i32, d_y_int8_deq, combined_scale, d_b, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_matrix_f32("FP32 output", d_y_fp32, M, N);
    print_matrix_f32("INT8 dequantized output", d_y_int8_deq, M, N);

    // Error analysis
    float* d_err;
    CUDA_CHECK(cudaMalloc(&d_err, M * N * sizeof(float)));
    abs_error_kernel<<<(M * N + 255) / 256, 256>>>(d_y_fp32, d_y_int8_deq, d_err, M * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_err = (float*)malloc(M * N * sizeof(float));
    float* h_fp32 = (float*)malloc(M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_err, d_err, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fp32, d_y_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0, sum_err = 0, sum_rel = 0;
    for (int i = 0; i < M * N; i++) {
        if (h_err[i] > max_err) max_err = h_err[i];
        sum_err += h_err[i];
        sum_rel += h_err[i] / (fabsf(h_fp32[i]) + 1e-8f) * 100.0f;
    }
    printf("\nMax absolute error:  %.6f\n", max_err);
    printf("Mean absolute error: %.6f\n", sum_err / (M * N));
    printf("Mean relative error: %.2f%%\n", sum_rel / (M * N));

    free(h_err);
    free(h_fp32);


    // ══════════════════════════════════════════
    // STEP 5: Kernel-level profiling with CUDA Events
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 5: Kernel-Level Profiling (CUDA Events)\n");
    printf("============================================================\n");
    printf("\nProfiling individual kernels on %dx%d weight matrix:\n", out_features, in_features);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    float elapsed;

    // Profile: FP32 GEMM (CUDA Cores)
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, d_W, K, d_x, K, &beta, d_y_fp32, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    printf("\n  FP32 SGEMM (CUDA Cores, TF32 OFF): %.3f ms\n", elapsed);
    printf("    → cuBLAS dispatches: sgemm / ampere_sgemm kernel\n");
    printf("    → Uses: CUDA Cores (FP32 FMA units)\n");

    // Profile: FP32 GEMM (Tensor Cores via TF32)
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, d_W, K, d_x, K, &beta, d_y_fp32, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    printf("\n  FP32 SGEMM (Tensor Cores, TF32 ON): %.3f ms\n", elapsed);
    printf("    → cuBLAS dispatches: cutlass_tensorop / xmma kernel\n");
    printf("    → Uses: Tensor Cores (TF32 mode, 10-bit mantissa)\n");

    // Profile: Quantize kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    quantize_symmetric_kernel<<<(n_W + 255) / 256, 256>>>(d_W, d_W_i8, scale_W, n_W);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    printf("\n  Quantize FP32→INT8: %.3f ms\n", elapsed);
    printf("    → Custom CUDA kernel (elementwise)\n");
    printf("    → Uses: CUDA Cores (FP32 + INT8 conversion)\n");

    // Profile: INT8 IGEMM (Tensor Cores)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    cublasLtMatmul(ltHandle, matmulDesc,
                   &alpha_i, d_W_i8, layoutA,
                   d_x_i8, layoutB,
                   &beta_i, d_y_i32, layoutC,
                   d_y_i32, layoutC,
                   NULL, NULL, 0, 0);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    printf("\n  INT8 IGEMM (Tensor Cores): %.3f ms\n", elapsed);
    printf("    → cuBLASLt dispatches: imma / s8_tensorop kernel\n");
    printf("    → Uses: Tensor Cores (INT8 IMMA, 16x16x16 tiles)\n");

    // Profile: Dequantize kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    dequantize_bias_kernel<<<(M * N + 255) / 256, 256>>>(
        d_y_i32, d_y_int8_deq, combined_scale, d_b, M, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    printf("\n  Dequantize INT32→FP32 + bias: %.3f ms\n", elapsed);
    printf("    → Custom CUDA kernel (elementwise)\n");
    printf("    → Uses: CUDA Cores (INT32→FP32 + FMA)\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Clean up small demo
    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y_fp32)); CUDA_CHECK(cudaFree(d_W_i8)); CUDA_CHECK(cudaFree(d_x_i8));
    CUDA_CHECK(cudaFree(d_y_i32)); CUDA_CHECK(cudaFree(d_y_int8_deq)); CUDA_CHECK(cudaFree(d_err));
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);


    // ══════════════════════════════════════════
    // STEP 6: LLM-Scale Benchmark
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("STEP 6: LLM-Scale Benchmark\n");
    printf("============================================================\n");

    const int WARMUP = 50, ITERS = 200;
    printf("\nWarmup: %d iters, Benchmark: %d iters\n\n", WARMUP, ITERS);

    struct BenchConfig {
        const char* name;
        int batch, in_f, out_f;
    };

    BenchConfig configs[] = {
        {"GEMV  4Kx4K   (b=1)",     1,  4096,  4096},
        {"GEMV  4Kx11K  (b=1)",     1,  4096,  11008},
        {"GEMV  8Kx8K   (b=1)",     1,  8192,  8192},
        {"GEMM  4Kx11K  (b=32)",    32, 4096,  11008},
        {"GEMM  8Kx8K   (b=32)",    32, 8192,  8192},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    printf("%-28s %5s %7s %12s %12s %12s %12s\n",
           "Config", "Op", "Weight", "FP32 CUDA", "FP32 TF32", "INT8 IGEMM", "Best");
    printf("────────────────────────────────────────────────────────────"
           "────────────────────────────────────────\n");

    for (int ci = 0; ci < n_configs; ci++) {
        BenchConfig& cfg = configs[ci];
        int bM = cfg.batch, bK = cfg.in_f, bN = cfg.out_f;
        float wt_mb = (float)bK * bN * 4 / 1024 / 1024;
        const char* op = (bM == 1) ? "GEMV" : "GEMM";

        // Allocate
        float *bx, *bW, *by;
        CUDA_CHECK(cudaMalloc(&bx, bM * bK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&bW, bN * bK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&by, bM * bN * sizeof(float)));
        fill_random(bx, bM * bK, 100 + ci);
        fill_random(bW, bN * bK, 200 + ci);

        // ── FP32 TF32-OFF (CUDA Cores) ──
        FP32GemmCtx ctx_a = {cublas_handle, bM, bN, bK, bx, bW, by, 1.0f, 0.0f, false};
        BenchResult res_a = benchmark_cuda(fp32_gemm_fn, &ctx_a, WARMUP, ITERS);

        // ── FP32 TF32-ON (Tensor Cores) ──
        FP32GemmCtx ctx_b = {cublas_handle, bM, bN, bK, bx, bW, by, 1.0f, 0.0f, true};
        BenchResult res_b = benchmark_cuda(fp32_gemm_fn, &ctx_b, WARMUP, ITERS);

        // ── INT8 IGEMM (Tensor Cores) ──
        // Quantize
        int8_t *bx_i8, *bW_i8;
        int32_t *by_i32;
        CUDA_CHECK(cudaMalloc(&bx_i8, bM * bK));
        CUDA_CHECK(cudaMalloc(&bW_i8, bN * bK));
        CUDA_CHECK(cudaMalloc(&by_i32, bM * bN * sizeof(int32_t)));

        float s_w = gpu_abs_max(bW, bN * bK) / 127.0f;
        float s_x = gpu_abs_max(bx, bM * bK) / 127.0f;
        if (s_w < 1e-8f) s_w = 1e-8f;
        if (s_x < 1e-8f) s_x = 1e-8f;

        quantize_symmetric_kernel<<<(bN * bK + 255) / 256, 256>>>(bW, bW_i8, s_w, bN * bK);
        quantize_symmetric_kernel<<<(bM * bK + 255) / 256, 256>>>(bx, bx_i8, s_x, bM * bK);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Setup cuBLASLt for INT8
        cublasLtMatmulDesc_t bMatmulDesc;
        cublasLtMatrixLayout_t bLayoutA, bLayoutB, bLayoutC;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&bMatmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        cublasOperation_t bOpA = CUBLAS_OP_T, bOpB = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(bMatmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &bOpA, sizeof(bOpA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(bMatmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &bOpB, sizeof(bOpB)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayoutA, CUDA_R_8I, bK, bN, bK));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayoutB, CUDA_R_8I, bK, bM, bK));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayoutC, CUDA_R_32I, bN, bM, bN));

        INT8GemmCtx ctx_c = {ltHandle, bMatmulDesc, bLayoutA, bLayoutB, bLayoutC,
                             bM, bN, bK, bx_i8, bW_i8, by_i32, 1, 0};
        BenchResult res_c = benchmark_cuda(int8_gemm_fn, &ctx_c, WARMUP, ITERS);

        // Determine best
        const char* best = "CUDA";
        float best_ms = res_a.mean_ms;
        if (res_b.mean_ms < best_ms) { best_ms = res_b.mean_ms; best = "TF32"; }
        if (res_c.mean_ms < best_ms) { best_ms = res_c.mean_ms; best = "INT8"; }

        printf("%-28s %5s %5.0fMB   %6.3f ms    %6.3f ms    %6.3f ms     %s\n",
               cfg.name, op, wt_mb,
               res_a.mean_ms, res_b.mean_ms, res_c.mean_ms, best);

        // Cleanup
        cublasLtMatmulDescDestroy(bMatmulDesc);
        cublasLtMatrixLayoutDestroy(bLayoutA);
        cublasLtMatrixLayoutDestroy(bLayoutB);
        cublasLtMatrixLayoutDestroy(bLayoutC);
        CUDA_CHECK(cudaFree(bx)); CUDA_CHECK(cudaFree(bW)); CUDA_CHECK(cudaFree(by));
        CUDA_CHECK(cudaFree(bx_i8)); CUDA_CHECK(cudaFree(bW_i8)); CUDA_CHECK(cudaFree(by_i32));
    }


    // ══════════════════════════════════════════
    // Summary
    // ══════════════════════════════════════════
    printf("\n============================================================\n");
    printf("KEY TAKEAWAYS (CUDA C vs PyTorch)\n");
    printf("============================================================\n");

    printf("\n");
    printf("CUDA kernel mapping:\n");
    printf("  ┌─────────────────────┬───────────────────────┬────────────────┐\n");
    printf("  │ Operation           │ cuBLAS Kernel         │ GPU Core Type  │\n");
    printf("  ├─────────────────────┼───────────────────────┼────────────────┤\n");
    printf("  │ FP32 GEMV (b=1)     │ gemvx                 │ CUDA Cores     │\n");
    printf("  │ FP32 GEMM TF32=OFF  │ ampere_sgemm          │ CUDA Cores     │\n");
    printf("  │ FP32 GEMM TF32=ON   │ cutlass_tensorop      │ Tensor Cores   │\n");
    printf("  │ INT8 IGEMM          │ imma / s8_tensorop    │ Tensor Cores   │\n");
    printf("  │ Quantize FP32→INT8  │ custom elementwise    │ CUDA Cores     │\n");
    printf("  │ Dequantize+bias     │ custom elementwise    │ CUDA Cores     │\n");
    printf("  └─────────────────────┴───────────────────────┴────────────────┘\n");

    printf("\n");
    printf("CUDA C advantages over PyTorch:\n");
    printf("  1. Direct cuBLASLt IGEMM — true INT8×INT8→INT32 on Tensor Cores\n");
    printf("     (PyTorch torchao weight-only dequantizes to FP32 first)\n");
    printf("  2. No Python overhead, no autograd graph, no dispatcher\n");
    printf("  3. Explicit control over TF32, math mode, memory layout\n");
    printf("  4. cuBLASLt allows algorithm selection and workspace tuning\n");

    printf("\n");
    printf("  Your %s (sm%d.%d) has:\n", prop.name, prop.major, prop.minor);
    printf("  • %d SMs × 128 CUDA Cores = %d FP32 ALUs\n",
           prop.multiProcessorCount, prop.multiProcessorCount * 128);
    printf("  • %d SMs × 4 Tensor Cores = %d Tensor Cores\n",
           prop.multiProcessorCount, prop.multiProcessorCount * 4);
    printf("  • Tensor Cores handle: FP16, BF16, TF32, INT8, INT4\n");

    printf("\n");
    printf("Profile with Nsight for exact core usage:\n");
    printf("  nsys profile ./fp32_to_int8_cuda\n");
    printf("  ncu --set full ./fp32_to_int8_cuda\n");

    // Cleanup
    cublasLtDestroy(ltHandle);
    cublasDestroy(cublas_handle);

    return 0;
}
