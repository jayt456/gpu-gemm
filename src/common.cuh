#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err_ = (call);                                 \
        if (err_ != cudaSuccess)                                   \
        {                                                          \
            fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n",          \
                    cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                       \
    do                                                           \
    {                                                            \
        cublasStatus_t st_ = (call);                             \
        if (st_ != CUBLAS_STATUS_SUCCESS)                        \
        {                                                        \
            fprintf(stderr, "[cuBLAS ERROR] code %d at %s:%d\n", \
                    (int)st_, __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

inline void fill_random(float *m, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        m[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
}

// CPU reference. i-k-j ordering: the inner loop walks B and C contiguously,
// which is far more cache-friendly than the textbook i-j-k.
inline void matmul_cpu(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        float *crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j)
            crow[j] = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            const float a = A[(size_t)i * N + k];
            const float *brow = B + (size_t)k * N;
            for (int j = 0; j < N; ++j)
                crow[j] += a * brow[j];
        }
    }
}

// Error normalized by the magnitude of the RESULT, not per-element.
// Per-element relative error is meaningless for GEMM: sums of random-signed
// products pass near zero by cancellation, so a tiny element with ordinary
// float rounding reports an enormous "relative" error.
inline double max_rel_err(const float *ref, const float *test, size_t n)
{
    double maxabs = 0.0, maxdiff = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        maxabs = std::max(maxabs, std::fabs((double)ref[i]));
        maxdiff = std::max(maxdiff, std::fabs((double)ref[i] - (double)test[i]));
    }
    return maxdiff / (maxabs > 1e-12 ? maxabs : 1.0);
}

inline double gflops(int N, double ms)
{
    return 2.0 * (double)N * N * N / (ms * 1e6);
}