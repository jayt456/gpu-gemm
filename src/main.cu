#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common.cuh"
#include "kernels.cuh"

struct Row
{
    int N;
    std::string name;
    double ms;
    double gf;
    double relerr;
};

using Launch = std::function<void(const float *, const float *, float *, int)>;

struct Kernel
{
    const char *name;
    Launch run;
};

// Median of `iters` timed runs after one warm-up. cudaEvent rather than a host
// timer because launches are async; median rather than mean because GPU timing
// noise is one-sided.
template <typename F>
double time_kernel(F launch, int iters = 7)
{
    launch(); // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<double> samples;
    for (int i = 0; i < iters; ++i)
    {
        CUDA_CHECK(cudaEventRecord(beg));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        samples.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// One launcher per kernel. Free functions rather than inline lambdas so the
// registry below reads as a table.
static void launch_naive(const float *A, const float *B, float *C, int N)
{
    dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
    k_naive<<<grd, blk>>>(A, B, C, N);
}

static void launch_naive_uncoal(const float *A, const float *B, float *C, int N)
{
    dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
    k_naive_uncoalesced<<<grd, blk>>>(A, B, C, N);
}

template <int T>
static void launch_tiled(const float *A, const float *B, float *C, int N)
{
    dim3 blk(T, T), grd((N + T - 1) / T, (N + T - 1) / T);
    k_tiled<T><<<grd, blk>>>(A, B, C, N);
}

static void launch_reg1d(const float *A, const float *B, float *C, int N)
{
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 blk(BM * BN / TM), grd((N + BN - 1) / BN, (N + BM - 1) / BM);
    k_reg1d<BM, BN, BK, TM><<<grd, blk>>>(A, B, C, N);
}

static void launch_reg2d(const float *A, const float *B, float *C, int N)
{
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 blk(BM * BN / (TM * TN)), grd((N + BN - 1) / BN, (N + BM - 1) / BM);
    k_reg2d<BM, BN, BK, TM, TN><<<grd, blk>>>(A, B, C, N);
}

static std::vector<Kernel> build_kernels(cublasHandle_t handle)
{
    return {
        {"naive", launch_naive},
        {"naive_uncoal", launch_naive_uncoal},
        {"tiled8", launch_tiled<8>},
        {"tiled16", launch_tiled<16>},
        {"tiled32", launch_tiled<32>},
        {"reg1d", launch_reg1d},
        {"reg2d", launch_reg2d},
        // cuBLAS is column-major. For row-major C = A*B, pass B and A swapped:
        // column-major then computes C^T = B^T * A^T = (A*B)^T, which is our C.
        {"cublas", [handle](const float *A, const float *B, float *C, int N)
         {
             const float alpha = 1.0f, beta = 0.0f;
             CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      N, N, N, &alpha, B, N, A, N, &beta, C, N));
         }},
    };
}

// With A = B = 1 every element of C must equal exactly N (integer-exact in fp32).
// N=257 is prime to every tile size here, so this also exercises every
// ragged-edge bounds guard.
static bool self_test(const std::vector<Kernel> &kernels, int N)
{
    const size_t n2 = (size_t)N * N, bytes = n2 * sizeof(float);
    std::vector<float> hOnes(n2, 1.0f), hOut(n2);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hOnes.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hOnes.data(), bytes, cudaMemcpyHostToDevice));

    printf("Self-test (A=B=1, N=%d, every C[i][j] must equal %d exactly)\n", N, N);
    bool all_ok = true;
    for (const auto &k : kernels)
    {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        k.run(dA, dB, dC, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));

        size_t bad = 0;
        for (size_t i = 0; i < n2; ++i)
            if (hOut[i] != (float)N)
                ++bad;
        all_ok &= (bad == 0);
        if (bad == 0)
            printf("  %-14s [PASS]\n", k.name);
        else
            printf("  %-14s [FAIL] %zu/%zu elements wrong (C[0][0]=%.6f)\n",
                   k.name, bad, n2, hOut[0]);
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    printf("\n");
    return all_ok;
}

int main(int argc, char **argv)
{
    const std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    const int CPU_LIMIT = 1024; // the O(N^3) CPU reference is too slow past this

    bool gate = true;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--no-gate")
            gate = false;

    // Opened up front: the benchmark takes minutes and the CSV is its only output.
    FILE *csv = fopen("results/results.csv", "w");
    if (!csv)
    {
        fprintf(stderr, "Cannot write results/results.csv -- run `mkdir -p results` first.\n");
        return 1;
    }
    fprintf(csv, "N,kernel,ms,gflops,rel_err\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | CC %d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const std::vector<Kernel> kernels = build_kernels(handle);

    if (gate)
    {
        if (!self_test(kernels, 257))
        {
            fprintf(stderr, "Self-test failed; not benchmarking.\n");
            return 1;
        }
    }
    else
        printf("WARNING: --no-gate given, correctness gate skipped.\n"
               "         Do not publish numbers from this run.\n\n");

    std::vector<Row> rows;
    for (int N : sizes)
    {
        const size_t n2 = (size_t)N * N, bytes = n2 * sizeof(float);
        std::vector<float> hA(n2), hB(n2), hOut(n2), hRef;
        fill_random(hA.data(), n2, 1);
        fill_random(hB.data(), n2, 2);

        const bool haveRef = (N <= CPU_LIMIT);
        if (haveRef)
        {
            hRef.resize(n2);
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_cpu(hA.data(), hB.data(), hRef.data(), N);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            rows.push_back({N, "cpu", ms, gflops(N, ms), 0.0});
            printf("  N=%-5d %-14s %9.3f ms  %8.1f GFLOP/s\n", N, "cpu", ms, gflops(N, ms));
        }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        for (const auto &k : kernels)
        {
            const double ms = time_kernel([&]
                                          { k.run(dA, dB, dC, N); });
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
            const double err = haveRef ? max_rel_err(hRef.data(), hOut.data(), n2) : -1.0;
            rows.push_back({N, k.name, ms, gflops(N, ms), err});
            printf("  N=%-5d %-14s %9.3f ms  %8.1f GFLOP/s  relerr=%.2e %s\n",
                   N, k.name, ms, gflops(N, ms), err,
                   err < 0 ? "(unverified)" : (err < 1e-3 ? "[PASS]" : "[FAIL]"));
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        printf("\n");
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    for (const auto &r : rows)
        fprintf(csv, "%d,%s,%.4f,%.2f,%.3e\n",
                r.N, r.name.c_str(), r.ms, r.gf, r.relerr);
    fclose(csv);
    printf("Wrote results/results.csv\n");
    return 0;
}
