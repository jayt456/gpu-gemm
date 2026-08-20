#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
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

// Median of `iters` timed runs, after one warm-up.
template <typename Launch>
double time_kernel(Launch launch, int iters = 7)
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

int main(int argc, char **argv)
{
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    const int CPU_LIMIT = 1024; // CPU reference is O(N^3) — slow

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | CC %d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    std::vector<Row> rows;

    for (int N : sizes)
    {
        const size_t n2 = (size_t)N * N, bytes = n2 * sizeof(float);
        std::vector<float> hA(n2), hB(n2), hRef(n2), hOut(n2);
        fill_random(hA.data(), n2, 1);
        fill_random(hB.data(), n2, 2);

        bool haveRef = (N <= CPU_LIMIT);
        double cpu_ms = 0.0;
        if (haveRef)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_cpu(hA.data(), hB.data(), hRef.data(), N);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            rows.push_back({N, "cpu", cpu_ms, gflops(N, cpu_ms), 0.0});
        }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        auto record = [&](const char *name, double ms)
        {
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
            double err = haveRef ? max_rel_err(hRef.data(), hOut.data(), n2) : -1.0;
            rows.push_back({N, name, ms, gflops(N, ms), err});
            printf("  N=%-5d %-14s %9.3f ms  %8.1f GFLOP/s  relerr=%.2e %s\n",
                   N, name, ms, gflops(N, ms), err,
                   (err < 0 ? "(unverified)" : (err < 1e-3 ? "[PASS]" : "[FAIL]")));
        };

        // --- naive ---
        {
            dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
            record("naive", time_kernel([&]
                                        { k_naive<<<grd, blk>>>(dA, dB, dC, N); }));
        }
        // --- naive uncoalesced (experiment) ---
        {
            dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
            record("naive_uncoal", time_kernel([&]
                                               { k_naive_uncoalesced<<<grd, blk>>>(dA, dB, dC, N); }));
        }
        // --- tiled ---
        {
            constexpr int T = 32;
            dim3 blk(T, T), grd((N + T - 1) / T, (N + T - 1) / T);
            record("tiled32", time_kernel([&]
                                          { k_tiled<T><<<grd, blk>>>(dA, dB, dC, N); }));
        }
        // --- register tiled ---
        {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 blk(BM * BN / TM), grd((N + BN - 1) / BN, (N + BM - 1) / BM);
            record("reg1d", time_kernel([&]
                                        { k_reg1d<BM, BN, BK, TM><<<grd, blk>>>(dA, dB, dC, N); }));
        }
        // --- cuBLAS ---
        // cuBLAS is COLUMN-major. For row-major C = A*B, pass B and A swapped:
        // column-major sees C^T = B^T * A^T = (A*B)^T, which is exactly our C.
        {
            const float alpha = 1.0f, beta = 0.0f;
            record("cublas", time_kernel([&]
                                         { cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                       N, N, N, &alpha, dB, N, dA, N, &beta, dC, N); }));
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        printf("\n");
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    FILE *f = fopen("results/results.csv", "w");
    fprintf(f, "N,kernel,ms,gflops,rel_err\n");
    for (auto &r : rows)
        fprintf(f, "%d,%s,%.4f,%.2f,%.3e\n",
                r.N, r.name.c_str(), r.ms, r.gf, r.relerr);
    fclose(f);
    printf("Wrote results/results.csv\n");
    return 0;
}