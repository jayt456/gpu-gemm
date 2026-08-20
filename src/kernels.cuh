#pragma once
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// K1: NAIVE
// One thread per output element.
// threadIdx.x -> col, so consecutive threads in a warp read consecutive
// addresses of B and write consecutive addresses of C  => COALESCED.
// Bottleneck: every element of A is re-read from global memory N times.
// Arithmetic intensity = 2 FLOP / 8 bytes = 0.25 FLOP/byte  (memory bound).
// ---------------------------------------------------------------------------
__global__ void k_naive(const float *__restrict__ A,
                        const float *__restrict__ B,
                        float *__restrict__ C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// K1b: NAIVE, UNCOALESCED  (experiment only — deliberately bad)
// Identical arithmetic, but threadIdx.x -> row. Consecutive threads now touch
// addresses N floats apart, so each warp access scatters across many memory
// transactions. Used to isolate the cost of access pattern alone.
// ---------------------------------------------------------------------------
__global__ void k_naive_uncoalesced(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // <-- swapped
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// K2: SHARED-MEMORY TILED
// Each block stages a TILE x TILE sub-block of A and B into shared memory,
// then reuses it TILE times. Global traffic drops from O(N^3) to O(N^3/TILE).
// Arithmetic intensity = TILE/4 FLOP/byte (8.0 at TILE=32) — a 32x improvement.
//
// Two __syncthreads() are required:
//   (1) after loading  — no thread may read a tile before all threads wrote it
//   (2) after computing — no thread may overwrite a tile before all threads
//       finished reading it. Omitting (2) is a silent, intermittent race.
//
// Out-of-range loads are zero-filled, so N need not be a multiple of TILE.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void k_tiled(const float *__restrict__ A,
                        const float *__restrict__ B,
                        float *__restrict__ C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int nTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < nTiles; ++t)
    {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        As[ty][tx] = (row < N && aCol < N) ? A[(size_t)row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads(); // (1) tiles are loaded

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads(); // (2) everyone done reading
    }

    if (row < N && col < N)
        C[(size_t)row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// K3: REGISTER-TILED (1D thread coarsening)
// Shared-memory tiling alone tops out around TILE/4 FLOP/byte, still short of
// the ~13 FLOP/byte modern GPUs need to be compute bound. The next lever is to
// give each thread MORE WORK: here each thread computes TM output elements
// (a vertical strip), accumulating in registers — the fastest storage there is.
// One value of B loaded from shared memory now feeds TM multiply-adds.
//
// Config: BM=BN=64, BK=8, TM=8  ->  blockDim.x = BM*BN/TM = 512 threads.
//   As holds BM*BK = 512 floats, Bs holds BK*BN = 512  -> one load per thread.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void k_reg1d(const float *__restrict__ A,
                        const float *__restrict__ B,
                        float *__restrict__ C, int N)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;

    // compute mapping: this thread owns column `threadCol`, rows [threadRow*TM, +TM)
    const int threadCol = tid % BN; // 0 .. BN-1
    const int threadRow = tid / BN; // 0 .. BM/TM-1

    // load mapping (one element each)
    const int aRow = tid / BK, aCol = tid % BK; // As is BM x BK
    const int bRow = tid / BN, bCol = tid % BN; // Bs is BK x BN

    float acc[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i)
        acc[i] = 0.0f;

    for (int t = 0; t < N; t += BK)
    {
        const int gARow = cRow + aRow, gACol = t + aCol;
        const int gBRow = t + bRow, gBCol = cCol + bCol;

        As[aRow * BK + aCol] =
            (gARow < N && gACol < N) ? A[(size_t)gARow * N + gACol] : 0.0f;
        Bs[bRow * BN + bCol] =
            (gBRow < N && gBCol < N) ? B[(size_t)gBRow * N + gBCol] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const float bval = Bs[k * BN + threadCol]; // one load...
#pragma unroll
            for (int m = 0; m < TM; ++m) // ...feeds TM FMAs
                acc[m] += As[(threadRow * TM + m) * BK + k] * bval;
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        const int r = cRow + threadRow * TM + m;
        const int c = cCol + threadCol;
        if (r < N && c < N)
            C[(size_t)r * N + c] = acc[m];
    }
}