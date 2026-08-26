#pragma once
#include <cuda_runtime.h>

// One thread per output element. threadIdx.x -> col, so a warp reads consecutive
// B and writes consecutive C. Every element of A is re-read from DRAM N times.
// AI = 0.25 FLOP/byte.
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

// Control experiment. Arithmetically identical to k_naive; only threadIdx.x -> row
// changes, so a warp's addresses stride by N. Isolates access pattern as the one
// variable. Deliberately slow -- do not "fix".
__global__ void k_naive_uncoalesced(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // swapped
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

// The block stages a TILE x TILE sub-block of A and B into shared memory and
// reuses each staged value TILE times. AI = TILE/4 FLOP/byte.
// Out-of-range loads are zero-filled, so N need not be a multiple of TILE.
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
        __syncthreads(); // tile is staged

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads(); // everyone done reading before it is overwritten
    }

    if (row < N && col < N)
        C[(size_t)row * N + col] = acc;
}

// Each thread computes TM outputs (a column strip) in registers, so one B value
// from shared memory feeds TM FMAs instead of one. AI = 16 FLOP/byte.
// At 64,64,8,8 every thread stages exactly one element of each tile.
template <int BM, int BN, int BK, int TM>
__global__ void k_reg1d(const float *__restrict__ A,
                        const float *__restrict__ B,
                        float *__restrict__ C, int N)
{
    static_assert(BM * BK == BM * BN / TM, "As staging is not one load/thread");
    static_assert(BK * BN == BM * BN / TM, "Bs staging is not one load/thread");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;

    const int threadCol = tid % BN; // owns this column
    const int threadRow = tid / BN; // owns rows [threadRow*TM, +TM)

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

// Each thread computes a TM x TN block via an outer product per k-step, so
// TM + TN shared loads feed TM*TN FMAs (16 -> 64 at 8x8, vs 9 -> 8 in k_reg1d).
// AI = 32 FLOP/byte, past the T4's 25 FLOP/byte balance point.
//
// As is staged transposed so the compute loop reads a column of A contiguously,
// which is what makes the float4 shared reads below possible. Global loads stay
// scalar so the bounds guards still hold for any N.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void k_reg2d(const float *__restrict__ A,
                        const float *__restrict__ B,
                        float *__restrict__ C, int N)
{
    constexpr int nThreads = BM * BN / (TM * TN);
    constexpr int strideA = nThreads / BK; // rows of As staged per pass
    constexpr int strideB = nThreads / BN; // rows of Bs staged per pass

    static_assert(TM % 4 == 0 && TN % 4 == 0, "float4 reads need TM,TN % 4 == 0");
    static_assert(nThreads % BK == 0 && nThreads % BN == 0, "bad staging map");
    static_assert(strideA > 0 && strideB > 0, "block too small to stage a tile");
    static_assert(BM % strideA == 0 && BK % strideB == 0, "staging leaves holes");

    __shared__ __align__(16) float As[BK * BM]; // transposed: As[k][row]
    __shared__ __align__(16) float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;

    const int threadCol = tid % (BN / TN); // owns cols [threadCol*TN, +TN)
    const int threadRow = tid / (BN / TN); // owns rows [threadRow*TM, +TM)

    const int aCol = tid % BK, aRow = tid / BK;
    const int bCol = tid % BN, bRow = tid / BN;

    float acc[TM][TN];
    float regA[TM], regB[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    for (int t = 0; t < N; t += BK)
    {
#pragma unroll
        for (int off = 0; off < BM; off += strideA)
        {
            const int r = aRow + off;
            const int gr = cRow + r, gc = t + aCol;
            As[aCol * BM + r] =
                (gr < N && gc < N) ? A[(size_t)gr * N + gc] : 0.0f;
        }
#pragma unroll
        for (int off = 0; off < BK; off += strideB)
        {
            const int r = bRow + off;
            const int gr = t + r, gc = cCol + bCol;
            Bs[r * BN + bCol] =
                (gr < N && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            // Unpacked element-wise rather than via *(float4*)&regA[i]: taking the
            // address of a local array can push it out of registers into local memory.
#pragma unroll
            for (int i = 0; i < TM; i += 4)
            {
                const float4 v = *reinterpret_cast<const float4 *>(
                    &As[k * BM + threadRow * TM + i]);
                regA[i] = v.x; regA[i + 1] = v.y;
                regA[i + 2] = v.z; regA[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4)
            {
                const float4 v = *reinterpret_cast<const float4 *>(
                    &Bs[k * BN + threadCol * TN + j]);
                regB[j] = v.x; regB[j + 1] = v.y;
                regB[j + 2] = v.z; regB[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        const int r = cRow + threadRow * TM + i;
        if (r >= N)
            continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const int c = cCol + threadCol * TN + j;
            if (c < N)
                C[(size_t)r * N + c] = acc[i][j];
        }
    }
}
