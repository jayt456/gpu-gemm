# gpu-gemm

Single-precision matrix multiply (SGEMM) on an NVIDIA GPU, built up in stages from a
one-thread-per-output kernel to a 2-D register-tiled kernel, and measured against cuBLAS
at every step.

The point isn't to beat cuBLAS. It's that each kernel removes one specific hardware
bottleneck, the predicted effect was written down before the run, and the measurement
either confirmed it or didn't. Where it didn't, that's in here too.

**Result:** the 2-D register-tiled kernel reaches **79% of cuBLAS** at N=4096 and **7.1×**
a coalesced naive kernel, at 50% occupancy with zero register spills.

Hardware: Tesla T4 (Turing, 40 SMs, `sm_75`) on Google Colab, CUDA 12.x.

![Benchmark](results/benchmark.png)

## Results at N = 4096

| Kernel | Idea | GFLOP/s | vs. naive | % of cuBLAS | AI |
|---|---|---:|---:|---:|---:|
| `cpu` | i-k-j loop order, `-O3` | 4.8 *(N=1024)* | — | — | — |
| `naive_uncoal` | control: `threadIdx.x` → row | 120 | 0.28× | 3.1% | 0.25 |
| `tiled8` | shared-memory tiling, T=8 | 383 | 0.90× | 10.0% | 2 |
| `naive` | one thread per output | 427 | 1.00× | 11.1% | 0.25 |
| `tiled16` | T=16 | 630 | 1.47× | 16.4% | 4 |
| `tiled32` | T=32 | 836 | 1.96× | 21.8% | 8 |
| `reg1d` | 8 outputs/thread in registers | 1,725 | 4.04× | 44.9% | 16 |
| **`reg2d`** | **8×8 outputs/thread** | **3,028** | **7.09×** | **78.9%** | **32** |
| `cublas` | NVIDIA reference | 3,840 | 8.99× | 100% | — |

One run, median of 7 after a warm-up. Session-to-session variance is ~3%, so the last
digit is noise. AI is arithmetic intensity in FLOP/byte of DRAM traffic; the T4's balance
point is ~25, so everything below that line is memory-bound.

## The progression

**1. CPU baseline.** `i-k-j` loop order so the inner loop walks `B` and `C` contiguously —
several times faster than textbook `i-j-k` for identical arithmetic. A slow baseline would
make every number below meaningless.

**2. Naive.** One thread per output, accumulating in a register. `threadIdx.x` maps to the
column, so a warp's reads of `B` and writes to `C` are consecutive and coalesce. Intensity
0.25 FLOP/byte. Its roofline predicts 80 GFLOP/s; it measured 427, because the 4 MB L2
absorbs reuse the intensity figure doesn't model.

**3. The coalescing experiment.** `naive_uncoal` is arithmetically identical — one line
changes `threadIdx.x` to map to the row. A warp's 32 addresses are now `N` floats apart, so
one request scatters into 32 transactions delivering 4 useful bytes each. **Cost: 3.6×**,
from nothing but the thread-to-data mapping. It's the only controlled experiment here where
a single variable moves, so it stays in the repo on purpose.

**4. Shared-memory tiling.** Each block stages a `TILE × TILE` sub-block of `A` and `B` and
reuses each staged value `TILE` times, dropping DRAM traffic from O(N³) to O(N³/TILE).
Intensity becomes `TILE/4`. Two barriers are required — write-then-read, and read-then-write
before the next iteration overwrites the tile. Dropping the second is a silent race.

**5. Tile-size sweep.** Bigger tiles raise intensity but cost resident blocks, so the
optimum is empirical. Two results in [Negative results](#negative-results).

**6. 1-D register tiling.** Tiling plateaus because the inner loop issues two shared loads
per FMA. Each thread now computes 8 outputs in registers, so one `B` value feeds 8
multiply-adds. Intensity 16, and 512-thread blocks fit two per SM.

**7. 2-D register tiling.** `reg1d` gets no register reuse on the `A` side — its 8 `A`
values are each used once. An 8×8 thread block taking an outer product per k-step fixes it:
16 shared loads feed 64 FMAs. Intensity 32, clearing the T4's balance point of 25.

Two details make it fast rather than just better on paper. `A` is staged **transposed**, so
the compute loop reads a column contiguously — which is what makes the shared reads
vectorizable as `float4`, cutting load instructions 4× and dropping a 4-way bank conflict
to 2-way. Global loads stay scalar so the bounds guards hold for any N.

Measured **3,028 GFLOP/s, 1.76× over `reg1d`**, against a 1.5–2× prediction. `ptxas`
confirms the resource model: **119 registers, 0 bytes spilled**, 8 KB shared → two blocks
per SM at 50% occupancy, exactly the trade the design was making.

## Negative results

- **`tiled8` is slower than not tiling at all** — 383 vs. `naive`'s 427. Eight times the
  arithmetic intensity, and it loses. An 8×8 tile leaves only eight iterations of work
  between two barriers, so synchronization overhead eats the gain while `naive` collects a
  large uncounted L2 assist. Intensity is necessary, not sufficient.
- **`tiled16` decays with N while `tiled32` holds flat** — 855 → 658 → 630 GFLOP/s across
  N = 1024/2048/4096, against 863 → 866 → 836. Within 1% of each other at N=1024, 32% apart
  by N=2048. Consistent with L2 covering the lower-intensity kernel at small N and
  progressively failing to.
- **`reg2d` is the worst kernel in the repo at N=256** — 251 GFLOP/s, below naive. A 128×128
  block tile yields 4 blocks, so 36 of 40 SMs idle. Big tiles need a big problem.
- **Padding shared memory (`[TILE][TILE+1]`) did nothing.** With a 32×32 block, `As[ty][k]`
  is one address for the whole warp (broadcast) and `Bs[k][tx]` is 32 consecutive floats.
  There was no conflict to remove.
- **cuBLAS drops from ~5,600 GFLOP/s at N=2048 to ~3,840 at N=4096.** My kernels don't.
  Reproduced across three sessions, so systematic rather than noise — most likely thermal
  throttling, since cuBLAS is benchmarked last at each size. Unconfirmed, so it stays
  flagged rather than explained away.

## Correctness

A misindexed CUDA kernel returns garbage without crashing, so performance numbers mean
nothing without gates:

- **Deterministic self-test before any benchmarking.** With `A = B = 1`, every element of
  `C` must equal exactly `N` — no floating-point tolerance. It runs at **N = 257**, prime to
  every tile size here, so it also exercises every ragged-edge bounds guard. The benchmark
  refuses to run if it fails.
- **Element-wise comparison against the CPU reference** at N ≤ 1024 (max relative error
  < 1e-3, typically ~2e-7). Larger sizes are reported as unverified, not assumed correct.
- **`compute-sanitizer` clean** after every kernel change.

The error metric normalizes by `max|C|`, not per element. Per-element relative error is
meaningless for GEMM on random inputs: outputs are sums of random-signed products, so many
land near zero by cancellation and ordinary rounding then reports a huge relative error on
a correct number. Tried it per-element first and got `[FAIL]` on everything, cuBLAS included.

## Build and run

```bash
make ARCH=sm_75          # T4. sm_80 for A100, sm_89 for L4
mkdir -p results
./bin/gemm               # self-test, then benchmark, then writes results/results.csv
python bench/plot.py     # -> results/benchmark.png, results/summary.png
```

Takes a couple of minutes, most of it `naive_uncoal` at N=4096 being slow on purpose.
Optional checks: `make regs` (register counts and spill bytes) and
`compute-sanitizer ./bin/gemm`.

## Methodology

Kernel time only, via `cudaEvent` — launches are asynchronous, so a host timer measures
enqueue time, not execution. One untimed warm-up, then the median of 7 (GPU timing noise is
one-sided). GFLOP/s = 2N³ / time, transfers excluded. Square matrices,
N ∈ {256, 512, 1024, 2048, 4096}.

## Limitations and what's next

Square only (M = N = K), FP32 only with no Tensor Core path, scalar global loads, and no
double buffering. In rough order of expected payoff: double-buffered tile prefetch,
warp-level tiling, vectorized global loads, `ncu` profiling, then a TF32 `mma` path — which
is most of the remaining gap to cuBLAS.

[`docs/ANALYSIS.md`](docs/ANALYSIS.md) has the intensity derivations, roofline placement,
occupancy accounting, and the shared-memory load-to-FMA analysis that actually explains the
speedups.
