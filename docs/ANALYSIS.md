# Analysis

Derivations and accounting behind the numbers in the README. Hardware throughout is a
Tesla T4 (Turing, `sm_75`, 40 SMs).

## 1. Machine model

| Quantity | Value |
|---|---|
| FP32 peak | ~8,100 GFLOP/s |
| DRAM bandwidth | ~320 GB/s |
| **Balance point** | **~25 FLOP/byte** |
| Shared memory / SM | 64 KB |
| 32-bit registers / SM | 65,536 |
| Max threads / SM | 1,024 (32 warps) |
| Max threads / block | 1,024 |
| L2 cache | 4 MB |

The balance point is the number that organizes everything: for each byte the T4 pulls from
DRAM, it can retire 25 FLOPs in the time that byte takes to arrive. A kernel below 25
FLOP/byte leaves the arithmetic units idle waiting on memory.

GEMM's *inherent* intensity is enormous — 2N³ FLOPs over 12N² bytes is N/6, which is 683 at
N=4096. None of the work below is about reducing arithmetic. It's entirely about actually
capturing reuse that the algorithm already contains, at successively faster levels of the
memory hierarchy.

## 2. Arithmetic intensity derivations

**Naive.** Per inner-loop step: two 4-byte loads, one multiply, one add.

```
AI = 2 FLOP / 8 bytes = 0.25 FLOP/byte
```

**Tiled, tile size T.** Per tile step the block stages `2T²` floats and performs `2T³`
FLOPs (a T×T by T×T product):

```
AI = 2T³ / (4 · 2T²) = T/4        ->  2.0 (T=8), 4.0 (T=16), 8.0 (T=32)
```

**1-D register tiled** (`BM=BN=64, BK=8, TM=8`). Per k-chunk the block stages
`BM·BK + BK·BN` floats and performs `BM·BN·BK·2` FLOPs:

```
bytes = (64·8 + 8·64) · 4 = 4,096
FLOPs = 64 · 64 · 8 · 2   = 65,536
AI    = 16 FLOP/byte
```

**2-D register tiled** (`BM=BN=128, BK=8, TM=TN=8`). Same formula, larger block tile:

```
bytes = (128·8 + 8·128) · 4 = 8,192
FLOPs = 128 · 128 · 8 · 2   = 262,144
AI    = 32 FLOP/byte
```

Intensity for these kernels is `1 / (1/BM + 1/BN) · 2`, i.e. it's governed by the *block*
tile, not the per-thread tile. Doubling BM and BN doubles it; changing BK does not.

## 3. Roofline placement, and where the model stops working

| Kernel | AI | DRAM roof (AI × 320 GB/s) | Measured @4096 | Fraction of its roof |
|---|---:|---:|---:|---:|
| `naive` | 0.25 | 80 | 427 | **5.3×** |
| `tiled8` | 2 | 640 | 383 | 0.60× |
| `tiled16` | 4 | 1,280 | 630 | 0.49× |
| `tiled32` | 8 | 2,560 | 836 | 0.33× |
| `reg1d` | 16 | 5,120 | 1,725 | 0.34× |
| `reg2d` | 32 | 8,100 (compute-capped) | 3,028 | 0.37× |

Two things are wrong with a naive reading of this table, and both are informative.

**`naive` beats its roofline by 5×.** The intensity figure counts every load as a DRAM
load. It isn't — the 4 MB L2 catches a large fraction of the reuse for free. The roofline
bounds *DRAM* traffic, not total traffic, so for cache-friendly access patterns the DRAM-only 
intensity underestimates the kernel’s real intensity, and the measured point can land 
above the roof.

**Everything from `tiled8` up falls far short of its roof.** If DRAM bandwidth were the
binding constraint they would sit near it; instead they cluster at 0.33–0.60×. This matters,
because the obvious story — "AI doubled 8 → 16, throughput doubled 836 → 1725" — is a
correlation with the wrong mechanism attached. Both kernels are well clear of the DRAM
roof. What actually changed is the next level down.

Note also that the fraction-of-roof column is roughly *flat* from `tiled32` to `reg2d`
(0.33 → 0.34 → 0.37). Doubling arithmetic intensity moved the roof and the measurement by
about the same factor each time, which is what you would expect if the roof is not the
thing being hit.

## 4. The shared-memory bottleneck

Once a kernel is tiled, the inner loop's cost is set by how many shared-memory loads it
issues per FMA, because shared memory cannot sustain one load per cycle per lane the way
the FMA pipeline can retire arithmetic.

| Kernel | Inner loop, per k-step, per thread | Loads : FMAs |
|---|---|---:|
| `k_tiled<T>` | 2 loads, 1 FMA | 2 : 1 |
| `k_reg1d` | 1 `B` load + 8 `A` loads, 8 FMAs | 9 : 8 |
| `k_reg2d`, scalar | 8 `A` + 8 `B` loads, 64 FMAs | 1 : 4 |
| `k_reg2d`, `float4` | 2 + 2 `LDS.128`, 64 FMAs | 1 : 16 |

The `tiled32` → `reg1d` step improves this ratio by ~1.8×, against a measured 2.06×. The
`reg1d` → `reg2d` step improves it by ~4.5× before vectorization and ~18× after, against a
measured **1.76×** — so the load-to-FMA ratio stops being the binding constraint somewhere
inside that step, which is consistent with `reg2d` being the first kernel whose intensity
(32) clears the machine balance point (25).

It also shows exactly what `reg1d` leaves on the table. Its 8 `A` values per k-step are
each used once — all the register-level reuse is on the `B` side. The 2-D tile fixes the
asymmetry: `TM` values of `A` and `TN` values of `B` combine into `TM·TN` products, so both
operands get reused `TM` or `TN` times instead of one of them getting reused and the other
not. That's a 4.5× improvement in the ratio at 8×8, before vectorization, and 18× after.

## 5. Occupancy accounting

Occupancy is resident warps ÷ 32. It matters because it's the only mechanism the GPU has
for hiding memory latency: when a warp stalls, the scheduler switches to another one.

| Kernel | Threads/block | Shared/block | Blocks/SM | Warps/SM | Occupancy |
|---|---:|---:|---:|---:|---:|
| `naive` | 256 | 0 | 4 | 32 | 100% |
| `tiled8` | 64 | 512 B | 16 (block cap) | 32 | 100% |
| `tiled16` | 256 | 2 KB | 4 | 32 | 100% |
| `tiled32` | 1024 | 8 KB | **1** | 32 | 100% |
| `reg1d` | 512 | 4 KB | 2 | 32 | 100% |
| `reg2d` | 256 | 8 KB | 2 (register-limited) | 16 | 50% |

**Measured** (`ptxas -v`, sm_75): `reg2d` uses **119 registers, 0 bytes spill stores, 0 spill
loads, 8192 bytes smem**. 119 × 256 = 30,464 registers per block, so 65,536 / 30,464 = 2
blocks resident — 16 warps, 50% occupancy, exactly as predicted above.

`tiled32` is the interesting row. Its occupancy is 100% by the standard metric, so the
usual "raise occupancy" advice doesn't apply. The problem is that all 32 resident warps
belong to the *same block*, so they all hit `__syncthreads()` together and the SM stalls
with nothing to overlap. `reg1d`'s 512-thread block fits two blocks per SM, and one block's
staging phase can overlap the other's compute phase. Part of the `tiled32 → reg1d` speedup
is this, not intensity.

`reg2d` trades the other way. 64 accumulators per thread plus `regA`/`regB` pushes register
use to 119 per thread, capping the SM at two blocks and 50% occupancy. That's the intended
trade: the kernel is dense enough in arithmetic that it needs less latency hiding, and at
3,028 GFLOP/s against `reg1d`'s 1,725 at full occupancy, the trade paid.

## 6. Memory access patterns

**Global coalescing.** A warp's 32 addresses are serviced in 32-byte sectors.

| Access | Pattern across a warp | Sectors | Bus efficiency |
|---|---|---:|---:|
| `naive`, reads of `B` | 32 consecutive floats | 4 | 100% |
| `naive`, reads of `A` | all lanes same address | 1 (broadcast) | 100% |
| `naive_uncoal`, reads of `A` | stride N floats | 32 | 12.5% |
| `reg1d`, staging `B` | 64 consecutive floats | 8 | 100% |
| `reg1d`, staging `A` | 8 floats, then jump a row | 4 | 100%¹ |
| `reg2d`, staging `B` | 128 consecutive floats | 16 | 100% |
| `reg2d`, staging `A` | 8 floats, then jump a row | 4 | 100%¹ |

¹ Four separate 32-byte transactions rather than one 128-byte one, but each sector is fully
consumed, so no bandwidth is wasted. Requires `N % 8 == 0` for the sectors to be aligned;
at other sizes some transactions straddle a boundary.

The 12.5% row is `naive_uncoal`: each transaction hauls 32 bytes to deliver 4. Theory says
8× worse; measured is 3.6×, and the difference is L2 recovering some of the scatter.

**Shared-memory bank conflicts.** 32 banks, 4 bytes wide, interleaved by word address.

- `k_tiled<32>`: `As[ty][k]` is one address for the whole warp (broadcast, free);
  `Bs[k][tx]` is 32 consecutive words across 32 distinct banks. **No conflict** — which is
  why padding to `[TILE][TILE+1]` measured no change.
- `k_reg1d`: `As[...]` is a broadcast (all lanes share `threadRow`); `Bs[k·BN + threadCol]`
  is consecutive. **No conflict.**
- `k_reg2d` with scalar loads: `Bs[k·BN + threadCol·TN + j]` puts lanes 8 words apart, so
  the 16 distinct addresses land on 4 banks. **4-way conflict.** `As` is fine (broadcast
  within each half of the warp).
- `k_reg2d` with `float4`: the warp is serviced in quarters of 8 lanes. `A` broadcasts;
  `B` drops to a **2-way conflict** and does it in a quarter of the instructions. Net
  ~8× less time in the `B` load. Alignment is structural — the offsets `k·BM + threadRow·TM`
  and `k·BN + threadCol·TN` are multiples of 8 floats by construction, so this doesn't
  depend on N.

## 7. Verification without a GPU

The development machine has no NVIDIA hardware, so correctness has to survive being written
before it can be run. Three things carry the load:

1. **`static_assert` on the tiling algebra.** `k_reg1d` asserts
   `BM·BK == BK·BN == BM·BN/TM` (its one-load-per-thread staging depends on it);
   `k_reg2d` asserts that the strided staging map covers each tile exactly once and that
   `TM, TN` are multiples of 4 for the `float4` reads. A bad retune fails to compile
   instead of returning wrong numbers.
2. **CPU emulation of the index arithmetic.** `k_reg2d`'s block/thread/staging loops were
   replayed serially on the host at N ∈ {1, 7, 37, 128, 129, 257, 512}, checking both the
   result against a plain matmul and that every shared-memory slot is written exactly once
   per k-chunk. Exactly-once is the useful invariant: a hole means an uninitialized read, a
   duplicate means a race.
3. **The all-ones self-test at N=257**, described in the README, as the on-device gate.

All three held: `k_reg2d` compiled and passed the gate on first contact with hardware, and
its measured occupancy matched the prediction in §5 exactly.

## 8. Open questions

- Confirm the cuBLAS dip at N=4096 is throttling. It has now reproduced in three separate
  sessions (39%, 36%, 32% drops from N=2048), so it is systematic. Sample
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` during the run, or run the sizes in
  reverse and see whether the dip follows the clock rather than the size.
- `ncu` per kernel: achieved occupancy, DRAM throughput as % of peak, and compute
  throughput as % of peak. The expected story is `naive` pinned near bandwidth peak with
  low compute utilization, and `reg2d` inverted. Colab's `ncu` support is inconsistent.
- Why does `tiled8` (383 GFLOP/s) lose to `naive` (427) despite 8× the arithmetic
  intensity? Working hypothesis: only 8 iterations of useful work between two barriers, so
  synchronization and index overhead dominate, while `naive` collects a large uncounted L2
  assist. `ncu` on barrier stall percentage would settle it.
- Why does `tiled16` decay with N (855 → 658 → 630) while `tiled32` stays flat
  (863 → 866 → 836)? Consistent with L2 covering the lower-intensity kernel's traffic at
  small N and progressively failing to as the working set grows.
