"""results/results.csv -> scaling chart + N=4096 summary bars."""
import csv, collections
import matplotlib.pyplot as plt

order = ["cpu", "naive_uncoal", "naive", "tiled8", "tiled16", "tiled32",
         "reg1d", "reg2d", "cublas"]
labels = {"cpu": "CPU (i-k-j, -O3)",
          "naive_uncoal": "Naive, uncoalesced",
          "naive": "Naive (coalesced)",
          "tiled8": "Shared-mem tiled, T=8",
          "tiled16": "Shared-mem tiled, T=16",
          "tiled32": "Shared-mem tiled, T=32",
          "reg1d": "1-D register tiled (8/thread)",
          "reg2d": "2-D register tiled (8x8/thread)",
          "cublas": "cuBLAS"}

series = collections.defaultdict(lambda: ([], []))
with open("results/results.csv") as f:
    for r in csv.DictReader(f):
        series[r["kernel"]][0].append(int(r["N"]))
        series[r["kernel"]][1].append(float(r["gflops"]))

# --- scaling: GFLOP/s vs N ---
plt.figure(figsize=(9, 5.5))
for k in order:
    if k in series:
        xs, ys = series[k]
        style = dict(marker="o", lw=1.6)
        if k == "cublas":
            style.update(color="black", ls="--")
        elif k == "reg2d":
            style.update(lw=2.6)
        plt.plot(xs, ys, label=labels.get(k, k), **style)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Matrix size N")
plt.ylabel("GFLOP/s (higher is better)")
plt.title("SGEMM throughput by implementation")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/benchmark.png", dpi=160)
print("wrote results/benchmark.png")

# --- summary bars at the largest size ---
big = max(n for k in series for n in series[k][0])
names, vals, cols = [], [], []
shade = {"reg2d": "#1f4e79", "cublas": "#333333"}
for k in order:
    if k in series and big in series[k][0]:
        names.append(labels.get(k, k))
        vals.append(series[k][1][series[k][0].index(big)])
        cols.append(shade.get(k, "#b0b0b0"))

plt.figure(figsize=(8, 4.2))
bars = plt.barh(names, vals, color=cols)
plt.gca().invert_yaxis()
for b, v in zip(bars, vals):
    plt.text(v * 1.02, b.get_y() + b.get_height() / 2, f"{v:,.0f}",
             va="center", fontsize=8)
plt.xlabel("GFLOP/s")
plt.title(f"SGEMM at N={big}")
plt.xlim(0, max(vals) * 1.15)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("results/summary.png", dpi=160)
print("wrote results/summary.png")
