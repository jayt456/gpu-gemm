import csv, collections
import matplotlib.pyplot as plt

data = collections.defaultdict(lambda: ([], []))
with open("results/results.csv") as f:
    for r in csv.DictReader(f):
        data[r["kernel"]][0].append(int(r["N"]))
        data[r["kernel"]][1].append(float(r["gflops"]))

order = ["cpu", "naive_uncoal", "naive", "tiled32", "reg1d", "cublas"]
labels = {"cpu": "CPU (i-k-j)", "naive_uncoal": "Naive, uncoalesced",
          "naive": "Naive (coalesced)", "tiled32": "Shared-mem tiled (32)",
          "reg1d": "Register tiled (8x)", "cublas": "cuBLAS"}

plt.figure(figsize=(9, 5.5))
for k in order:
    if k in data:
        xs, ys = data[k]
        plt.plot(xs, ys, marker="o", label=labels.get(k, k))

plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("Matrix size N"); plt.ylabel("GFLOP/s (higher is better)")
plt.title("SGEMM performance by implementation")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("results/benchmark.png", dpi=160)
print("wrote results/benchmark.png")