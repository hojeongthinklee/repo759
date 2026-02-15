import csv
import math
import matplotlib.pyplot as plt

def read_csv(path):
    ns, ts = [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n = int(row["n"])
            t = float(row["time_ms"])
            ns.append(n)
            ts.append(t)
    return ns, ts

# Input CSV files (adjust if you used a different TPB2)
file1 = "task1_tpb1024.csv"
file2 = "task1_tpb256.csv"

n1, t1 = read_csv(file1)
n2, t2 = read_csv(file2)

plt.figure()
plt.plot(n1, t1, marker="o", label="threads_per_block = 1024")
plt.plot(n2, t2, marker="o", label="threads_per_block = 256")

plt.xlabel("n (matrix dimension)")
plt.ylabel("Time (ms)")
plt.title("Task 1 Scaling: CUDA MatMul (1D grid, no shared memory)")
plt.grid(True)
plt.legend()

# Optional: log-scale can make the growth easier to see for O(n^3)
# Uncomment if you want:
# plt.xscale("log", base=2)
# plt.yscale("log")

plt.tight_layout()
plt.savefig("task1.pdf")
print("Wrote task1.pdf")
