import csv
import matplotlib.pyplot as plt

def read_csv(path):
    ns, ts = [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ns.append(int(row["n"]))
            ts.append(float(row["time_ms"]))
    return ns, ts

# Update these if your filenames differ
file1 = "task2_R128_tpb1024.csv"
file2 = "task2_R128_tpb512.csv"

n1, t1 = read_csv(file1)
n2, t2 = read_csv(file2)

plt.figure()
plt.plot(n1, t1, marker="o", label="threads_per_block = 1024, R = 128")
plt.plot(n2, t2, marker="o", label="threads_per_block = 512, R = 128")

plt.xlabel("n (image length)")
plt.ylabel("Time (ms)")
plt.title("Task 2 Scaling: CUDA 1D Stencil (dynamic shared memory)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("task2.pdf")
print("Wrote task2.pdf")
