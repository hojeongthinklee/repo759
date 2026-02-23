import pandas as pd
import matplotlib.pyplot as plt

# File names (modify if needed)
file1 = "task2_tpb1024.csv"
file2 = "task2_tpb512.csv"

# Read CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Create plot
plt.figure(figsize=(8, 6))

plt.plot(df1["n"], df1["time_ms"], marker='o', label="TPB=1024")
plt.plot(df2["n"], df2["time_ms"], marker='s', label="TPB=512")

plt.xscale("log", base=2)
plt.xlabel("N (array size)")
plt.ylabel("Time (ms)")
plt.title("CUDA Reduction Scaling (Kernel 4)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("task2_scaling_plot_warmup.png", dpi=300)
# plt.show()