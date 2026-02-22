import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("task1_times_bd32.csv")
df2 = pd.read_csv("task1_times_bd16.csv")

plt.figure()
plt.plot(df1["n"], df1["ms_float"], marker="o", label="block_dim=32 (float)")
plt.plot(df2["n"], df2["ms_float"], marker="o", label="block_dim=16 (float)")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("HW05 Task1: Tiled MatMul Runtime vs n (float)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("task1_runtime_vs_n_float.png", dpi=200)