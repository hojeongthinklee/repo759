import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("task2.csv")

# plot
plt.figure(figsize=(6,4))
plt.plot(df["n"], df["time_ms"], marker='o')

# log-log scale
plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("n")
plt.ylabel("time (ms)")
plt.title("Task2 Scaling")

plt.grid(True)
plt.tight_layout()

plt.savefig("task2.pdf")