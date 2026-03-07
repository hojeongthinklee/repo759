import pandas as pd
import matplotlib.pyplot as plt

t = pd.read_csv("task1_thrust.csv")
c = pd.read_csv("task1_cub.csv")

plt.plot(t["n"], t["time_ms"], label="Thrust")
plt.plot(c["n"], c["time_ms"], label="CUB")

plt.xscale("log", base=2)
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("task1.pdf")