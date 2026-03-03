import csv
import matplotlib.pyplot as plt

ns = []
ts = []

with open("task2.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ns.append(int(row["n"]))
        ts.append(float(row["time_ms"]))

plt.figure()
plt.plot(ns, ts, marker="o")
plt.xlabel("n")
plt.ylabel("Scan time (ms)")
plt.xscale("log", base=2)
plt.tight_layout()
plt.savefig("task2.pdf")