import csv
import matplotlib.pyplot as plt

t_vals = []
time_vals = []

with open("task3_times.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_vals.append(int(row["t"]))
        time_vals.append(float(row["time_ms"]))

plt.figure(figsize=(6,4))
plt.plot(t_vals, time_vals, marker="o")
plt.xlabel("t")
plt.ylabel("Time (ms)")
plt.title("HW8 Task3: merge sort time vs threads")
plt.grid(True)
plt.tight_layout()
plt.savefig("hw8_task3.pdf")
