import csv
import matplotlib.pyplot as plt

t_vals = []
time_vals = []

with open("task2_times.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_vals.append(int(row["t"]))
        time_vals.append(float(row["time_ms"]))

plt.figure(figsize=(6, 4))
plt.plot(t_vals, time_vals, marker="o")
plt.xlabel("t")
plt.ylabel("Time (ms)")
plt.title("HW8 Task2: convolution time vs. threads")
plt.grid(True)
plt.tight_layout()
plt.savefig("hw8_task2.pdf")
