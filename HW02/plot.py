import csv
import matplotlib.pyplot as plt

# Read data
p_vals = []
n_vals = []
t_vals = []

with open("task1_times.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p_vals.append(int(row["p"]))
        n_vals.append(int(row["n"]))
        t_vals.append(float(row["time_ms"]))

# Split data
n_main = [n for p, n in zip(p_vals, n_vals) if p <= 28]
t_main = [t for p, t in zip(p_vals, t_vals) if p <= 28]

n_out = [n for p, n in zip(p_vals, n_vals) if p == 29]
t_out = [t for p, t in zip(p_vals, t_vals) if p == 29]

# Create broken y-axis plot
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(6, 6),
    gridspec_kw={"height_ratios": [1, 3]}
)

# Top axis: outlier
ax_top.scatter(n_out, t_out)
ax_top.set_ylim(15000, 21000)
ax_top.set_ylabel("Time (ms)")

# Bottom axis: main scaling region
ax_bot.plot(n_main, t_main, marker="o")
ax_bot.set_xscale("log", base=2)
ax_bot.set_ylim(0, 700)
ax_bot.set_xlabel("n (array length)")
ax_bot.set_ylabel("Time (ms)")

# Draw diagonal break marks
d = 0.015
kwargs = dict(color="k", clip_on=False)

ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)

ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

plt.suptitle("Scaling Analysis of Inclusive Scan (Broken Y-axis)")
plt.tight_layout()
plt.savefig("task1.pdf")
