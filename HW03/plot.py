import matplotlib.pyplot as plt

# Data copied from task3_times.csv (cat output)
n = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456
]

time_512 = [
    1.129, 0.685, 0.806, 0.895, 0.649, 0.811, 0.738, 0.756,
    0.719, 0.709, 0.793, 0.769, 0.590, 0.804, 1.190,
    1.946, 3.391, 6.104, 11.609
]

time_16 = [
    0.724, 0.678, 0.783, 0.713, 0.757, 0.707, 0.764, 0.704,
    0.784, 0.768, 0.744, 0.703, 0.847, 1.077, 1.639,
    2.502, 4.160, 6.928, 13.821
]

plt.figure(figsize=(6, 4))
plt.plot(n, time_512, marker='o', label='512 threads/block')
plt.plot(n, time_16, marker='o', label='16 threads/block')

plt.xscale('log', base=2)
plt.xlabel('n (log2 scale)')
plt.ylabel('Kernel time (ms)')
plt.title('vscale time vs n')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('task3.pdf')
plt.show()
