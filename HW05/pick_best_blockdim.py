import glob
import re

def parse_task1(path):
    with open(path, "r") as f:
        lines = [x.strip() for x in f if x.strip() != ""]
    if len(lines) < 9:
        raise ValueError(f"Unexpected output lines({len(lines)}): {path}")
    times = [float(lines[2]), float(lines[5]), float(lines[8])]
    return {"int_ms": times[0], "float_ms": times[1], "double_ms": times[2]}

rows = []
for p in glob.glob("results/blockdim/*.txt"):
    m = re.search(r"bd(\d+)\.txt$", p)
    if not m:
        continue
    bd = int(m.group(1))
    t = parse_task1(p)
    rows.append((bd, t["int_ms"], t["float_ms"], t["double_ms"]))

if not rows:
    raise SystemExit("No results found under results/blockdim/*.txt")

def best(col_idx):
    return min(rows, key=lambda r: r[col_idx])

b_int = best(1)
b_float = best(2)
b_double = best(3)

print(f"best int:    block_dim={b_int[0]}  time_ms={b_int[1]:.6f}")
print(f"best float:  block_dim={b_float[0]}  time_ms={b_float[2]:.6f}")
print(f"best double: block_dim={b_double[0]}  time_ms={b_double[3]:.6f}")

print("\nAll:")
for r in sorted(rows):
    print(f"bd={r[0]:>4}  int={r[1]:>10.6f}  float={r[2]:>10.6f}  double={r[3]:>10.6f}")