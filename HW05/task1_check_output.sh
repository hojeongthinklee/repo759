#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw05_task1_check
#SBATCH --output=task1_check_%j.out
#SBATCH --error=task1_check_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

module load nvidia/cuda/13.0

# ----------------------------------------
# Build (exactly as assignment specifies)
# ----------------------------------------
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# ----------------------------------------
# Test settings (pick a small n and a typical block_dim)
# You can change these to anything you want.
# ----------------------------------------
N=1024
BD=32

# Save raw output for inspection
RAW_OUT="task1_raw_output_n${N}_bd${BD}.txt"

# Run once
./task1 "${N}" "${BD}" | tee "${RAW_OUT}" >/dev/null

# ----------------------------------------
# Validate output format:
# Expected 9 lines total:
#   1: int    C[0]
#   2: int    C[last]
#   3: int    time_ms
#   4: float  C[0]
#   5: float  C[last]
#   6: float  time_ms
#   7: double C[0]
#   8: double C[last]
#   9: double time_ms
# ----------------------------------------

lines=$(wc -l < "${RAW_OUT}")
if [[ "${lines}" -ne 9 ]]; then
  echo "FAIL: Expected 9 lines, got ${lines} lines."
  echo "Raw output saved to: ${RAW_OUT}"
  exit 1
fi

# Check each line looks like a number (int or float)
# This regex accepts: 123, -123, 1.23, -1.23, 1e-3, -1E+6, etc.
num_re='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'

ok=1
for i in $(seq 1 9); do
  val=$(sed -n "${i}p" "${RAW_OUT}" | tr -d '[:space:]')
  if [[ ! "${val}" =~ ${num_re} ]]; then
    echo "FAIL: Line ${i} is not numeric: '${val}'"
    ok=0
  fi
done

if [[ "${ok}" -ne 1 ]]; then
  echo "Raw output saved to: ${RAW_OUT}"
  exit 1
fi

# Check that timing lines (3,6,9) are > 0
t3=$(sed -n '3p' "${RAW_OUT}")
t6=$(sed -n '6p' "${RAW_OUT}")
t9=$(sed -n '9p' "${RAW_OUT}")

for t in "${t3}" "${t6}" "${t9}"; do
  gt0=$(awk -v x="${t}" 'BEGIN{print (x>0.0)?1:0}')
  if [[ "${gt0}" -ne 1 ]]; then
    echo "FAIL: Timing value not > 0: ${t}"
    echo "Raw output saved to: ${RAW_OUT}"
    exit 1
  fi
done

echo "PASS: Output format looks correct."
echo "Raw output saved to: ${RAW_OUT}"
echo "Times (ms): int=${t3}, float=${t6}, double=${t9}"