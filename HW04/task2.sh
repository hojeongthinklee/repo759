#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw04_task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail


# ----------------------------------------
# Experiment settings (as required by HW)
# ----------------------------------------
R=128
TPB1=1024
TPB2=256   # Second TPB value for overlay plot; you may change to 512 if desired

OUT1="task2_R${R}_tpb${TPB1}.csv"
OUT2="task2_R${R}_tpb${TPB2}.csv"

# CSV headers
echo "n,time_ms" > "${OUT1}"
echo "n,time_ms" > "${OUT2}"

# ----------------------------------------
# Scaling experiment
# n = 2^10, 2^11, ..., 2^29
# Note: stencil.cuh assumes threads_per_block >= 2*R+1 (=257 when R=128)
# So TPB2 must be >= 257. TPB2=256 would violate the assumption.
# We therefore set TPB2=512 by default below if R=128.
# ----------------------------------------

if [ "${TPB2}" -lt $((2*R + 1)) ]; then
  echo "ERROR: TPB2 (${TPB2}) < 2*R+1 ($((2*R + 1))). Choose TPB2 >= $((2*R + 1))." >&2
  exit 1
fi

for p in $(seq 10 29); do
  n=$((2**p))

  # Program prints:
  #   line 1: last output element
  #   line 2: execution time (ms)
  t1=$(./task2 "${n}" "${R}" "${TPB1}" | tail -n 1)
  echo "${n},${t1}" >> "${OUT1}"

  t2=$(./task2 "${n}" "${R}" "${TPB2}" | tail -n 1)
  echo "${n},${t2}" >> "${OUT2}"
done

echo "Scaling experiment completed."
echo "Generated files:"
echo "  ${OUT1}"
echo "  ${OUT2}"
