#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw04_task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

# ----------------------------------------
# Experiment settings
# ----------------------------------------

# Threads per block required by assignment
TPB1=1024

# Second threads-per-block value for overlay comparison
TPB2=256

# Output CSV files
OUT1="task1_tpb${TPB1}.csv"
OUT2="task1_tpb${TPB2}.csv"

# Write CSV headers
echo "n,time_ms" > "${OUT1}"
echo "n,time_ms" > "${OUT2}"

# ----------------------------------------
# Scaling experiment
# n = 2^5, 2^6, ..., 2^14
# ----------------------------------------

for p in $(seq 5 14); do
    n=$((2**p))

    # Run task1 with TPB1
    # Program prints:
    #   line 1: last matrix element
    #   line 2: execution time (ms)
    t1=$(./task1 "${n}" "${TPB1}" | tail -n 1)
    echo "${n},${t1}" >> "${OUT1}"

    # Run task1 with TPB2
    t2=$(./task1 "${n}" "${TPB2}" | tail -n 1)
    echo "${n},${t2}" >> "${OUT2}"

done

echo "Scaling experiment completed."
echo "Generated files:"
echo "  ${OUT1}"
echo "  ${OUT2}"
