#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw07_task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

OUT_THRUST="task1_thrust.csv"
OUT_CUB="task1_cub.csv"

echo "n,time_ms" > ${OUT_THRUST}
echo "n,time_ms" > ${OUT_CUB}

for p in $(seq 10 20); do
    n=$((2**p))

    echo "Running n=$n"

    thrust_time=$(./task1_thrust ${n} | tail -n 1)
    cub_time=$(./task1_cub ${n} | tail -n 1)

    echo "${n},${thrust_time}" >> ${OUT_THRUST}
    echo "${n},${cub_time}" >> ${OUT_CUB}
done