#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw06_task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

OUT="task2.csv"
echo "n,time_ms" > "${OUT}"

TPB=1024

for p in $(seq 10 16); do
  n=$((2**p))

  time_ms=$(./task2 "${n}" "${TPB}" | tail -n 1)
  echo "${n},${time_ms}" >> "${OUT}"
done