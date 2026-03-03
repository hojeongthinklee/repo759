#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw06_task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

OUT="task1.csv"
echo "n,time_ms" > "${OUT}"

for p in $(seq 5 11); do
  n=$((2**p))

  # choose reasonable n_tests (adjust if needed)
  if   [ "${p}" -le 7 ]; then n_tests=200
  elif [ "${p}" -le 9 ]; then n_tests=50
  else                       n_tests=10
  fi

  time_ms=$(./task1 "${n}" "${n_tests}")
  echo "${n},${time_ms}" >> "${OUT}"
done