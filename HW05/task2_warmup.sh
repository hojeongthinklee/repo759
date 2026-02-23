#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw05_task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

TPB1=1024
TPB2=512

OUT1="task2_tpb${TPB1}.csv"
OUT2="task2_tpb${TPB2}.csv"

echo "n,time_ms" > "${OUT1}"
echo "n,time_ms" > "${OUT2}"

# Global warm-up (initialize CUDA context once)
./task2 1024 1024 > /dev/null

for p in $(seq 10 30); do
  n=$((2**p))

  # Warm-up for this configuration
  ./task2 "${n}" "${TPB1}" > /dev/null
  t1=$(./task2 "${n}" "${TPB1}" | tail -n 1)
  echo "${n},${t1}" >> "${OUT1}"

  ./task2 "${n}" "${TPB2}" > /dev/null
  t2=$(./task2 "${n}" "${TPB2}" | tail -n 1)
  echo "${n},${t2}" >> "${OUT2}"
done