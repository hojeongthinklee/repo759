#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw04_task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

module load nvidia/cuda/13.0.0

R=128
TPB1=1024
TPB2=512

OUT1="task2_R${R}_tpb${TPB1}.csv"
OUT2="task2_R${R}_tpb${TPB2}.csv"

echo "n,time_ms" > "${OUT1}"
echo "n,time_ms" > "${OUT2}"

for p in $(seq 10 29); do
  n=$((2**p))

  t1=$(./task2 "${n}" "${R}" "${TPB1}" | tail -n 1)
  echo "${n},${t1}" >> "${OUT1}"

  t2=$(./task2 "${n}" "${R}" "${TPB2}" | tail -n 1)
  echo "${n},${t2}" >> "${OUT2}"
done

echo "Scaling experiment completed."
echo "Generated files:"
echo "  ${OUT1}"
echo "  ${OUT2}"
