#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw05_task1_c
#SBATCH --output=task1_c_%j.out
#SBATCH --error=task1_c_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

# Assumes ./task1 already exists and is executable.

N=$((2**14))
REPS=5
BDS=(8 16 32)

OUT="task1_best_blockdim_n${N}.csv"
echo "block_dim,avg_ms_float" > "${OUT}"

best_bd=-1
best_t=1e30

for bd in "${BDS[@]}"; do
  sum=0.0
  for r in $(seq 1 "${REPS}"); do
    t=$(./task1 "${N}" "${bd}" | sed -n '6p')  # float ms
    sum=$(awk -v s="${sum}" -v x="${t}" 'BEGIN{printf "%.6f", s+x}')
  done
  avg=$(awk -v s="${sum}" -v k="${REPS}" 'BEGIN{printf "%.6f", s/k}')
  echo "${bd},${avg}" >> "${OUT}"

  is_better=$(awk -v a="${avg}" -v b="${best_t}" 'BEGIN{print (a<b)?1:0}')
  if [[ "${is_better}" -eq 1 ]]; then
    best_t="${avg}"
    best_bd="${bd}"
  fi
done

echo "${best_bd} ${best_t}"