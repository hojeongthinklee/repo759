#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task3_scaling
#SBATCH --output=task3_c.out
#SBATCH --error=task3_c.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "p,n,time_ms_512,time_ms_16" > task3_times.csv

for p in $(seq 10 29); do
  n=$((1<<p))

  out512=$(./task3 "$n" 512)
  t512=$(echo "$out512" | sed -n '1p')

  out16=$(./task3 "$n" 16)
  t16=$(echo "$out16" | sed -n '1p')

  echo "${p},${n},${t512},${t16}" >> task3_times.csv
done