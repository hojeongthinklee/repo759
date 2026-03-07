#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw07_task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail


OUT="task2.csv"
echo "n,time_ms" > $OUT

for p in $(seq 10 20); do
    n=$((2**p))

    echo "Running n=$n"

    ./task2 $n

    time_ms=$(./task2 $n | tail -n 1)

    echo "$n,$time_ms" >> task2.csv
done