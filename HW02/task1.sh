#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task1_scaling
#SBATCH --output=task1_scaling.out
#SBATCH --error=task1_scaling.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1

set -euo pipefail

# Compile task1
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# Initialize CSV output
echo "p,n,time_ms,first,last" > task1_times.csv

# Run task1 for n = 2^10 to 2^30
for p in $(seq 10 30); do
  n=$((1<<p))

  output=$(./task1 "$n")

  time_ms=$(echo "$output" | sed -n '1p')
  first=$(echo "$output" | sed -n '2p')
  last=$(echo "$output" | sed -n '3p')

  echo "${p},${n},${time_ms},${first},${last}" >> task1_times.csv
done
