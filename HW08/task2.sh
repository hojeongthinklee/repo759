#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task2_scaling
#SBATCH --output=task2_scaling.out
#SBATCH --error=task2_scaling.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=20

set -euo pipefail

# Compile
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Init CSV
echo "t,n,time_ms,first,last" > task2_times.csv

# Run for t = 1..20
for t in $(seq 1 20); do
  output=$(./task2 1024 "$t")

  first=$(echo "$output" | sed -n '1p')
  last=$(echo "$output" | sed -n '2p')
  time_ms=$(echo "$output" | sed -n '3p')

  echo "${t},1024,${time_ms},${first},${last}" >> task2_times.csv
done
