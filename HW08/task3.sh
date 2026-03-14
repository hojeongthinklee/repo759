#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task3_scaling
#SBATCH --output=task3_scaling.out
#SBATCH --error=task3_scaling.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=20

set -euo pipefail

# Compile
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

echo "t,n,time_ms,first,last" > task3_times.csv

# fixed parameters
n=1048576
threshold=2048

for t in $(seq 1 20); do

  output=$(./task3 "$n" "$t" "$threshold")

  first=$(echo "$output" | sed -n '1p')
  last=$(echo "$output" | sed -n '2p')
  time_ms=$(echo "$output" | sed -n '3p')

  echo "${t},${n},${time_ms},${first},${last}" >> task3_times.csv

done
