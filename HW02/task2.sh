#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task2_conv
#SBATCH --output=task2.out
#SBATCH --error=task2.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1

set -euo pipefail

# Choose input sizes here
# n: image dimension (n x n)
# m: mask dimension (m x m), must be odd
n=4096
m=7

./task2 "${n}" "${m}"

