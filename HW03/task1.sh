#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task1_factorial
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load cuda

nvcc task1.cu -O3 -std=c++17 -Xcompiler -Wall -Xptxas -O3 -o task1

./task1
