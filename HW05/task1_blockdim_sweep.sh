#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw5_p1c
#SBATCH --output=blockdim_%a.out
#SBATCH --error=blockdim_%a.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

set -euo pipefail

module load nvidia/cuda/13.0

n=16384
candidates=(8 16 32 24 12)
block_dim="${candidates[$SLURM_ARRAY_TASK_ID]}"

mkdir -p results/blockdim
./task1 "$n" "$block_dim" > "results/blockdim/n${n}_bd${block_dim}.txt"