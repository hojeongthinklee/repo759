#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw05_task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail


N=1024
BLOCK_DIM=32

echo "Running: ./task1 ${N} ${BLOCK_DIM}"
./task1 ${N} ${BLOCK_DIM}