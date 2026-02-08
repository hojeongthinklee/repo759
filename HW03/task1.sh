#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task1_factorial
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

./task1
