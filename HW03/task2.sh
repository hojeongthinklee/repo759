#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task2
#SBATCH --output=task2.out
#SBATCH --error=task2.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

./task2
