#!/bin/bash

#SBATCH -p instruction
#SBATCH --job-name=task6
#SBATCH --output=task6.out
#SBATCH --error=task6.err
#SBATCH --cpus-per-task=1

./task6 6