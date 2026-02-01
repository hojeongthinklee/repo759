#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=task3
#SBATCH --output=task3.out
#SBATCH --error=task3.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1



./task3
