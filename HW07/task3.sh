#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw07_task3
#SBATCH --output=task3_%j.out
#SBATCH --error=task3_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

set -euo pipefail

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3