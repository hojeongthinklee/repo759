#!/bin/bash
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cuda-memcheck ./task2 1024 1024