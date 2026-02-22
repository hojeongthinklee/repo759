#!/bin/bash
#SBATCH -p instruction
#SBATCH --job-name=hw05_task1_b
#SBATCH --output=task1_b_%j.out
#SBATCH --error=task1_b_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

# Assumes ./task1 already exists and is executable.

BD1=32
BD2=16

OUT1="task1_times_bd${BD1}.csv"
OUT2="task1_times_bd${BD2}.csv"

echo "n,ms_int,ms_float,ms_double" > "${OUT1}"
echo "n,ms_int,ms_float,ms_double" > "${OUT2}"

for p in $(seq 5 14); do
  n=$((2**p))

  # Lines 3,6,9 are ms for int,float,double respectively
  # bd1
  ms_int=$(./task1 "${n}" "${BD1}" | sed -n '3p')
  ms_float=$(./task1 "${n}" "${BD1}" | sed -n '6p')
  ms_double=$(./task1 "${n}" "${BD1}" | sed -n '9p')
  echo "${n},${ms_int},${ms_float},${ms_double}" >> "${OUT1}"

  # bd2
  ms_int2=$(./task1 "${n}" "${BD2}" | sed -n '3p')
  ms_float2=$(./task1 "${n}" "${BD2}" | sed -n '6p')
  ms_double2=$(./task1 "${n}" "${BD2}" | sed -n '9p')
  echo "${n},${ms_int2},${ms_float2},${ms_double2}" >> "${OUT2}"
done