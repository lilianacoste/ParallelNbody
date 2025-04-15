#!/bin/bash
#SBATCH --job-name=nbody_benchmark
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark.out
#SBATCH --partition=Centaurus
export OMP_NUM_THREADS=8
./nbody 200 5000000 1000


