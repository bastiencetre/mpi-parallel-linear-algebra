#!/bin/bash
# Weak scaling of the distributed conjugate gradient: 1000 rows per process.
# Usage:  sbatch -N 1 -p cpu_prod cluster/cg_weak_scaling.slurm
#SBATCH --job-name=cg-weak
#SBATCH --time=00:10:00
#SBATCH --exclusive

source /etc/profile
module load openblas/0.3.26/gcc-13.1.0-openmp
module load py-mpi4py/3.1.4/gcc-12.3.0-openmpi
cd "$SLURM_SUBMIT_DIR"

for MODE in allgather halo; do
  for NP in 1 2 4 8 16 32; do
    mpirun -np "$NP" --oversubscribe python3 scripts/cg_solve.py --rows-per-process 1000 --mode "$MODE"
  done
done
