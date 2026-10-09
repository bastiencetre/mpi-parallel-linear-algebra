#!/bin/bash
# Strong-scaling campaign of the ring matrix products on N nodes (2 sockets x 8 cores each).
# Usage:  sbatch -N 4 -p cpu_prod cluster/matprod.slurm
#SBATCH --job-name=ring-matprod
#SBATCH --time=00:15:00
#SBATCH --exclusive

source /etc/profile
module load openblas/0.3.26/gcc-13.1.0-openmp
module load py-mpi4py/3.1.4/gcc-12.3.0-openmpi
cd "$SLURM_SUBMIT_DIR"

SIZE=16384
NODES=$SLURM_JOB_NUM_NODES

# Same 16 cores per node, split differently between MPI processes and OpenBLAS threads.
for PROC_PER_SOCKET in 1 2 4 8; do
  THREADS=$((8 / PROC_PER_SOCKET))
  NP=$((NODES * 2 * PROC_PER_SOCKET))
  echo "=== $NODES nodes, $PROC_PER_SOCKET process(es)/socket x $THREADS thread(s) ==="
  mpirun -np "$NP" -map-by "ppr:${PROC_PER_SOCKET}:socket:PE=${THREADS}" -rank-by core \
    python3 scripts/matprod.py --size "$SIZE" --threads "$THREADS" --p2p --bcast
done
