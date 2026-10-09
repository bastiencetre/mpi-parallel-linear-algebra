"""Run the distributed conjugate gradient.

    mpirun -np 16 python scripts/cg_solve.py --rows-per-process 1000 --mode halo

``--rows-per-process`` keeps the work per process constant (weak scaling); ``--n`` fixes the
global size instead (strong scaling).
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description="Row-block distributed conjugate gradient")
group = parser.add_mutually_exclusive_group()
group.add_argument("--n", type=int, help="global number of unknowns")
group.add_argument("--rows-per-process", type=int, default=1000)
parser.add_argument("--mode", choices=["allgather", "halo"], default="halo")
parser.add_argument("--rtol", type=float, default=1e-8)
args = parser.parse_args()

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from mpi_linalg.cg import solve  # noqa: E402

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
n = args.n if args.n else args.rows_per_process * size

comm.Barrier()
t0 = time.perf_counter()
x_local, iterations, history, _ = solve(comm, n, args.mode, args.rtol)
comm.Barrier()
elapsed = time.perf_counter() - t0

if rank == 0:
    print(f"mode={args.mode} processes={size} n={n} iterations={iterations} "
          f"residual={history[-1]:.1e} time={elapsed:.4f}s throughput={n / elapsed / 1e3:.1f} Kvar/s")
    if n <= 20:
        print("x[:n] on rank 0:", np.round(x_local, 6))
