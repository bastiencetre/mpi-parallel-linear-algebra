"""Benchmark the ring matrix products.

    mpirun -np 8 python scripts/matprod.py --size 4096 --threads 2 --p2p --bcast --min

Each process runs its local products with ``--threads`` OpenBLAS threads, so the total number
of cores used is (number of processes) x (threads per process).
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description="Distributed C = A B on a ring of MPI processes")
parser.add_argument("--size", type=int, default=4096, help="matrix side n (multiple of the number of processes)")
parser.add_argument("--threads", type=int, default=1, help="OpenBLAS threads per process")
parser.add_argument("--p2p", action="store_true", help="ring with point-to-point Sendrecv_replace")
parser.add_argument("--bcast", action="store_true", help="ring with collective Bcast")
parser.add_argument("--min", action="store_true", help="also compute the minimum of every row of C")
args = parser.parse_args()
if not (args.p2p or args.bcast):
    parser.error("choose at least one of --p2p / --bcast")

# The thread count must be set before NumPy (and therefore OpenBLAS) is loaded.
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpi4py import MPI  # noqa: E402

from mpi_linalg.matmul import local_slices, ring_bcast, ring_p2p, row_minimums, spot_check  # noqa: E402

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
n = args.size
if rank == 0:
    print(f"n = {n}, {size} processes x {args.threads} threads")

for name, algorithm, enabled in (("ring P2P", ring_p2p, args.p2p), ("ring Bcast", ring_bcast, args.bcast)):
    if not enabled:
        continue
    A_r, B_r = local_slices(n, rank, size)
    comm.Barrier()
    t0 = time.perf_counter()
    C_r = algorithm(comm, A_r, B_r)
    comm.Barrier()
    elapsed = time.perf_counter() - t0

    error = comm.reduce(spot_check(C_r, rank, size, n), op=MPI.MAX, root=0)
    if rank == 0:
        print(f"{name:>10}: {elapsed:8.3f} s  {2 * n**3 / elapsed / 1e9:9.2f} Gflops  (spot-check error {error:.1e})")
    if args.min:
        mins = row_minimums(comm, C_r)
        if rank == 0:
            print(f"{'':>10}  row minimums: C[0] {mins[0]:.6e}, C[n/2] {mins[n // 2]:.6e}, C[n-1] {mins[-1]:.6e}")
