"""Distributed dense matrix product C = A B on a 1D ring of P processes.

Data layout (n divisible by P, m = n / P), rank r owns:

* ``A_r`` -- rows    [r m, (r+1) m) of A, shape (m, n)
* ``B_r`` -- columns [r m, (r+1) m) of B, shape (n, m)
* ``C_r`` -- columns [r m, (r+1) m) of C, shape (n, m)

Since C[:, cols_r] = A B_r, rank r needs every row block of A. The two algorithms differ only
in how those blocks travel:

* ``ring_p2p``   -- blocks of A circulate around the ring with ``Sendrecv_replace``:
                    P steps, each rank talks only to its two neighbours.
* ``ring_bcast`` -- at step s, rank s broadcasts its block to everybody (``Bcast``):
                    a global synchronisation at every step.

Both perform P local products of size (m x n) @ (n x m), i.e. 2 n^3 / P flops per rank.
"""

from __future__ import annotations

import numpy as np

from .comm import MIN


def a_entries(rows, cols, n):
    """Deterministic test matrix A[i, j] = (i - j) / n (cheap to regenerate anywhere)."""
    return (np.asarray(rows)[:, None] - np.asarray(cols)[None, :]) / n


def b_entries(rows, cols, n):
    """Deterministic test matrix B[i, j] = (i + 2 j) / n."""
    return (np.asarray(rows)[:, None] + 2.0 * np.asarray(cols)[None, :]) / n


def local_slices(n, rank, size):
    if n % size:
        raise ValueError(f"matrix size {n} must be a multiple of the number of processes {size}")
    m = n // size
    own = np.arange(rank * m, (rank + 1) * m)
    every = np.arange(n)
    A_r = a_entries(own, every, n)
    B_r = b_entries(every, own, n)
    return A_r, B_r


def ring_p2p(comm, A_r, B_r):
    """Rotate the A blocks around the ring; A_r is back to its initial content at the end."""
    rank, size = comm.Get_rank(), comm.Get_size()
    m, n = A_r.shape
    C_r = np.empty((n, m), dtype=A_r.dtype)
    owner = rank                       # whose block of A we currently hold
    for _ in range(size):
        C_r[owner * m:(owner + 1) * m] = A_r @ B_r
        if size > 1:
            # Send to the left neighbour, receive from the right one:
            # after the exchange we hold the block of rank owner + 1.
            comm.Sendrecv_replace(A_r, dest=(rank - 1) % size, source=(rank + 1) % size)
        owner = (owner + 1) % size
    return C_r


def ring_bcast(comm, A_r, B_r):
    """At step s, rank s broadcasts its block of A to all the others."""
    rank, size = comm.Get_rank(), comm.Get_size()
    m, n = A_r.shape
    C_r = np.empty((n, m), dtype=A_r.dtype)
    block = np.empty_like(A_r)         # separate buffer, so A_r is never overwritten
    for step in range(size):
        if rank == step:
            block[...] = A_r
        comm.Bcast(block, root=step)
        C_r[step * m:(step + 1) * m] = block @ B_r
    return C_r


def row_minimums(comm, C_r, root=0):
    """Minimum of every row of C, gathered on ``root`` with a single MIN reduction."""
    local = np.ascontiguousarray(C_r.min(axis=1))
    result = np.empty_like(local) if comm.Get_rank() == root else None
    comm.Reduce(local, result, op=MIN, root=root)
    return result


def spot_check(C_r, rank, size, n, n_checks=8, seed=0):
    """Compare a few entries of the local block with direct dot products.

    Returns the largest error relative to the largest checked entry.
    """
    m = n // size
    rng = np.random.default_rng(seed + rank)
    rows = rng.integers(0, n, n_checks)
    cols = rng.integers(0, m, n_checks)
    every = np.arange(n)
    exact = np.array([a_entries([i], every, n)[0] @ b_entries(every, [rank * m + j], n)[:, 0]
                      for i, j in zip(rows, cols)])
    return float(np.abs(C_r[rows, cols] - exact).max() / np.abs(exact).max())
