"""Conjugate gradient with a row-block distribution of a sparse SPD matrix.

Test problem: the n x n tridiagonal matrix with 4 on the diagonal and -1 next to it (a 1D
discrete Laplacian shifted to be well conditioned), with a Dirichlet condition on unknown 0
(row and column 0 replaced by the identity), and b = 1.

Rank r owns rows [lo, hi) of A, stored as a sparse block: it is built directly from the
formula, never from the global matrix, so memory per rank is O(n / P).

The only communications are:

* the matrix-vector product, which needs entries of the search direction owned by others:
  - ``allgather``: every rank rebuilds the whole vector (O(n) data per rank and per iteration);
  - ``halo``: a tridiagonal row only needs one entry from each neighbour, exchanged with two
    ``Sendrecv`` calls (O(1) data per rank, no global synchronisation);
* two global ``Allreduce`` per iteration, for (p, Ap) and (r, r).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .comm import PROC_NULL, SUM


def row_range(n, rank, size):
    if n % size:
        raise ValueError(f"n = {n} must be a multiple of the number of processes {size}")
    m = n // size
    return rank * m, (rank + 1) * m


def local_matrix(n, lo, hi):
    """Rows [lo, hi) of the test matrix, with global column indices."""
    i = np.arange(lo, hi)
    diag = np.where(i == 0, 1.0, 4.0)
    rows, cols, vals = [i], [i], [diag]
    lower = i[i >= 2]                       # column 0 is eliminated by the Dirichlet condition
    rows.append(lower), cols.append(lower - 1), vals.append(-np.ones(lower.size))
    upper = i[(i >= 1) & (i <= n - 2)]      # row 0 is the identity row
    rows.append(upper), cols.append(upper + 1), vals.append(-np.ones(upper.size))
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    return sp.csr_matrix((vals, (rows - lo, cols)), shape=(hi - lo, n))


class RowBlockOperator:
    """Distributed y = A x for row-block vectors, with the chosen communication scheme."""

    def __init__(self, comm, n, mode="halo"):
        self.comm, self.n, self.mode = comm, n, mode
        rank, size = comm.Get_rank(), comm.Get_size()
        self.lo, self.hi = row_range(n, rank, size)
        A = local_matrix(n, self.lo, self.hi)
        if mode == "allgather":
            self.A = A
            self._full = np.empty(n)
        elif mode == "halo":
            # Re-index columns lo-1 .. hi as 0 .. m+1: [left ghost | own entries | right ghost].
            coo = A.tocoo()
            m = self.hi - self.lo
            self.A = sp.csr_matrix((coo.data, (coo.row, coo.col - (self.lo - 1))), shape=(m, m + 2))
            self._ext = np.zeros(m + 2)
            self.left = rank - 1 if rank > 0 else PROC_NULL
            self.right = rank + 1 if rank < size - 1 else PROC_NULL
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def matvec(self, x_local):
        if self.mode == "allgather":
            self.comm.Allgather(x_local, self._full)
            return self.A @ self._full
        ext = self._ext
        ext[1:-1] = x_local
        # Shift right: send my last entry to the right, receive the left neighbour's last one.
        self.comm.Sendrecv(x_local[-1:], dest=self.right, recvbuf=ext[:1], source=self.left)
        # Shift left: send my first entry to the left, receive the right neighbour's first one.
        self.comm.Sendrecv(x_local[:1], dest=self.left, recvbuf=ext[-1:], source=self.right)
        return self.A @ ext


def global_dot(comm, a, b):
    local = np.array([np.dot(a, b)])
    total = np.empty(1)
    comm.Allreduce(local, total, op=SUM)
    return float(total[0])


def conjugate_gradient(comm, op: RowBlockOperator, b_local, rtol=1e-8, maxiter=1000):
    """Standard CG (Hestenes & Stiefel). Returns (x_local, iterations, relative residuals)."""
    x = np.zeros_like(b_local)
    r = b_local.copy()
    p = r.copy()
    rr = global_dot(comm, r, r)
    r0 = np.sqrt(rr)
    history = [1.0]
    for it in range(1, maxiter + 1):
        Ap = op.matvec(p)
        alpha = rr / global_dot(comm, p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = global_dot(comm, r, r)
        history.append(np.sqrt(rr_new) / r0)
        if history[-1] <= rtol:
            return x, it, history
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, maxiter, history


def solve(comm, n, mode="halo", rtol=1e-8, maxiter=1000):
    """Assemble the local block, run CG, return (x_local, iterations, history, operator)."""
    op = RowBlockOperator(comm, n, mode)
    b_local = np.ones(op.hi - op.lo)
    x, iterations, history = conjugate_gradient(comm, op, b_local, rtol, maxiter)
    return x, iterations, history, op
