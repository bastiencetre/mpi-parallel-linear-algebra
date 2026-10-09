"""MPI constants, with an in-process stand-in for testing without an MPI installation.

``ThreadComm`` implements the handful of mpi4py buffer methods used in this package
(Barrier, Bcast, Reduce, Allreduce, Allgather, Sendrecv, Sendrecv_replace) for ranks that
run as threads in one Python process. The algorithms are therefore tested with exactly the
code that runs on the cluster, only the transport changes.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    from mpi4py import MPI

    SUM, MIN, PROC_NULL = MPI.SUM, MPI.MIN, MPI.PROC_NULL
except ImportError:  # local development / CI without MPI
    MPI = None
    SUM, MIN, PROC_NULL = "sum", "min", -1


def _reduce(arrays, op):
    # mpi4py Op objects are not hashable, hence comparisons instead of a dict lookup.
    if op == SUM:
        return np.sum(arrays, axis=0)
    if op == MIN:
        return np.min(arrays, axis=0)
    raise NotImplementedError(op)


class _World:
    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size


class ThreadComm:
    def __init__(self, world: _World, rank: int):
        self._world, self._rank = world, rank

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._world.size

    def Barrier(self):
        self._world.barrier.wait()

    def _all_exchange(self, value):
        """Every rank posts a value and gets the list of all posted values."""
        w = self._world
        w.slots[self._rank] = value
        w.barrier.wait()
        values = list(w.slots)
        w.barrier.wait()
        return values

    def Bcast(self, buf, root=0):
        values = self._all_exchange(np.copy(buf) if self._rank == root else None)
        buf[...] = values[root]

    def Reduce(self, sendbuf, recvbuf, op=SUM, root=0):
        values = self._all_exchange(np.copy(sendbuf))
        if self._rank == root:
            recvbuf[...] = _reduce(values, op)

    def Allreduce(self, sendbuf, recvbuf, op=SUM):
        recvbuf[...] = _reduce(self._all_exchange(np.copy(sendbuf)), op)

    def Allgather(self, sendbuf, recvbuf):
        recvbuf[...] = np.concatenate(self._all_exchange(np.copy(sendbuf)))

    def Sendrecv(self, sendbuf, dest, recvbuf=None, source=PROC_NULL):
        values = self._all_exchange((dest, np.copy(sendbuf)))
        if source != PROC_NULL:
            target, data = values[source]
            assert target == self._rank, "mismatched Sendrecv pair"
            recvbuf[...] = data

    def Sendrecv_replace(self, buf, dest, source):
        self.Sendrecv(buf, dest, buf, source)


def run_spmd(size: int, program, *args):
    """Run ``program(comm, *args)`` on ``size`` thread-ranks and return the list of results."""
    world = _World(size)
    results, errors = [None] * size, []

    def target(rank):
        try:
            results[rank] = program(ThreadComm(world, rank), *args)
        except BaseException as exc:  # make failures visible and unblock the others
            errors.append(exc)
            world.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results
