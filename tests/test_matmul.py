import numpy as np
import pytest

from mpi_linalg.comm import run_spmd
from mpi_linalg.matmul import a_entries, b_entries, local_slices, ring_bcast, ring_p2p, row_minimums, spot_check

N = 24


def full_product(n):
    every = np.arange(n)
    return a_entries(every, every, n) @ b_entries(every, every, n)


def program(comm, algorithm):
    rank, size = comm.Get_rank(), comm.Get_size()
    A_r, B_r = local_slices(N, rank, size)
    A_before = A_r.copy()
    C_r = algorithm(comm, A_r, B_r)
    return C_r, np.array_equal(A_r, A_before), row_minimums(comm, C_r), spot_check(C_r, rank, size, N)


@pytest.mark.parametrize("algorithm", [ring_p2p, ring_bcast])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 6])
def test_ring_product_matches_numpy(algorithm, size):
    results = run_spmd(size, program, algorithm)
    C = np.hstack([r[0] for r in results])
    expected = full_product(N)
    np.testing.assert_allclose(C, expected, atol=1e-12)
    assert all(r[1] for r in results), "the A slices must come back to their initial state"
    np.testing.assert_allclose(results[0][2], expected.min(axis=1))
    assert all(r[3] < 1e-12 for r in results)


def test_size_must_divide():
    with pytest.raises(ValueError):
        local_slices(10, 0, 3)
