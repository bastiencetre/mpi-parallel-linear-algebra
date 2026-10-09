import numpy as np
import pytest
import scipy.sparse.linalg as spla

from mpi_linalg.cg import local_matrix, solve
from mpi_linalg.comm import run_spmd


def reference(n):
    A = local_matrix(n, 0, n).tocsc()
    return spla.spsolve(A, np.ones(n))


def test_matrix_is_symmetric_with_dirichlet_row():
    A = local_matrix(10, 0, 10).toarray()
    assert np.allclose(A, A.T)
    assert A[0, 0] == 1 and np.count_nonzero(A[0]) == 1 and np.count_nonzero(A[:, 0]) == 1
    assert np.all(np.linalg.eigvalsh(A) > 0)


def test_blocks_assemble_to_the_global_matrix():
    full = local_matrix(12, 0, 12).toarray()
    blocks = [local_matrix(12, lo, lo + 4).toarray() for lo in (0, 4, 8)]
    assert np.array_equal(np.vstack(blocks), full)


@pytest.mark.parametrize("mode", ["allgather", "halo"])
@pytest.mark.parametrize("size", [1, 2, 4, 5])
def test_distributed_cg_matches_direct_solver(mode, size):
    n = 60
    results = run_spmd(size, lambda comm: solve(comm, n, mode, rtol=1e-12))
    x = np.concatenate([r[0] for r in results])
    np.testing.assert_allclose(x, reference(n), rtol=1e-9)
    iterations = {r[1] for r in results}
    assert len(iterations) == 1, "all ranks must stop at the same iteration"
