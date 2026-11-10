import cpo3
import numpy as np
import pytest
from scipy import sparse


@pytest.mark.parametrize("fixed_variable", [False, True])
def test_metadata_and_original_coordinates(fixed_variable: bool) -> None:
    p = cpo3.Problem(
        P=sparse.csc_matrix(
            [[2, 1, 0], [1, 2, 0], [0, 0, 1]] if fixed_variable else [[2, 1], [1, 2]]
        ),
        c=[-1, -1, 0] if fixed_variable else [-1, -1],
        A=sparse.csc_matrix(
            [[1, 1, 0], [-1, 1, 0], [0, 0, 1]] if fixed_variable else [[1, 1], [-1, 1]]
        ),
        b=[4, 1, 2] if fixed_variable else [4, 1],
        cones=[cpo3.NonnegativeCone(2)]
        + ([cpo3.ZeroCone(1)] if fixed_variable else []),
    )
    s = cpo3.solve(p, settings=cpo3.Settings(threads=1))
    assert s.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(
        s.x, [1 / 3, 1 / 3] + ([2] if fixed_variable else []), rtol=0, atol=1e-6
    )
    assert s.info.presolve_rows_removed == int(fixed_variable)
    assert s.info.presolve_columns_removed == int(fixed_variable)
    assert s.info.factorizations > 0
    assert s.info.solve_time >= s.info.presolve_time


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("primal", [False, True], ids=["dual", "primal"])
def test_infeasibility_certificates(presolve: bool, primal: bool) -> None:
    p = cpo3.Problem(
        P=None,
        c=[1] if primal else [-1],
        A=sparse.csc_matrix([[1], [-1]] if primal else [[-1]]),
        b=[-1, 0] if primal else [0],
        cones=[cpo3.NonnegativeCone(2 if primal else 1)],
    )
    s = cpo3.solve(p, settings=cpo3.Settings(presolve=presolve, threads=1))
    assert (s.info.iters == 0) == presolve
    assert (s.info.factorizations == 0) == presolve
    assert s.info.cvxopt_init_used is (None if presolve else False)
    assert np.isnan(s.info.pobj) and np.isnan(s.info.dobj)
    assert s.info.solve_time >= s.info.presolve_time
    if primal:
        assert s.status is cpo3.SolverStatus.PRIMAL_INFEASIBLE
        assert np.all(np.isnan(s.x))
        assert p.b @ s.y == pytest.approx(-1, abs=1e-6)
        np.testing.assert_allclose(p.A.T @ s.y, 0, rtol=0, atol=1e-6)
        assert np.all(s.y >= 0)
    else:
        assert s.status is cpo3.SolverStatus.DUAL_INFEASIBLE
        assert np.all(np.isnan(s.y))
        assert p.c @ s.x == pytest.approx(-1, abs=1e-6)
        np.testing.assert_allclose(p.A @ s.x + s.s, 0, rtol=0, atol=1e-6)
