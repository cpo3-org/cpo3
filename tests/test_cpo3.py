import inspect
import dataclasses
import math
from pathlib import Path
from typing import Any

import cpo3
import numpy as np
import pytest
import scipy.sparse
from cpo3 import _core

FIXTURES = Path(__file__).parent / "fixtures"


def sparse_problem() -> cpo3.Problem:
    return cpo3.Problem(
        P=scipy.sparse.eye(2, format="csc"),
        c=[1.0, 2.0],
        A=scipy.sparse.eye(2, format="csc"),
        b=[3.0, 4.0],
        cones=[cpo3.NonnegativeCone(2)],
    )


def test_primal_infeasible_problem_returns_a_certificate() -> None:
    problem = cpo3.Problem(
        P=None,
        c=np.array([1.0]),
        A=scipy.sparse.csc_matrix(np.array([[1.0], [-1.0]])),
        b=np.array([-1.0, 0.0]),
        cones=[cpo3.NonnegativeCone(2)],
    )
    solution = cpo3.solve(problem)
    assert solution.status is cpo3.SolverStatus.PRIMAL_INFEASIBLE
    assert np.all(np.isnan(solution.x))
    assert abs(solution.y @ np.array([-1.0, 0.0]) + 1.0) < 1e-6
    assert np.all(solution.y >= 0.0)


def test_dual_infeasible_problem_returns_a_ray() -> None:
    problem = cpo3.Problem(
        P=None,
        c=np.array([-1.0]),
        A=scipy.sparse.csc_matrix(np.array([[-1.0]])),
        b=np.array([0.0]),
        cones=[cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(problem)
    assert solution.status is cpo3.SolverStatus.DUAL_INFEASIBLE
    assert np.all(np.isnan(solution.y))
    assert abs(-solution.x[0] + 1.0) < 1e-6


def test_sparse_formats_are_canonicalized() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.eye(2, format="csr"),
        c=[1.0, 2.0],
        A=scipy.sparse.eye(2, format="csr"),
        b=[3.0, 4.0],
        cones=[cpo3.NonnegativeCone(2)],
    )
    assert scipy.sparse.isspmatrix_csc(problem.P)
    assert scipy.sparse.isspmatrix_csc(problem.A)


@pytest.mark.parametrize("suffix,n,m", [("mps", 6, 11), ("qps", 2, 3)])
def test_read_mps(suffix: str, n: int, m: int) -> None:
    problem = cpo3.read_mps(FIXTURES / f"problem_data.{suffix}")

    assert isinstance(problem, cpo3.Problem)
    match suffix:
        case "mps":
            assert problem.P is None
        case "qps":
            assert scipy.sparse.isspmatrix_csc(problem.P)
    assert scipy.sparse.isspmatrix_csc(problem.A)
    assert problem.variable_count == n
    assert problem.constraint_count == m


def test_solves_a_toy_quadratic() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.csc_matrix([[1.0]]),
        c=[-2.0],
        A=scipy.sparse.csc_matrix([[1.0]]),
        b=[1.0],
        cones=[cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(
        problem,
        settings=cpo3.Settings(max_iters=400, presolve=False),
    )

    assert solution.status is cpo3.SolverStatus.SOLVED
    assert solution.info.data_equilibration is cpo3.Equilibration.RUIZ
    np.testing.assert_allclose(solution.x, [1.0], atol=2e-6)
    np.testing.assert_allclose(solution.y, [1.0], atol=2e-6)
    np.testing.assert_allclose(solution.s, [0.0], atol=2e-6)
    assert solution.info.pobj == pytest.approx(-1.5, abs=2e-6)
    assert solution.info.dobj == pytest.approx(-1.5, abs=2e-6)


def test_solves_a_quadratic_with_an_inactive_nonnegative_constraint() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.csc_matrix([[1.0]]),
        c=[-3.0],
        A=scipy.sparse.csc_matrix([[-1.0]]),
        b=[0.0],
        cones=[cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(max_iters=400))

    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [3.0], atol=2e-6)
    np.testing.assert_allclose(solution.y, [0.0], atol=2e-6)
    np.testing.assert_allclose(solution.s, [3.0], atol=2e-6)
    assert solution.info.pobj == pytest.approx(-4.5, abs=2e-6)
    assert solution.info.dobj == pytest.approx(-4.5, abs=2e-6)


def test_solves_mixed_equality_and_inequality_constraints() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.csc_matrix([[1.0]]),
        c=[0.0],
        A=scipy.sparse.csc_matrix([[1.0], [-1.0]]),
        b=[3.0, 0.0],
        cones=[cpo3.ZeroCone(1), cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(max_iters=400))

    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [3.0], atol=2e-6)
    np.testing.assert_allclose(solution.y, [-3.0, 0.0], atol=2e-6)
    np.testing.assert_allclose(solution.s, [0.0, 3.0], atol=2e-6)
    assert solution.info.pobj == pytest.approx(4.5, abs=2e-6)
    assert solution.info.dobj == pytest.approx(4.5, abs=2e-6)

def test_default_settings_match_in_rust_and_python() -> None:
    py_settings = cpo3.Settings()
    assert _core._default_settings_match(py_settings)
