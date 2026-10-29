"""Python conversion, ownership, settings, and result contracts."""

from dataclasses import replace
from pathlib import Path

import cpo3
import numpy as np
import pytest
from scipy import sparse


@pytest.fixture
def problem() -> cpo3.Problem:
    return cpo3.Problem(
        P=sparse.csc_matrix([[2.0, 1.0], [1.0, 2.0]]),
        c=[-5.0, -5.0],
        A=sparse.csc_matrix([[1.0, 0.0], [1.0, 1.0]]),
        b=[0.5, 2.0],
        cones=[cpo3.NonnegativeCone(2)],
    )


@pytest.mark.parametrize("format", ["csc", "csr", "coo"])
def test_sparse_inputs_and_strided_vectors_are_copied(format: str) -> None:
    a = sparse.coo_matrix(([0.5, 0.5], ([0, 0], [0, 0])), shape=(1, 1)).asformat(format)
    c = np.array([-1.0, 99.0])[::2]
    p = cpo3.Problem(P=None, c=c, A=a, b=[2.0], cones=[cpo3.NonnegativeCone(1)])
    a.data[:] = 99.0
    c[:] = 99.0
    np.testing.assert_array_equal(p.A.toarray(), [[1.0]])
    np.testing.assert_array_equal(p.c, [-1.0])
    returned = p.c
    returned[:] = 10.0
    np.testing.assert_array_equal(p.c, [-1.0])


def test_python_defaults_match_rust() -> None:
    from cpo3 import _core

    assert _core._default_settings_match(cpo3.Settings())
    assert not _core._default_settings_match(cpo3.Settings(ruiz_iters=0))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("tol_feas", float("nan"), ValueError),
        ("max_iters", 0, ValueError),
        ("threads", -1, OverflowError),
        ("time_limit", float("inf"), ValueError),
    ],
)
def test_settings_validation(field: str, value: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        replace(cpo3.Settings(), **{field: value})


@pytest.mark.parametrize("bad_c", [[[1.0]], [float("nan")]])
def test_invalid_vectors_raise_value_error(bad_c: list) -> None:
    with pytest.raises(ValueError):
        cpo3.Problem(
            P=None, c=bad_c, A=sparse.eye(1), b=[1.0], cones=[cpo3.NonnegativeCone(1)]
        )


def test_unsupported_cone_raises_not_implemented() -> None:
    p = cpo3.Problem(
        P=None,
        c=[0.0],
        A=sparse.csc_matrix((2, 1)),
        b=[0.0, 0.0],
        cones=[cpo3.SecondOrderCone(2)],
    )
    with pytest.raises(NotImplementedError):
        cpo3.solve(p)


@pytest.mark.parametrize("backend", [cpo3.LinearSolver.FAER, cpo3.LinearSolver.FERAL])
def test_results_warm_start_and_array_ownership(
    problem: cpo3.Problem, backend: cpo3.LinearSolver
) -> None:
    settings = cpo3.Settings(
        threads=1, presolve=False, force_warm_start=True, linear_solver=backend
    )
    warm = (np.array([0.4, 1.4]), np.ones(2), np.array([0.1, 0.2]))
    before = tuple(v.copy() for v in warm)
    result = cpo3.solve(problem, settings=settings, warm_start=warm)
    assert result.status is cpo3.SolverStatus.SOLVED
    assert result.info.linear_solver is backend
    assert result.info.warm_start_used is True
    assert result.info.primary_solves > 0
    assert result.info.refinement_solves >= 0
    assert result.info.solve_time >= 0.0
    np.testing.assert_allclose(result.x, [0.5, 1.5], atol=1e-7)
    for value, original in zip(warm, before):
        np.testing.assert_array_equal(value, original)
    for name in ("x", "y", "s"):
        values = getattr(result, name)
        assert values.shape == (2,)
        assert values.dtype == np.float64
        original = values.copy()
        values[:] = 99.0
        np.testing.assert_array_equal(getattr(result, name), original)


@pytest.mark.parametrize(
    "limit,status",
    [
        ({"max_iters": 1}, cpo3.SolverStatus.MAX_ITERATIONS),
        ({"time_limit": 0.0}, cpo3.SolverStatus.TIME_LIMIT),
    ],
)
def test_limit_statuses_are_exposed(
    problem: cpo3.Problem, limit: dict, status: cpo3.SolverStatus
) -> None:
    settings = replace(cpo3.Settings(presolve=False, threads=1), **limit)
    assert cpo3.solve(problem, settings=settings).status is status


def test_read_mps_paths_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "small.mps"
    with pytest.raises(OSError, match="small.mps"):
        cpo3.read_mps(path)
    path.write_text(
        "NAME SMALL\nROWS\n N OBJ\n L LIM\nCOLUMNS\n X OBJ -1 LIM 1\nRHS\n RHS1 LIM 2\nENDATA\n"
    )
    result = cpo3.solve(cpo3.read_mps(path), settings=cpo3.Settings(threads=1))
    assert result.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(result.x, [2.0], atol=1e-7)
    path.write_text("not an MPS file")
    with pytest.raises(ValueError):
        cpo3.read_mps(path)


@pytest.mark.parametrize(
    "c,a,b,status",
    [
        ([0.0], [[1.0], [-1.0]], [0.0, -1.0], cpo3.SolverStatus.PRIMAL_INFEASIBLE),
        ([-1.0], [[-1.0]], [0.0], cpo3.SolverStatus.DUAL_INFEASIBLE),
    ],
)
def test_certificate_statuses_and_vector_exposure(
    c: list, a: list, b: list, status: cpo3.SolverStatus
) -> None:
    problem = cpo3.Problem(
        P=None,
        c=c,
        A=sparse.csc_matrix(a),
        b=b,
        cones=[cpo3.NonnegativeCone(len(b))],
    )
    result = cpo3.solve(problem, settings=cpo3.Settings(threads=1))
    assert result.status is status
    assert result.x.shape == (len(c),)
    assert result.y.shape == result.s.shape == (len(b),)
    if status is cpo3.SolverStatus.PRIMAL_INFEASIBLE:
        assert np.isfinite(result.y).all()
        assert np.isnan(result.x).all()
        assert np.isnan(result.s).all()
    else:
        assert np.isfinite(result.x).all()
        assert np.isfinite(result.s).all()
        assert np.isnan(result.y).all()
