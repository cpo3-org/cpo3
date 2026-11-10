from dataclasses import replace

import cpo3
import numpy as np
import pytest
from scipy import sparse


def problem(p, c, a, b) -> cpo3.Problem:
    return cpo3.Problem(
        P=None if p is None else sparse.csc_matrix(p),
        c=c,
        A=sparse.csc_matrix(a),
        b=b,
        cones=[cpo3.ZeroCone(len(b))] if len(b) else [],
    )


def settings(**kwargs) -> cpo3.Settings:
    return replace(cpo3.Settings(presolve=False, threads=1), **kwargs)


def norm(x):
    return np.max(np.abs(x), initial=0.0)


def verify(p: cpo3.Problem, s: cpo3.Solution, expected: cpo3.SolverStatus) -> None:
    assert s.status is expected
    px = np.zeros_like(s.x) if p.P is None else p.P @ s.x
    if expected is cpo3.SolverStatus.SOLVED:
        ax, aty = p.A @ s.x, p.A.T @ s.y
        assert norm(ax - p.b) <= 1e-7 * (1 + max(norm(p.b), norm(ax)))
        assert norm(px + aty + p.c) <= 1e-7 * (1 + max(norm(px), norm(aty), norm(p.c)))
        assert np.all(np.isfinite(s.x)) and np.all(np.isfinite(s.y))
    elif expected is cpo3.SolverStatus.PRIMAL_INFEASIBLE:
        assert p.b @ s.y == pytest.approx(-1, abs=1e-7)
        assert norm(p.A.T @ s.y) < 1e-7
    elif expected is cpo3.SolverStatus.DUAL_INFEASIBLE:
        assert p.c @ s.x == pytest.approx(-1, abs=1e-7)
        assert norm(p.A @ s.x) < 1e-7
        assert norm(px) < 1e-7
    else:
        pytest.fail(f"Unexpected status: {expected}")


@pytest.mark.parametrize("eps", [1.0, 1e-4, 1e-8])
def test_inconsistent_equalities(eps: float) -> None:
    # Identical left sides, contradictory right sides; scale one KKT column.
    p = problem([[1, 0], [0, eps]], [0, 0], [[eps, 1], [eps, 1]], [1, 2])
    s = cpo3.solve(p, settings=settings())
    verify(p, s, cpo3.SolverStatus.PRIMAL_INFEASIBLE)
    assert s.info.lsmr_iterations > 0


@pytest.mark.parametrize("eps", [1.0, 1e-4, 1e-8, 1e-12])
def test_unbounded(eps: float) -> None:
    p = problem([[eps, 0], [0, 0]], [0, -1], [[1, 0]], [1])
    s = cpo3.solve(p, settings=settings())
    verify(p, s, cpo3.SolverStatus.DUAL_INFEASIBLE)


@pytest.mark.parametrize("eps", [1e-4, 1e-8, 1e-12])
def test_singular_consistent(eps: float) -> None:
    # Redundant equalities and a free variable give a nonunique optimum.
    p = problem(np.diag([eps, 0, 0]), [-1, 0, 0], [[0, 1, 0], [0, 2, 0]], [1, 2])
    cfg = settings(
        refine_init_reg=1e-4, refine_iters=0, ruiz_iters=0, tol_infeas_rel=1e-14
    )
    s = cpo3.solve(p, settings=cfg)
    verify(p, s, cpo3.SolverStatus.SOLVED)
    assert abs(s.x[0] * eps - 1) < 1e-7
    if eps == 1e-12:
        assert s.info.lsmr_iterations > 0
        # The default certificate tolerance accepts this small positive curvature.
        loose = replace(cfg, tol_infeas_rel=cpo3.Settings().tol_infeas_rel)
        verify(p, cpo3.solve(p, settings=loose), cpo3.SolverStatus.DUAL_INFEASIBLE)


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-6])
def test_nearly_dependent_but_feasible(eps: float) -> None:
    p = problem(np.eye(2), [0, 0], [[1, 0], [1, eps]], [1, 2])
    verify(p, cpo3.solve(p, settings=settings()), cpo3.SolverStatus.SOLVED)


def test_extreme_near_dependence_is_not_declared_infeasible() -> None:
    p = problem(np.eye(2), [0, 0], [[1, 0], [1, 1e-8]], [1, 2])
    # A verified solution or numerical failure is acceptable; infeasibility is not.
    try:
        s = cpo3.solve(p, settings=settings(tol_infeas_rel=1e-14))
    except RuntimeError as error:
        assert "numerical" in str(error)
    else:
        verify(p, s, cpo3.SolverStatus.SOLVED)


@pytest.mark.parametrize(
    "c,b,status",
    [
        (0.0, [1.0], cpo3.SolverStatus.PRIMAL_INFEASIBLE),
        (-1.0, [0.0], cpo3.SolverStatus.DUAL_INFEASIBLE),
        (-1.0, [], cpo3.SolverStatus.DUAL_INFEASIBLE),
    ],
)
def test_zero_operator(c, b, status) -> None:
    p = problem(None, [c], np.zeros((len(b), 1)), b)
    s = cpo3.solve(p, settings=settings())
    verify(p, s, status)


@pytest.mark.parametrize("corrections", [0, 2, 5])
def test_iteration_budgets_are_independent(corrections: int) -> None:
    p = problem([[1]], [0], [[1], [1]], [1, 2])
    cfg = settings(
        max_iters=1, refine_iters=0, eq_lsmr_iters=1, eq_refine_iters=corrections
    )
    s = cpo3.solve(p, settings=cfg)
    verify(p, s, cpo3.SolverStatus.PRIMAL_INFEASIBLE)
    assert s.info.refinement_solves == corrections
    assert s.info.lsmr_iterations == 1
    # One direct solve, LSMR initialization, and two applications per iteration.
    assert s.info.primary_solves == 4
    with pytest.raises(RuntimeError, match=r"after 0 iterations \(IterationLimit\)"):
        cpo3.solve(p, settings=replace(cfg, eq_lsmr_iters=0))


@pytest.mark.parametrize("feasible", [False, True])
def test_refinement_ignores_refine_tol(feasible: bool) -> None:
    p = (
        problem(np.eye(2), [0, 0], [[1, 0], [1, 1e-6]], [1, 2])
        if feasible
        else problem([[1]], [0], [[1], [1]], [1, 2])
    )
    status = (
        cpo3.SolverStatus.SOLVED if feasible else cpo3.SolverStatus.PRIMAL_INFEASIBLE
    )
    reference = None
    for tol in [1e-30, 1e-7, 1e2]:
        s = cpo3.solve(p, settings=settings(refine_tol=tol))
        verify(p, s, status)
        result = (
            s.x.tobytes(),
            s.y.tobytes(),
            s.info.refinement_solves,
            s.info.lsmr_iterations,
        )
        if reference is not None:
            assert result == reference
        reference = result


@pytest.mark.parametrize("presolve", [False, True])
def test_presolve_recovers_points_and_bypasses_initialization(presolve: bool) -> None:
    n = 32
    p = problem(
        1.999 * np.eye(n) + np.full((n, n), 0.001),
        np.full(n, -(2 + 0.001 * (n - 1))),
        [[1] + [0] * (n - 1), [0] + [1] * (n - 1)],
        [1, n - 1],
    )
    s = cpo3.solve(
        p,
        settings=settings(
            presolve=presolve,
            cvxopt_init=True,
            force_warm_start=True,
            linear_solver=cpo3.LinearSolver.FERAL,
        ),
        warm_start=(np.full(n, 7.0), np.full(2, 2.0), np.zeros(2)),
    )
    verify(p, s, cpo3.SolverStatus.SOLVED)
    np.testing.assert_allclose(s.x, 1, rtol=0, atol=1e-7)
    assert s.info.warm_start_used is False
    assert s.info.cvxopt_init_used is None
    if presolve:
        assert s.info.presolve_columns_removed > 0
