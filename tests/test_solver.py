from dataclasses import dataclass

import cpo3
import numpy as np
import pytest
from scipy import sparse


@dataclass
class Case:
    problem: cpo3.Problem
    x: list[float]
    y: list[float]
    s: list[float]
    objective: float


@pytest.fixture
def lp() -> Case:
    # min x, subject to x >= 1.
    return Case(
        cpo3.Problem(
            P=None,
            c=[1.0],
            A=sparse.csc_matrix([[-1.0]]),
            b=[-1.0],
            cones=[cpo3.NonnegativeCone(1)],
        ),
        x=[1.0],
        y=[1.0],
        s=[0.0],
        objective=1.0,
    )


@pytest.fixture
def bounded_qp() -> Case:
    # min x²/2 - 2x, subject to x <= 1 (active bound).
    return Case(
        cpo3.Problem(
            P=sparse.eye(1, format="csc"),
            c=[-2.0],
            A=sparse.csc_matrix([[1.0]]),
            b=[1.0],
            cones=[cpo3.NonnegativeCone(1)],
        ),
        x=[1.0],
        y=[1.0],
        s=[0.0],
        objective=-1.5,
    )


@pytest.fixture
def inactive_qp() -> Case:
    # min x²/2 - 3x, subject to x >= 0 (inactive bound).
    return Case(
        cpo3.Problem(
            P=sparse.eye(1, format="csc"),
            c=[-3.0],
            A=sparse.csc_matrix([[-1.0]]),
            b=[0.0],
            cones=[cpo3.NonnegativeCone(1)],
        ),
        x=[3.0],
        y=[0.0],
        s=[3.0],
        objective=-4.5,
    )


@pytest.fixture
def mixed_qp() -> Case:
    # min x²/2, subject to x = 3 and x >= 0.
    return Case(
        cpo3.Problem(
            P=sparse.eye(1, format="csc"),
            c=[0.0],
            A=sparse.csc_matrix([[1.0], [-1.0]]),
            b=[3.0, 0.0],
            cones=[cpo3.ZeroCone(1), cpo3.NonnegativeCone(1)],
        ),
        x=[3.0],
        y=[-3.0, 0.0],
        s=[0.0, 3.0],
        objective=4.5,
    )


@pytest.fixture
def coupled_qp() -> Case:
    # Coupled curvature and a general inequality survive presolve.
    return Case(
        cpo3.Problem(
            P=sparse.csc_matrix([[2.0, 1.0], [1.0, 2.0]]),
            c=[-5.0, -5.0],
            A=sparse.csc_matrix([[1.0, 0.0], [1.0, 1.0]]),
            b=[0.5, 2.0],
            cones=[cpo3.NonnegativeCone(2)],
        ),
        x=[0.5, 1.5],
        y=[1.0, 1.5],
        s=[0.0, 0.0],
        objective=-6.75,
    )


@pytest.fixture
def equality_qp() -> Case:
    # min x²/2, subject to x = 2.
    return Case(
        cpo3.Problem(
            P=sparse.eye(1, format="csc"),
            c=[0.0],
            A=sparse.csc_matrix([[1.0]]),
            b=[2.0],
            cones=[cpo3.ZeroCone(1)],
        ),
        x=[2.0],
        y=[-2.0],
        s=[0.0],
        objective=2.0,
    )


@pytest.fixture(params=list(cpo3.LinearSolver), ids=lambda solver: solver.value)
def backend(request: pytest.FixtureRequest) -> cpo3.LinearSolver:
    return request.param


def assert_solution(case: Case, solution: cpo3.Solution) -> None:
    assert solution.status is cpo3.SolverStatus.SOLVED
    for actual, expected in [
        (solution.x, case.x),
        (solution.y, case.y),
        (solution.s, case.s),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert solution.info.pobj == pytest.approx(case.objective, rel=2e-6, abs=2e-6)
    assert solution.info.dobj == pytest.approx(case.objective, rel=2e-6, abs=2e-6)
    # Check the original equations independently of the reported residuals.
    p = case.problem
    px = np.zeros_like(solution.x) if p.P is None else p.P @ solution.x
    np.testing.assert_allclose(p.A @ solution.x + solution.s, p.b, rtol=0, atol=1e-7)
    np.testing.assert_allclose(px + p.A.T @ solution.y + p.c, 0, rtol=0, atol=1e-7)
    assert abs(solution.s @ solution.y) < 2e-6
    assert solution.info.pres < 1e-7
    assert solution.info.dres < 1e-7
    assert solution.info.comp < 2e-6


@pytest.mark.parametrize(
    "name,cvxopt_init",
    [
        ("lp", False),
        ("bounded_qp", False),
        ("inactive_qp", False),
        ("mixed_qp", False),
        ("lp", True),
        ("coupled_qp", True),
    ],
)
def test_numerical_solve(
    request: pytest.FixtureRequest,
    name: str,
    backend: cpo3.LinearSolver,
    cvxopt_init: bool,
) -> None:
    case = request.getfixturevalue(name)
    solution = cpo3.solve(
        case.problem,
        settings=cpo3.Settings(
            presolve=False,
            threads=1,
            linear_solver=backend,
            cvxopt_init=cvxopt_init,
        ),
    )
    assert_solution(case, solution)
    assert solution.info.linear_solver is backend
    assert solution.info.factorizations > 0
    assert solution.info.primary_solves > 0
    assert solution.info.iters > 0
    assert solution.info.cvxopt_init_used is cvxopt_init


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("eliminate_bounds", [False, True])
def test_bound_elimination(
    coupled_qp: Case,
    backend: cpo3.LinearSolver,
    presolve: bool,
    eliminate_bounds: bool,
) -> None:
    solution = cpo3.solve(
        coupled_qp.problem,
        settings=cpo3.Settings(
            presolve=presolve,
            eliminate_bounds=eliminate_bounds,
            threads=1,
            linear_solver=backend,
        ),
    )
    assert_solution(coupled_qp, solution)
    assert solution.info.linear_solver is backend
    assert solution.info.factorizations > 0
    assert solution.info.primary_solves > 0


def test_equality_solver_bypasses_initialization(equality_qp: Case) -> None:
    solution = cpo3.solve(
        equality_qp.problem,
        settings=cpo3.Settings(
            presolve=False,
            threads=1,
            linear_solver=cpo3.LinearSolver.FERAL,
            cvxopt_init=True,
        ),
    )
    assert_solution(equality_qp, solution)
    assert solution.info.linear_solver is cpo3.LinearSolver.FAER
    assert solution.info.cvxopt_init_used is None
    assert solution.info.factorizations == 1
    assert solution.info.primary_solves == 1
    assert solution.info.refinement_solves == 0
    assert solution.info.lsmr_iterations == 0
    assert solution.info.iters == 1


@pytest.mark.parametrize("name", ["lp", "bounded_qp"])
def test_presolve_solves_without_factorization(
    request: pytest.FixtureRequest, name: str
) -> None:
    case = request.getfixturevalue(name)
    solution = cpo3.solve(
        case.problem, settings=cpo3.Settings(presolve=True, threads=1)
    )
    assert_solution(case, solution)
    assert solution.info.iters == 0
    assert solution.info.factorizations == 0
    assert solution.info.primary_solves == 0
    assert solution.info.cvxopt_init_used is None
    assert solution.info.solve_time >= solution.info.presolve_time


def test_equality_chain() -> None:
    # min ||x||²/2, subject to x_i - x_(i+1) = 1: a centered ramp.
    n = 100
    problem = cpo3.Problem(
        P=sparse.eye(n, format="csc"),
        c=np.zeros(n),
        A=sparse.diags([1.0, -1.0], [0, 1], shape=(n - 1, n), format="csc"),
        b=np.ones(n - 1),
        cones=[cpo3.ZeroCone(n - 1)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(presolve=False, threads=1))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(
        solution.x, (n - 1) / 2 - np.arange(n), rtol=0, atol=1e-6
    )
    assert solution.info.factorizations == 1
    assert solution.info.primary_solves > 0
    assert solution.info.pres <= 2e-8
    assert solution.info.dres <= 1e-8


def test_initial_optimality_takes_precedence_over_timeout() -> None:
    p = cpo3.Problem(
        P=sparse.eye(1, format="csc"),
        c=[0.0],
        A=sparse.eye(1, format="csc"),
        b=[0.0],
        cones=[cpo3.ZeroCone(1)],
    )
    s = cpo3.solve(p, settings=cpo3.Settings(presolve=False, threads=1, time_limit=0.0))
    assert s.status is cpo3.SolverStatus.SOLVED
    assert s.info.iters == 0
    assert s.info.factorizations == 0
    np.testing.assert_array_equal(s.x, [0.0])


def test_warm_start_screening_and_forcing() -> None:
    p = cpo3.Problem(
        P=None,
        c=[1, 1],
        A=sparse.csc_matrix([[-1, -1], [-1, 0], [0, -1]]),
        b=[-1, 0, 0],
        cones=[cpo3.NonnegativeCone(3)],
    )
    settings = cpo3.Settings(presolve=False, threads=1)
    cold = cpo3.solve(p, settings=settings)
    assert cold.info.warm_start_used is None
    assert cold.info.cvxopt_init_used is False
    again = cpo3.solve(p, settings=settings, warm_start=(cold.x, cold.y, cold.s))
    assert again.status is cpo3.SolverStatus.SOLVED
    assert again.info.warm_start_used is True
    assert again.info.cvxopt_init_used is None

    junk = (np.array([1e6, -1e6]), np.array([-1, 1e-9, 5e3]), np.array([1e3, -7, 1e-9]))
    rejected = cpo3.solve(p, settings=settings, warm_start=junk)
    assert rejected.status is cpo3.SolverStatus.SOLVED
    assert rejected.info.warm_start_used is False
    assert rejected.info.cvxopt_init_used is False

    forced = cpo3.Settings(presolve=False, threads=1, force_warm_start=True)
    used = cpo3.solve(p, settings=forced, warm_start=junk)
    assert used.status is cpo3.SolverStatus.SOLVED
    assert used.info.warm_start_used is True
    assert used.info.cvxopt_init_used is None
    without_warm = cpo3.solve(p, settings=forced)
    assert without_warm.status is cpo3.SolverStatus.SOLVED
    assert without_warm.info.warm_start_used is None
    with pytest.raises(ValueError, match="warm start"):
        cpo3.solve(
            p, settings=settings, warm_start=(np.zeros(1), np.zeros(3), np.zeros(3))
        )
