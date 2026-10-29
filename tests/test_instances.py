"""Small explicit cases and benchmark LPs/QPs; --full runs all benchmark instances."""

import gzip
from pathlib import Path

import cpo3
import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse

Array = npt.NDArray[np.float64]


def infty_norm(vector: Array) -> np.float64:
    # Unconstrained problems have empty constraint residuals.
    if vector.size == 0:
        return np.float64(0.0)
    return np.linalg.norm(vector, ord=np.inf)


def _check_kkt_residuals(
    problem: cpo3.Problem, x: Array, y: Array, s: Array, tolerance: float
) -> None:
    P, c, A, b = problem.P, problem.c, problem.A, problem.b
    eqs = sum(cone.dim for cone in problem.cones if isinstance(cone, cpo3.ZeroCone))
    n, m = c.size, b.size
    for name, vector, size in (("x", x, n), ("y", y, m), ("s", s, m)):
        assert vector.shape == (size,), name
        assert np.isfinite(vector).all(), name

    Ax = A @ x
    ATy = A.T @ y
    Px = np.zeros_like(x) if P is None else P @ x
    p_scale = 1.0 + max(infty_norm(Ax), infty_norm(s), infty_norm(b))
    d_scale = 1.0 + max(infty_norm(Px), infty_norm(ATy), infty_norm(c))
    quadratic = 0.5 * x @ Px
    primal = quadratic + c @ x
    dual = -quadratic - b @ y
    g_scale = 1.0 + abs(primal) + abs(dual)
    residuals = {
        "feas": infty_norm(Ax + s - b) / p_scale,
        "stat": infty_norm(Px + ATy + c) / d_scale,
        "eq": infty_norm(s[:eqs]) / p_scale,
        "slack": infty_norm(np.minimum(s[eqs:], 0.0)) / p_scale,
        "mult": infty_norm(np.minimum(y[eqs:], 0.0)) / (1.0 + infty_norm(y)),
        "gap": abs(primal - dual) / g_scale,
    }
    assert np.isfinite([primal, dual, p_scale, d_scale, g_scale]).all()
    for name, value in residuals.items():
        assert value <= tolerance, f"{name}: {value:.3e} > {tolerance:.3e}"


def _check_primal_certificate(
    problem: cpo3.Problem, y: Array, tolerance: float
) -> None:
    A, b = problem.A, problem.b
    eqs = sum(cone.dim for cone in problem.cones if isinstance(cone, cpo3.ZeroCone))
    assert y.shape == b.shape and np.isfinite(y).all()
    slope = b @ y
    assert np.isfinite(slope) and slope < 0.0
    y = y / -slope
    assert np.isfinite(y).all()
    assert infty_norm(A.T @ y) <= tolerance
    assert infty_norm(np.minimum(y[eqs:], 0.0)) <= tolerance


def _check_dual_certificate(
    problem: cpo3.Problem, x: Array, s: Array, tolerance: float
) -> None:
    P, c, A, b = problem.P, problem.c, problem.A, problem.b
    eqs = sum(cone.dim for cone in problem.cones if isinstance(cone, cpo3.ZeroCone))
    assert x.shape == c.shape and np.isfinite(x).all()
    assert s.shape == b.shape and np.isfinite(s).all()
    slope = c @ x
    assert np.isfinite(slope) and slope < 0.0
    x, s = x / -slope, s / -slope
    assert np.isfinite(x).all() and np.isfinite(s).all()
    assert infty_norm(A @ x + s) <= tolerance
    if P is not None:
        assert infty_norm(P @ x) <= tolerance
    assert infty_norm(s[:eqs]) <= tolerance
    assert infty_norm(np.minimum(s[eqs:], 0.0)) <= tolerance


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks" / "data"


SHORT_INSTANCES = {
    "netlib/AFIRO",
    "netlib/SC50B",
    "netlib/ADLITTLE",
    "netlib/BLEND",
    "netlib/SC105",
    "netlib/SCAGR7",
    "netlib/SHARE2B",
    "netlib/STOCFOR1",
    "maros-meszaros/HS21",
    "maros-meszaros/HS35",
    "maros-meszaros/HS51",
    "maros-meszaros/HS76",
    "maros-meszaros/HS118",
    "maros-meszaros/QAFIRO",
    "maros-meszaros/QSCAGR7",
    "maros-meszaros/CVXQP1_S",
    "netlib-infeasible/itest2",
    "netlib-infeasible/itest6",
    "netlib-infeasible/galenet",
    "netlib-infeasible/bgprtr",
    "netlib-infeasible/woodinfe",
    "netlib-infeasible/forest6",
    "netlib-infeasible/klein1",
    "netlib-infeasible/ex73a",
}


ITERATION_LIMIT = ("Iteration limit", AssertionError)


def benchmark_cases(
    *collections: str,
    failures: dict[tuple[str, bool], tuple[str, type[Exception]]] | None = None,
) -> list[pytest.ParameterSet]:
    cases = []
    for collection in collections:
        paths = sorted((BENCHMARKS / collection).glob("*.mps.gz"))
        assert paths, f"Missing benchmark collection: {collection}"
        for path in paths:
            name = f"{collection}/{path.name.removesuffix('.mps.gz')}"
            for presolve in (False, True):
                marks = [] if name in SHORT_INSTANCES else [pytest.mark.full]
                if failures and (name, presolve) in failures:
                    reason, error = failures[name, presolve]
                    marks.append(
                        pytest.mark.xfail(reason=reason, raises=error, strict=True)
                    )
                mode = "presolve" if presolve else "no_presolve"
                cases.append(
                    pytest.param(name, presolve, id=f"{mode}-{name}", marks=marks)
                )
    return cases


@pytest.fixture
def problem(request: pytest.FixtureRequest, tmp_path: Path) -> cpo3.Problem:
    source = BENCHMARKS / f"{request.param}.mps.gz"
    path = tmp_path / source.stem
    path.write_bytes(gzip.decompress(source.read_bytes()))
    return cpo3.read_mps(path, first_named_set=True)


@pytest.mark.parametrize(
    "problem,presolve",
    benchmark_cases(
        "netlib",
        "maros-meszaros",
        failures={
            ("netlib/CYCLE", False): ITERATION_LIMIT,
            ("netlib/GREENBEA", False): ITERATION_LIMIT,
            ("netlib/GREENBEA", True): ITERATION_LIMIT,
            ("netlib/PILOT.JA", True): ITERATION_LIMIT,
            ("netlib/PILOT87", True): ITERATION_LIMIT,
            ("maros-meszaros/QPILOTNO", False): ITERATION_LIMIT,
            ("maros-meszaros/QFORPLAN", True): (
                "Nonfinite RHS",
                RuntimeError,
            ),
        },
    ),
    indirect=["problem"],
)
def test_benchmark_solution(problem: cpo3.Problem, presolve: bool) -> None:
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    _check_kkt_residuals(problem, solution.x, solution.y, solution.s, tolerance=1e-7)


@pytest.mark.parametrize(
    "problem,presolve",
    benchmark_cases(
        "netlib-infeasible",
        failures={
            ("netlib-infeasible/cplex2", False): ITERATION_LIMIT,
            ("netlib-infeasible/cplex2", True): ITERATION_LIMIT,
            ("netlib-infeasible/gosh", False): ITERATION_LIMIT,
            ("netlib-infeasible/greenbea", False): ITERATION_LIMIT,
            ("netlib-infeasible/pang", False): ITERATION_LIMIT,
            ("netlib-infeasible/pang", True): ITERATION_LIMIT,
            ("netlib-infeasible/refinery", True): ITERATION_LIMIT,
            ("netlib-infeasible/gran", False): (
                "Certificate residual exceeds tolerance",
                AssertionError,
            ),
        },
    ),
    indirect=["problem"],
)
def test_benchmark_infeasibility(problem: cpo3.Problem, presolve: bool) -> None:
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.PRIMAL_INFEASIBLE
    _check_primal_certificate(problem, solution.y, tolerance=1e-7)


@pytest.mark.parametrize(
    "problem,presolve",
    benchmark_cases(
        "netlib-infeasible",
        failures={
            ("netlib-infeasible/gosh", False): ITERATION_LIMIT,
            ("netlib-infeasible/gosh", True): ITERATION_LIMIT,
            ("netlib-infeasible/greenbea", False): ITERATION_LIMIT,
            ("netlib-infeasible/pang", False): ITERATION_LIMIT,
            ("netlib-infeasible/cplex2", False): (
                "Returns SOLVED for near-feasible LP",
                AssertionError,
            ),
            ("netlib-infeasible/cplex2", True): (
                "Returns SOLVED for near-feasible LP",
                AssertionError,
            ),
        },
    ),
    indirect=["problem"],
)
def test_benchmark_dual_infeasibility(problem: cpo3.Problem, presolve: bool) -> None:
    # Dualize min 0 subject to A*x+s=b, s in K:
    # min b^T*y subject to A^T*y=0, y in K*.
    # The origin is feasible, and a Farkas certificate gives an unbounded ray.
    assert problem.P is None
    A, b = problem.A, problem.b
    m, n = A.shape
    eqs = sum(cone.dim for cone in problem.cones if isinstance(cone, cpo3.ZeroCone))
    ineqs = m - eqs
    cones: list[cpo3.Cone] = [cpo3.ZeroCone(n)]
    if ineqs:
        cones.append(cpo3.NonnegativeCone(ineqs))
    dual = cpo3.Problem(
        P=None,
        c=b,
        A=sparse.vstack([A.T, -sparse.eye(m, format="csc")[eqs:]], format="csc"),
        b=np.zeros(n + ineqs),
        cones=cones,
    )
    solution = cpo3.solve(dual, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.DUAL_INFEASIBLE

    _check_dual_certificate(dual, solution.x, solution.s, tolerance=1e-7)


@pytest.mark.parametrize("presolve", [False, True])
def test_unconstrained_qp(presolve: bool) -> None:
    # min x^2 - 4x; optimum x = 2.
    problem = cpo3.Problem(
        P=sparse.csc_matrix([[2.0]]),
        c=[-4.0],
        A=sparse.csc_matrix((0, 1)),
        b=[],
        cones=[],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [2.0], atol=1e-7, rtol=0)
    _check_kkt_residuals(problem, solution.x, solution.y, solution.s, tolerance=1e-7)


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("quadratic", [False, True], ids=["lp", "qp"])
def test_equality_only(presolve: bool, quadratic: bool) -> None:
    # min x (LP) or x^2 + x (QP), subject to x = 2.
    problem = cpo3.Problem(
        P=sparse.csc_matrix([[2.0]]) if quadratic else None,
        c=[1.0],
        A=sparse.csc_matrix([[1.0]]),
        b=[2.0],
        cones=[cpo3.ZeroCone(1)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [2.0], atol=1e-7, rtol=0)
    _check_kkt_residuals(problem, solution.x, solution.y, solution.s, tolerance=1e-7)


@pytest.mark.parametrize(
    "presolve,rhs",
    [
        pytest.param(False, 4.0, id="consistent-no_presolve"),
        pytest.param(True, 4.0, id="consistent-presolve"),
        pytest.param(
            False,
            5.0,
            id="inconsistent-no_presolve",
            marks=pytest.mark.xfail(
                reason="LSMR fails to certify infeasibility",
                raises=RuntimeError,
                strict=True,
            ),
        ),
        pytest.param(True, 5.0, id="inconsistent-presolve"),
    ],
)
def test_redundant_constraints(presolve: bool, rhs: float) -> None:
    # min x, subject to x = 2, 2x = rhs, and 0x = 0.
    problem = cpo3.Problem(
        P=None,
        c=[1.0],
        A=sparse.csc_matrix([[1.0], [2.0], [0.0]]),
        b=[2.0, rhs, 0.0],
        cones=[cpo3.ZeroCone(3)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    if rhs == 4.0:
        assert solution.status is cpo3.SolverStatus.SOLVED
        np.testing.assert_allclose(solution.x, [2.0], atol=1e-7, rtol=0)
        _check_kkt_residuals(
            problem, solution.x, solution.y, solution.s, tolerance=1e-7
        )
    else:
        assert solution.status is cpo3.SolverStatus.PRIMAL_INFEASIBLE
        _check_primal_certificate(problem, solution.y, tolerance=1e-7)


@pytest.mark.parametrize("presolve", [False, True])
def test_singular_qp(presolve: bool) -> None:
    # min x1^2 - 2x1, subject to 0 <= x2 <= 1; every (1, x2) is optimal.
    problem = cpo3.Problem(
        P=sparse.diags([2.0, 0.0], format="csc"),
        c=[-2.0, 0.0],
        A=sparse.csc_matrix([[0.0, -1.0], [0.0, 1.0]]),
        b=[0.0, 1.0],
        cones=[cpo3.NonnegativeCone(2)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x[0], 1.0, atol=1e-7, rtol=0)
    _check_kkt_residuals(problem, solution.x, solution.y, solution.s, tolerance=1e-7)


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("quadratic", [False, True], ids=["lp", "qp"])
def test_unbounded(presolve: bool, quadratic: bool) -> None:
    # min -x2 (LP) or x1^2 - x2 (QP), subject to x2 >= 0.
    # The descent direction (0, 1) lies in the Hessian's nullspace.
    problem = cpo3.Problem(
        P=sparse.diags([2.0, 0.0], format="csc") if quadratic else None,
        c=[0.0, -1.0],
        A=sparse.csc_matrix([[0.0, -1.0]]),
        b=[0.0],
        cones=[cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.DUAL_INFEASIBLE
    _check_dual_certificate(problem, solution.x, solution.s, tolerance=1e-7)


@pytest.mark.parametrize("presolve", [False, True])
def test_fixed_variable(presolve: bool) -> None:
    # min x, subject to 2 <= x <= 2; presolve can eliminate the whole problem.
    problem = cpo3.Problem(
        P=None,
        c=[1.0],
        A=sparse.csc_matrix([[-1.0], [1.0]]),
        b=[-2.0, 2.0],
        cones=[cpo3.NonnegativeCone(2)],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(threads=1, presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [2.0], atol=1e-7, rtol=0)
    _check_kkt_residuals(problem, solution.x, solution.y, solution.s, tolerance=1e-7)
