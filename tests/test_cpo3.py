import inspect
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


def test_cones_use_dim_attribute() -> None:
    cone = cpo3.NonnegativeCone(dim=2)

    assert isinstance(cone, cpo3.Cone)
    assert cone.dim == 2
    assert cone.vector_dim == 2
    assert not hasattr(cone, "dimension")


def test_cone_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        cpo3.Cone()


def test_psd_cone_reports_its_packed_vector_dim() -> None:
    cone = cpo3.PositiveSemidefiniteCone(order=3)

    assert isinstance(cone, cpo3.Cone)
    assert cone.vector_dim == 6


def test_settings_use_short_names() -> None:
    settings = cpo3.Settings(max_iters=200, tol_gap_abs=1e-7, tol_gap_rel=1e-6)

    assert settings.max_iters == 200
    assert settings.tol_gap_abs == 1e-7
    assert settings.tol_feas == 1e-8
    assert settings.tol_gap_rel == 1e-6
    assert settings.linear_solver is cpo3.LinearSolver.FAER
    assert settings.data_equilibration is cpo3.Equilibration.RUIZ
    assert settings.kkt_bound_elimination is True
    assert settings.initial_shift_floor == 1e-10
    assert settings.min_static_regularization == 0.0
    assert not hasattr(settings, "shift_floor")
    assert not hasattr(settings, "initial_scale")
    assert settings.tol_infeas_abs == 1e-8
    assert settings.tol_infeas_rel == 1e-8
    assert settings.certificate_ktratio == 1e9
    assert settings.force_warm_start is False
    assert settings.cvxopt_init is False
    assert settings.presolve is True
    assert not hasattr(settings, "max_iterations")
    assert not hasattr(settings, "absolute_tolerance")
    assert not hasattr(settings, "relative_tolerance")


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_cvxopt_init_requires_bool(value: Any) -> None:
    with pytest.raises(TypeError, match="cvxopt_init must be a bool"):
        cpo3.Settings(cvxopt_init=value)


def test_cvxopt_initialization_solves_and_is_reported() -> None:
    problem = sparse_problem()
    canonical = cpo3.solve(problem, settings=cpo3.Settings(presolve=False))
    cvxopt = cpo3.solve(
        problem,
        settings=cpo3.Settings(presolve=False, cvxopt_init=True),
    )
    assert canonical.status is cpo3.SolverStatus.SOLVED
    assert cvxopt.status is cpo3.SolverStatus.SOLVED
    assert cvxopt.info.cvxopt_init_used is True
    assert canonical.info.cvxopt_init_used is False
    assert cvxopt.info.pobj == pytest.approx(canonical.info.pobj, abs=1e-6)
    np.testing.assert_allclose(cvxopt.x, canonical.x, atol=1e-6)
    # The initialization solve is counted with the main loop's linear algebra.
    assert cvxopt.info.factorizations >= 1
    assert cvxopt.info.primary_solves >= 1


def test_presolved_problem_reports_no_initialization() -> None:
    # Presolve solves this bound-only problem before any interior point forms.
    solution = cpo3.solve(sparse_problem(), settings=cpo3.Settings(cvxopt_init=True))
    assert solution.status is cpo3.SolverStatus.SOLVED
    assert solution.info.iters == 0
    assert solution.info.cvxopt_init_used is None


def test_accepted_warm_start_reports_no_cold_strategy() -> None:
    problem = sparse_problem()
    settings = cpo3.Settings(presolve=False, cvxopt_init=True)
    cold = cpo3.solve(problem, settings=settings)
    warm = cpo3.solve(problem, settings=settings, warm_start=(cold.x, cold.y, cold.s))
    assert warm.info.warm_start_used is True
    assert warm.info.cvxopt_init_used is None


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_kkt_bound_elimination_requires_bool(value: Any) -> None:
    with pytest.raises(TypeError, match="kkt_bound_elimination must be a bool"):
        cpo3.Settings(kkt_bound_elimination=value)


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


def test_warm_start_from_numpy_arrays_is_used() -> None:
    problem = sparse_problem()
    settings = cpo3.Settings(presolve=False)
    cold = cpo3.solve(problem, settings=settings)
    assert cold.info.warm_start_used is None
    assert cold.info.warm_start_score is None
    warm = cpo3.solve(problem, settings=settings, warm_start=(cold.x, cold.y, cold.s))
    assert warm.status is cpo3.SolverStatus.SOLVED
    assert warm.info.warm_start_used is True
    assert warm.info.iters <= cold.info.iters
    assert warm.info.warm_start_score is not None
    assert math.isfinite(warm.info.warm_start_score)
    assert warm.info.warm_start_score <= 100.0


def test_warm_start_requires_arrays_from_solution() -> None:
    problem = sparse_problem()
    cold = cpo3.solve(problem)
    with pytest.raises(TypeError):
        cpo3.solve(problem, warm_start=cold)  # ty: ignore[invalid-argument-type]


def test_junk_warm_start_is_ignored_unless_forced(
    capfd: pytest.CaptureFixture[str],
) -> None:
    problem = sparse_problem()
    # The cpo3-style score rejects this point even after sphere normalization.
    junk = (np.array([1e6, -1e6]), np.array([-1.0, 1e-9]), np.array([25.0, -7.0]))
    solution = cpo3.solve(
        problem, settings=cpo3.Settings(presolve=False), warm_start=junk
    )
    assert solution.info.warm_start_used is False
    assert solution.info.warm_start_score is not None
    assert solution.info.warm_start_score > 100.0
    assert "warm start ignored" in capfd.readouterr().err
    forced = cpo3.solve(
        problem,
        settings=cpo3.Settings(force_warm_start=True, presolve=False),
        warm_start=junk,
    )
    assert forced.info.warm_start_used is True
    assert forced.info.warm_start_score == solution.info.warm_start_score
    assert forced.status is cpo3.SolverStatus.SOLVED
    cold = cpo3.solve(
        problem,
        settings=cpo3.Settings(force_warm_start=True, presolve=False),
        warm_start=None,
    )
    assert cold.info.warm_start_used is None
    assert cold.info.warm_start_score is None
    assert cold.info.iters == solution.info.iters
    with pytest.raises(ValueError, match="unpack"):
        # Deliberately pass two vectors to exercise runtime validation.
        cpo3.solve(problem, warm_start=(cold_x := np.zeros(2), cold_x))  # ty: ignore[invalid-argument-type]


def test_invalid_certificate_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        cpo3.Settings(certificate_ktratio=0.5)
    with pytest.raises(ValueError):
        cpo3.Settings(tol_infeas_abs=-1.0)


def test_linear_solver_choices() -> None:
    assert [solver.value for solver in cpo3.LinearSolver] == [
        "faer",
        "feral",
    ]


def test_equilibration_choices() -> None:
    assert [method.value for method in cpo3.Equilibration] == ["none", "ruiz"]


def test_sparse_formats_are_canonicalized() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.eye(2, format="csc"),
        c=[1.0, 2.0],
        A=scipy.sparse.eye(2, format="csr"),
        b=[3.0, 4.0],
        cones=[cpo3.NonnegativeCone(2)],
    )

    assert scipy.sparse.isspmatrix_csc(problem.A)


def test_read_mps_returns_a_sparse_problem() -> None:
    problem = cpo3.read_mps(FIXTURES / "parser_compatibility.mps")

    assert isinstance(problem, cpo3.Problem)
    assert scipy.sparse.isspmatrix_csc(problem.A)
    assert problem.variable_count == 6
    assert problem.constraint_count == 11


def test_read_mps_accepts_qps_files() -> None:
    problem = cpo3.read_mps(FIXTURES / "parser_compatibility.qps")

    assert isinstance(problem, cpo3.Problem)
    assert scipy.sparse.isspmatrix_csc(problem.A)
    assert problem.variable_count == 2
    assert problem.constraint_count == 3


def test_read_mps_reports_file_and_parse_errors() -> None:
    with pytest.raises(OSError, match="does_not_exist.mps"):
        cpo3.read_mps(FIXTURES / "does_not_exist.mps")
    with pytest.raises(ValueError, match="malformed.mps"):
        cpo3.read_mps(FIXTURES / "malformed.mps")


def test_read_mps_named_set_option() -> None:
    with pytest.raises(
        ValueError, match="BOUNDS section contains more than one named set"
    ):
        cpo3.read_mps(FIXTURES / "two_bound_sets.mps")
    problem = cpo3.read_mps(FIXTURES / "two_bound_sets.mps", first_named_set=True)
    assert problem.constraint_count == 4


def test_problem_validates_psd_cone_vector_dim() -> None:
    problem = cpo3.Problem(
        P=None,
        c=[1.0],
        A=scipy.sparse.csc_matrix(np.ones((6, 1))),
        b=np.ones(6),
        cones=[cpo3.PositiveSemidefiniteCone(3)],
    )

    assert problem.constraint_count == 6


def test_solve_has_one_problem_only_signature() -> None:
    signature = inspect.signature(cpo3.solve)

    assert list(signature.parameters) == ["problem", "settings", "warm_start"]
    assert signature.parameters["settings"].kind is inspect.Parameter.KEYWORD_ONLY

    invalid_problem: Any = np.eye(2)
    with pytest.raises(TypeError, match="requires a cpo3.Problem"):
        cpo3.solve(invalid_problem)


def test_solution_names_the_dual_vector_y() -> None:
    assert isinstance(cpo3.Solution.y, property)
    assert isinstance(cpo3.Solution.s, property)
    assert not hasattr(cpo3.Solution, "z")


def test_solution_info_uses_short_metric_names() -> None:
    assert isinstance(cpo3.SolutionInfo.iters, property)
    assert isinstance(cpo3.SolutionInfo.linear_solver, property)
    assert isinstance(cpo3.SolutionInfo.data_equilibration, property)
    assert isinstance(cpo3.SolutionInfo.kkt_bound_elimination, property)
    assert not hasattr(cpo3.SolutionInfo, "iterations")
    for name in ("pobj", "dobj", "pres", "dres"):
        assert isinstance(getattr(cpo3.SolutionInfo, name), property)
    for name in (
        "primal_objective",
        "dual_objective",
        "primal_residual",
        "dual_residual",
    ):
        assert not hasattr(cpo3.SolutionInfo, name)
    assert isinstance(cpo3.SolutionInfo.comp, property)
    assert not hasattr(cpo3.SolutionInfo, "complementarity")
    assert not hasattr(cpo3.SolutionInfo, "absolute_gap")
    assert not hasattr(cpo3.SolutionInfo, "relative_gap")


def test_verbose_solve_reports_progress_to_stderr(
    capfd: pytest.CaptureFixture[str],
) -> None:
    solution = cpo3.solve(
        sparse_problem(),
        settings=cpo3.Settings(max_iters=1, verbose=True, presolve=False),
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert "cpo3  |  Conic Quadratic Optimization" in captured.err
    assert "Matteo Santamaria" not in captured.err
    assert "sigma" in captured.err
    assert "maximum iterations" in captured.err
    assert solution.status is cpo3.SolverStatus.MAX_ITERATIONS


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


@pytest.mark.parametrize(
    "linear_solver",
    [
        cpo3.LinearSolver.FAER,
        cpo3.LinearSolver.FERAL,
    ],
)
@pytest.mark.parametrize("kkt_bound_elimination", [False, True])
def test_native_sparse_linear_solvers_accept_sparse_problems(
    linear_solver: cpo3.LinearSolver,
    kkt_bound_elimination: bool,
) -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.csc_matrix([[1.0]]),
        c=[-2.0],
        A=scipy.sparse.csc_matrix([[1.0]]),
        b=[1.0],
        cones=[cpo3.NonnegativeCone(1)],
    )
    settings = cpo3.Settings(
        max_iters=400,
        linear_solver=linear_solver,
        presolve=False,
        kkt_bound_elimination=kkt_bound_elimination,
    )
    solution = cpo3.solve(problem, settings=settings)

    assert solution.info.linear_solver is cpo3.LinearSolver(linear_solver)
    assert solution.info.kkt_bound_elimination is kkt_bound_elimination
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [1.0], atol=2e-6)
    np.testing.assert_allclose(solution.y, [1.0], atol=2e-6)
    np.testing.assert_allclose(solution.s, [0.0], atol=2e-6)


def test_same_problem_can_be_solved_concurrently() -> None:
    from concurrent.futures import ThreadPoolExecutor

    problem = sparse_problem()
    expected = cpo3.solve(problem)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cpo3.solve(problem), range(12)))
    for result in results:
        assert result.status == expected.status
        assert result.info.iters == expected.info.iters
        for actual, reference in zip(
            (result.x, result.y, result.s),
            (expected.x, expected.y, expected.s),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_mu_floor_is_validated_at_settings_construction(value: float) -> None:
    with pytest.raises(ValueError, match="initial_shift_floor"):
        cpo3.Settings(initial_shift_floor=value)


@pytest.mark.parametrize("name", ["A", "P"])
@pytest.mark.parametrize("value", [np.eye(2), [[1.0, 0.0], [0.0, 1.0]]])
def test_problem_rejects_dense_matrices(name: str, value: Any) -> None:
    matrices: dict[str, Any] = {"A": scipy.sparse.eye(2), "P": scipy.sparse.eye(2)}
    matrices[name] = value
    with pytest.raises(TypeError, match=f"{name} must be a SciPy sparse"):
        cpo3.Problem(
            **matrices, b=np.ones(2), c=np.ones(2), cones=[cpo3.NonnegativeCone(2)]
        )


@pytest.mark.parametrize(
    "constructor",
    [
        scipy.sparse.csc_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.coo_matrix,
        scipy.sparse.csc_array,
        scipy.sparse.csr_array,
        scipy.sparse.coo_array,
    ],
)
def test_sparse_inputs_preserve_values_without_mutation(constructor: Any) -> None:
    # Duplicate entries must be summed; the caller's data must remain unchanged.
    source = scipy.sparse.coo_matrix(
        ([0.5, 0.5, 2.0], ([0, 0, 1], [0, 0, 1])), shape=(2, 2)
    )
    matrix = constructor(source)
    before = matrix.copy()
    problem = cpo3.Problem(
        P=matrix, A=matrix, b=np.ones(2), c=np.ones(2), cones=[cpo3.NonnegativeCone(2)]
    )
    np.testing.assert_array_equal(problem.A.toarray(), np.diag([1.0, 2.0]))
    assert problem.P is not None
    np.testing.assert_array_equal(problem.P.toarray(), np.diag([1.0, 2.0]))
    assert problem.A.has_canonical_format
    assert matrix.nnz == before.nnz
    np.testing.assert_array_equal(matrix.data, before.data)
    exported = problem.A
    exported.data[:] = 0.0
    np.testing.assert_array_equal(problem.A.toarray(), np.diag([1.0, 2.0]))


def test_sparse_matrix_values_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        cpo3.Problem(
            P=None,
            A=scipy.sparse.csc_matrix([[np.nan]]),
            b=np.ones(1),
            c=np.ones(1),
            cones=[cpo3.NonnegativeCone(1)],
        )


def test_sparse_only_api_has_no_backend_or_auto_solver() -> None:
    assert "backend" not in inspect.signature(cpo3.Problem).parameters
    assert not hasattr(cpo3, "Backend")
    assert not hasattr(_core, "_DenseMatrix")
    assert not hasattr(sparse_problem(), "backend")
    assert not hasattr(cpo3.SolutionInfo, "backend")
    with pytest.raises(ValueError):
        cpo3.Settings(linear_solver="auto")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_static_regularization_is_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ValueError, match="min_static_regularization"):
        cpo3.Settings(min_static_regularization=value)


def test_predictor_corrector_work_counts() -> None:
    solution = cpo3.solve(sparse_problem(), settings=cpo3.Settings(presolve=False))
    assert solution.status is cpo3.SolverStatus.SOLVED
    assert solution.info.factorizations == solution.info.iters
    assert solution.info.primary_solves == 3 * solution.info.iters


def test_barrier_problem_uses_complementarity_instead_of_initial_floor() -> None:
    # A requested barrier-free floor must not change a positive-barrier solve.
    settings = cpo3.Settings(
        presolve=False,
        initial_shift_floor=100.0,
        max_iters=100,
    )
    solution = cpo3.solve(sparse_problem(), settings=settings)
    assert solution.status is cpo3.SolverStatus.SOLVED
    assert solution.info.factorizations == solution.info.iters < 100
    assert solution.info.primary_solves == 3 * solution.info.iters
    assert settings.initial_shift_floor == 100.0
    baseline = cpo3.solve(sparse_problem(), settings=cpo3.Settings(presolve=False))
    assert solution.info.iters == baseline.info.iters
    np.testing.assert_array_equal(solution.x, baseline.x)


@pytest.mark.parametrize("limit", ["iterations", "time"])
def test_work_counters_on_early_returns(limit: str) -> None:
    settings = cpo3.Settings(
        presolve=False, max_iters=1, time_limit=0.0 if limit == "time" else None
    )
    solution = cpo3.solve(sparse_problem(), settings=settings)
    expected = 0 if limit == "time" else 1
    assert solution.info.iters == solution.info.factorizations == expected


@pytest.mark.parametrize("backend", [cpo3.LinearSolver.FAER, cpo3.LinearSolver.FERAL])
def test_initial_and_refinement_solves_are_separate(backend: Any) -> None:
    problem = cpo3.Problem(
        P=None,
        c=[1.0],
        A=scipy.sparse.csc_matrix([[-1.0]]),
        b=[-1.0],
        cones=[cpo3.NonnegativeCone(1)],
    )
    solution = cpo3.solve(
        problem,
        settings=cpo3.Settings(
            presolve=False,
            linear_solver=backend,
            min_static_regularization=1e-4,
            refinement_tolerance=1e-14,
            max_iters=1000,
        ),
    )
    assert solution.status is cpo3.SolverStatus.SOLVED
    assert solution.info.primary_solves >= 3 * solution.info.iters
    assert solution.info.refinement_solves > 0

    unrefined = cpo3.solve(
        problem,
        settings=cpo3.Settings(
            presolve=False, linear_solver=backend, max_iters=1, max_refinement_steps=0
        ),
    )
    assert unrefined.info.primary_solves == 3
    assert unrefined.info.refinement_solves == 0


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("equality", [False, True])
def test_no_barrier_problems_with_redundant_equalities(
    presolve: bool, equality: bool
) -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.eye(2, format="csc"),
        c=[-1.0, -2.0],
        A=scipy.sparse.csc_matrix([[1.0, 1.0], [2.0, 2.0]])
        if equality
        else scipy.sparse.csc_matrix((0, 2)),
        b=[3.0, 6.0] if equality else [],
        cones=[cpo3.ZeroCone(2)] if equality else [],
    )
    solution = cpo3.solve(problem, settings=cpo3.Settings(presolve=presolve))
    assert solution.status is cpo3.SolverStatus.SOLVED
    np.testing.assert_allclose(solution.x, [1.0, 2.0], rtol=0, atol=1e-7)
    assert solution.info.factorizations == solution.info.iters
