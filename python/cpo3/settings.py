"""Solver configuration and linear-system backend selection."""

from dataclasses import dataclass
from enum import Enum

from . import bindings


class LinearSolver(str, Enum):
    """Linear-system implementation used by the Rust solver."""

    FAER = "faer"
    ACCELERATE = "accelerate"
    CHOLMOD = "cholmod"
    PARDISO = "pardiso"

    def is_available(self) -> bool:
        """Whether this build and runtime provide the backend."""
        return bindings._linear_solver_available(self.value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Solver controls, termination tolerances, and numerical settings.

    Parameters
    ----------
    max_iters : int, default=100
        Maximum number of interior-point iterations. Must be positive.
        Unused for equality-only problems.
    time_limit : float or None, default=None
        Elapsed-time limit in seconds, including presolve. Must be nonnegative;
        None disables the limit. Checked between solver iterations.
    verbose : bool, default=False
        Print solver settings, iteration progress, and the final summary.
    presolve : bool, default=True
        Simplify the problem before solving and recover the solution in the
        original coordinates.
    tol_feas : float, default=1e-8
        Positive primal and dual feasibility tolerance. Each residual infinity
        norm must be less than this value times one plus the largest infinity
        norm of the terms in its equation.
    tol_gap_abs : float, default=1e-8
        Positive absolute duality-gap tolerance. Optimality requires
        feasibility and either the absolute or relative gap criterion.
    tol_gap_rel : float, default=1e-8
        Positive duality-gap tolerance relative to the objective scale.
    tol_infeas_abs : float, default=1e-8
        Nonnegative certificate slope threshold. A primal or dual
        infeasibility certificate requires b.T @ y or c.T @ x, respectively,
        to be less than -tol_infeas_abs.
    tol_infeas_rel : float, default=1e-8
        Nonnegative tolerance for certificate residuals, normalized by the
        certificate slope and ray norms.
    certificate_ktratio : float, default=1e9
        Consider infeasibility certificates only when the homogeneous ratio
        kappa / tau exceeds this threshold. Must be at least one.
    force_warm_start : bool, default=False
        Accept the best finite interior warm-start candidate even if its local
        score exceeds the screening threshold. Has no effect without a warm
        start or for equality-only problems.
    cvxopt_init : bool, default=False
        Use a CVXOPT/Clarabel-style linear-system initialization when no warm
        start is accepted. Falls back to the canonical spherical starting
        point if initialization fails.
    initial_mu : float, default=1.0
        Initial complementarity for the canonical spherical starting point.
        Must be in (0, 1]. Has no effect on accepted warm or CVXOPT-style starts
        or on equality-only problems.
    ruiz_iters : int, default=2
        Maximum number of Ruiz equilibration passes on the augmented problem
        data (P, A, b, c). Zero disables equilibration.
    ruiz_tol : float, default=1e-2
        Nonnegative tolerance for equilibration: stop when all nonzero,
        cone-compatible augmented row infinity norms are within this distance
        of one.
    linear_solver : LinearSolver, default=LinearSolver.FAER
        Sparse factorization backend: FAER, ACCELERATE, CHOLMOD, or PARDISO.
        ACCELERATE requires macOS. CHOLMOD and PARDISO require separately
        installed SuiteSparse and Intel MKL libraries, respectively. Use
        the backend's is_available() method to check availability.
        Equality-only problems always use Faer with regularized LDL and LSMR.
    threads : int, default=0
        Maximum parallelism for factorization and solves. Zero uses the
        backend default; one runs sequentially. Must be nonnegative.
        Accelerate supports only zero because its sparse solver exposes no
        per-solve thread cap. PARDISO uses MKL's thread-local limit.
        CHOLMOD caps its OpenMP work; simplicial LDL factorization is sequential.
    eliminate_bounds : bool, default=True
        Eliminate nonnegative-cone constraint rows with a single nonzero
        coefficient from the KKT factorization by folding their contributions
        into the primal diagonal.
    refine_init_reg : float, default=1e-12
        Initial nonnegative floor for diagonal magnitudes in the full KKT
        system used for factorization. The solver adjusts this floor during
        outer iterations; zero starts at the lower bound of 1e-20. Iterative
        refinement targets the original KKT system. Unused for equality-only
        problems, which use fixed dynamic pivot regularization.
    refine_iters : int, default=30
        Maximum number of iterative refinement corrections per KKT solve.
        Zero disables corrections and keeps the initial linear solve.
        Equality-only problems use eq_refine_iters instead.
    refine_tol : float, default=1e-7
        Positive componentwise backward-error target for iterative refinement.
        Unused for equality-only problems, which check optimality directly.
        Refinement may stop before reaching this target if further corrections
        are unlikely to improve the solution or meet the target within budget.
    eq_lsmr_iters : int or None, default=None
        Maximum LSMR iterations for equality-only problems. None uses
        max(4 * (m + n), 32), with dimensions measured after presolve.
        Zero disables LSMR iterations, but still checks the initial point.
    eq_refine_iters : int, default=10
        Maximum refinement corrections before LSMR for equality-only problems.
        Zero skips corrections and goes from the direct solve to LSMR if needed.
    step_fraction : float, default=0.999
        Fraction of the step to the cone boundary to take. Must be strictly
        between zero and one to retain an interior margin. Second-order cones
        take at most 0.99 of their own boundary step.

    Notes
    -----
    All numerical values must be finite. Iteration limits must be nonnegative,
    with max_iters strictly positive.
    """

    max_iters: int = 100
    time_limit: float | None = None
    verbose: bool = False
    presolve: bool = True

    tol_feas: float = 1e-8
    tol_gap_abs: float = 1e-8
    tol_gap_rel: float = 1e-8
    tol_infeas_abs: float = 1e-8
    tol_infeas_rel: float = 1e-8
    certificate_ktratio: float = 1e9

    force_warm_start: bool = False
    cvxopt_init: bool = False
    initial_mu: float = 1.0
    ruiz_iters: int = 2
    ruiz_tol: float = 1e-2

    linear_solver: LinearSolver = LinearSolver.FAER
    threads: int = 0
    eliminate_bounds: bool = True
    refine_init_reg: float = 1e-12
    refine_iters: int = 30
    refine_tol: float = 1e-7

    eq_lsmr_iters: int | None = None
    eq_refine_iters: int = 10

    step_fraction: float = 0.999

    def __post_init__(self) -> None:
        bindings._validate_settings(self)
