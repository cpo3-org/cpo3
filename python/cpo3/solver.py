"""Solving entry point and result types."""

from enum import Enum

import numpy as np
import numpy.typing as npt

from . import bindings
from .problem import Problem
from .settings import LinearSolver, Settings


class SolverStatus(str, Enum):
    """Ordinary termination outcomes; numerical failures raise RuntimeError."""

    SOLVED = "solved"
    PRIMAL_INFEASIBLE = "primal_infeasible"
    DUAL_INFEASIBLE = "dual_infeasible"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"


class SolutionInfo:
    """Summary information for a completed solve."""

    __slots__ = ("_native",)

    def __init__(self, native: bindings._SolutionInfo) -> None:
        self._native = native

    @property
    def iters(self) -> int:
        return self._native.iters

    @property
    def solve_time(self) -> float:
        return self._native.solve_time

    @property
    def pobj(self) -> float:
        return self._native.pobj

    @property
    def dobj(self) -> float:
        return self._native.dobj

    @property
    def pres(self) -> float:
        return self._native.pres

    @property
    def dres(self) -> float:
        return self._native.dres

    @property
    def comp(self) -> float:
        return self._native.comp

    @property
    def warm_start_score(self) -> float | None:
        return self._native.warm_start_score

    @property
    def warm_start_used(self) -> bool | None:
        return self._native.warm_start_used

    @property
    def cvxopt_init_used(self) -> bool | None:
        """Whether the CVXOPT-style point started the solve.

        ``False`` covers the canonical point, including after a fallback;
        ``None`` when an accepted warm start was used or no interior-point
        iterate was formed.
        """

        return self._native.cvxopt_init_used

    @property
    def presolve_rows_removed(self) -> int:
        return self._native.presolve_rows_removed

    @property
    def presolve_columns_removed(self) -> int:
        return self._native.presolve_columns_removed

    @property
    def presolve_time(self) -> float:
        return self._native.presolve_time

    @property
    def presolve_parallel_comparisons(self) -> int | None:
        return self._native.presolve_parallel_comparisons

    @property
    def primary_solves(self) -> int:
        """Initial linear solves attempted for KKT right-hand sides.

        Includes initialization, internal retries, and equality-only LSMR
        preconditioner applications, but excludes refinement.
        """
        return self._native.primary_solves

    @property
    def lsmr_iterations(self) -> int:
        """Iterations of the equality-only LSMR fallback."""
        return self._native.lsmr_iterations

    @property
    def refinement_solves(self) -> int:
        """Correction linear solves attempted, including internally failed attempts.

        Excludes the initial solve for each right-hand side and residual checks.
        """
        return self._native.refinement_solves

    @property
    def refinement_settled(self) -> int:
        return self._native.refinement_settled

    @property
    def refinement_failures(self) -> int:
        return self._native.refinement_failures

    @property
    def factorizations(self) -> int:
        return self._native.factorizations

    @property
    def inertia_failures(self) -> int:
        return self._native.inertia_failures

    @property
    def linear_solver(self) -> LinearSolver:
        return LinearSolver(self._native.linear_solver)

    @property
    def eliminate_bounds(self) -> bool:
        return self._native.eliminate_bounds


class Solution:
    """Solver result vectors and termination information."""

    __slots__ = ("_native",)

    def __init__(self, native: bindings._Solution) -> None:
        self._native = native

    @property
    def status(self) -> SolverStatus:
        return SolverStatus(self._native.status)

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self._native.x

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self._native.y

    @property
    def s(self) -> npt.NDArray[np.float64]:
        return self._native.s

    @property
    def info(self) -> SolutionInfo:
        return SolutionInfo(self._native.info)


def solve(
    problem: Problem,
    *,
    settings: Settings | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Solution:
    """Solve a previously constructed problem.

    ``warm_start`` is a tuple of NumPy arrays ``(x, y, s)`` in original
    units. Three interior candidates are screened by a local residual score;
    the best is used when its score is at most 100. With
    ``settings.force_warm_start=True``, the best finite candidate is used
    regardless of score. All-equality problems skip warm screening.
    Omit it or pass ``None`` for a cold solve.
    """

    if not isinstance(problem, Problem):
        raise TypeError("solve() requires a cpo3.Problem")
    if settings is None:
        settings = Settings()
    elif not isinstance(settings, Settings):
        raise TypeError("settings must be a cpo3.Settings")
    warm_x = warm_y = warm_s = None
    if warm_start is not None:
        warm_x, warm_y, warm_s = warm_start
        warm_x = np.ascontiguousarray(warm_x, dtype=np.float64).ravel()
        warm_y = np.ascontiguousarray(warm_y, dtype=np.float64).ravel()
        warm_s = np.ascontiguousarray(warm_s, dtype=np.float64).ravel()

    inner = bindings._solve(
        problem._native,
        settings,
        warm_x=warm_x,
        warm_y=warm_y,
        warm_s=warm_s,
    )
    return Solution(inner)
