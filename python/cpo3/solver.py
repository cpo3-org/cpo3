"""Solving entry point and result types."""

from enum import Enum

import numpy as np

from . import bindings
from .bindings import Solution
from .problem import Problem
from .settings import Settings


class SolverStatus(str, Enum):
    """Ordinary termination outcomes; numerical failures raise RuntimeError."""

    SOLVED = "solved"
    PRIMAL_INFEASIBLE = "primal_infeasible"
    DUAL_INFEASIBLE = "dual_infeasible"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"


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

    return bindings._solve(
        problem._native,
        settings,
        warm_x=warm_x,
        warm_y=warm_y,
        warm_s=warm_s,
    )
