from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy import sparse

from . import _core


class LinearSolver(str, Enum):
    """Linear-system implementation used by the Rust solver."""

    FAER = "faer"
    FERAL = "feral"


class Equilibration(str, Enum):
    """Diagonal equilibration method for problem data."""

    NONE = "none"
    RUIZ = "ruiz"


class Cone(ABC):
    """Abstract base class for cone descriptors."""

    __slots__ = ()

    @property
    @abstractmethod
    def vector_dim(self) -> int:
        """Number of constraint rows occupied by this cone block."""


@dataclass(frozen=True, slots=True)
class ZeroCone(Cone):
    """An equality-constraint block."""

    dim: int

    def __post_init__(self) -> None:
        _validate_positive(self.dim, "zero-cone dim")

    @property
    def vector_dim(self) -> int:
        return self.dim


@dataclass(frozen=True, slots=True)
class NonnegativeCone(Cone):
    """A nonnegative-orthant block."""

    dim: int

    def __post_init__(self) -> None:
        _validate_positive(self.dim, "nonnegative-cone dim")

    @property
    def vector_dim(self) -> int:
        return self.dim


@dataclass(frozen=True, slots=True)
class SecondOrderCone(Cone):
    """A second-order-cone block."""

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("second-order-cone dim must be at least two")

    @property
    def vector_dim(self) -> int:
        return self.dim


@dataclass(frozen=True, slots=True)
class PositiveSemidefiniteCone(Cone):
    """A packed positive-semidefinite block of the given matrix order."""

    order: int

    def __post_init__(self) -> None:
        _validate_positive(self.order, "positive-semidefinite-cone order")

    @property
    def vector_dim(self) -> int:
        return self.order * (self.order + 1) // 2


PSDCone = PositiveSemidefiniteCone


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:

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
    initial_mu: float = 1e-3
    data_equilibration: Equilibration = Equilibration.RUIZ
    max_equilibration_passes: int = 2
    equilibration_tolerance: float = 1e-2

    linear_solver: LinearSolver = LinearSolver.FAER
    kkt_bound_elimination: bool = True
    initial_diagonal_floor: float = 1e-12
    max_refinement_steps: int = 30
    refinement_tolerance: float = 1e-7

    step_fraction: float = 0.999

    def __post_init__(self) -> None:
        _core._validate_settings(self)


class SolverStatus(str, Enum):
    """Ordinary termination outcomes; numerical failures raise RuntimeError."""

    SOLVED = "solved"
    PRIMAL_INFEASIBLE = "primal_infeasible"
    DUAL_INFEASIBLE = "dual_infeasible"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"


class Problem:
    """Validated, Rust-owned conic quadratic problem data."""

    __slots__ = ("_native",)

    def __init__(
        self,
        *,
        P: sparse.spmatrix | sparse.sparray | None,
        c: npt.ArrayLike,
        A: sparse.spmatrix | sparse.sparray,
        b: npt.ArrayLike,
        cones: Sequence[Cone],
    ) -> None:
        native_p = None if P is None else _matrix_parts(P, "P")
        native_a = _matrix_parts(A, "A")
        native_c = _vector(c, "c")
        native_b = _vector(b, "b")
        native_cones = [_native_cone(cone) for cone in cones]
        self._native = _core._Problem(
            native_p,
            native_c,
            native_a,
            native_b,
            native_cones,
        )

    @classmethod
    def _from_native(cls, native: _core._Problem) -> Problem:
        problem = cls.__new__(cls)
        problem._native = native
        return problem

    @property
    def b(self) -> npt.NDArray[np.float64]:
        return self._native.b

    @property
    def c(self) -> npt.NDArray[np.float64]:
        return self._native.c

    @property
    def A(self) -> sparse.csc_matrix:
        """Constraint matrix as a SciPy CSC matrix."""

        return _matrix_from_parts(self._native._a_parts())

    @property
    def P(self) -> sparse.csc_matrix | None:
        """Export the full symmetric quadratic matrix, or None.

        The native problem stores only its upper triangle; this property
        materializes both halves for ordinary SciPy matrix operations.
        """

        parts = self._native._p_parts()
        if parts is None:
            return None
        upper = _matrix_from_parts(parts)
        return upper + sparse.triu(upper, k=1, format="csc").T

    @property
    def cones(self) -> list[Cone]:
        return [_cone_from_parts(name, size) for name, size in self._native.cones]

    @property
    def variable_count(self) -> int:
        return self._native.variable_count

    @property
    def constraint_count(self) -> int:
        return self._native.constraint_count


class SolutionInfo:
    """Summary information for a completed solve."""

    __slots__ = ("_native",)

    def __init__(self, native: _core._SolutionInfo) -> None:
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

        Includes initialization and internal retries, but excludes refinement.
        """
        return self._native.primary_solves

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
    def data_equilibration(self) -> Equilibration:
        return Equilibration(self._native.data_equilibration)

    @property
    def kkt_bound_elimination(self) -> bool:
        return self._native.kkt_bound_elimination


class Solution:
    """Solver result vectors and termination information."""

    __slots__ = ("_native",)

    def __init__(self, native: _core._Solution) -> None:
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


def read_mps(path: str | PathLike[str], *, first_named_set: bool = False) -> Problem:
    """Read an MPS or QPS file into a sparse problem.

    Files with several named RHS, RANGES, or BOUNDS sets are rejected unless
    ``first_named_set`` keeps the first set of each section.
    """

    return Problem._from_native(_core._read_mps(path, first_named_set=first_named_set))


def solve(
    problem: Problem,
    *,
    settings: Settings | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Solution:
    """Solve a previously constructed problem.

    ``warm_start`` is a tuple of NumPy arrays ``(x, y, s)`` in original
    units. Four interior candidates are screened by a local residual score;
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

    inner = _core._solve(
        problem._native,
        settings,
        warm_x=warm_x,
        warm_y=warm_y,
        warm_s=warm_s,
    )
    return Solution(inner)


def _cone_from_parts(name: str, size: int) -> Cone:
    if name == "zero":
        return ZeroCone(size)
    if name == "nonnegative":
        return NonnegativeCone(size)
    if name == "second_order":
        return SecondOrderCone(size)
    if name == "positive_semidefinite":
        return PositiveSemidefiniteCone(size)
    raise ValueError(f"unknown cone {name!r}")


def _matrix_from_parts(parts: tuple[Any, ...]) -> sparse.csc_matrix:
    rows, columns, pointers, indices, values = parts
    return sparse.csc_matrix((values, indices, pointers), shape=(rows, columns))


def _matrix_parts(value: object, name: str) -> _core._CscParts:
    if not sparse.issparse(value):
        raise TypeError(f"{name} must be a SciPy sparse matrix or sparse array")
    matrix: Any = value
    matrix = matrix.tocsc(copy=False)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional sparse matrix")
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.uintp)
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.uintp)
    rows, columns = matrix.shape
    return rows, columns, indptr, indices, data


def _vector(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return np.ascontiguousarray(vector)


def _native_cone(cone: Cone) -> tuple[str, int]:
    if isinstance(cone, ZeroCone):
        return ("zero", cone.dim)
    if isinstance(cone, NonnegativeCone):
        return ("nonnegative", cone.dim)
    if isinstance(cone, SecondOrderCone):
        return ("second_order", cone.dim)
    if isinstance(cone, PositiveSemidefiniteCone):
        return ("positive_semidefinite", cone.order)
    raise TypeError(f"unsupported cone descriptor {cone!r}")


def _validate_positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
