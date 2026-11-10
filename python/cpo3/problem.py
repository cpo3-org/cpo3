"""Problem data, sparse-array conversion, and MPS/QPS loading."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from . import bindings
from .cones import Cone, _cone_from_parts, _native_cone


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
        self._native = bindings._Problem(
            native_p,
            native_c,
            native_a,
            native_b,
            native_cones,
        )

    @classmethod
    def _from_native(cls, native: bindings._Problem) -> Problem:
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


def read_mps(path: str | PathLike[str], *, first_named_set: bool = False) -> Problem:
    """Read an MPS or QPS file into a sparse problem.

    Files with several named RHS, RANGES, or BOUNDS sets are rejected unless
    ``first_named_set`` keeps the first set of each section.
    """

    return Problem._from_native(
        bindings._read_mps(path, first_named_set=first_named_set)
    )


def _matrix_from_parts(parts: tuple[Any, ...]) -> sparse.csc_matrix:
    rows, columns, pointers, indices, values = parts
    return sparse.csc_matrix((values, indices, pointers), shape=(rows, columns))


def _matrix_parts(value: object, name: str) -> bindings._CscParts:
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
