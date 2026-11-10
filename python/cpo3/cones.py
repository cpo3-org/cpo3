"""Cone descriptors and their representation in the Rust bindings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


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


def _native_cone(cone: Cone) -> tuple[str, int]:
    if isinstance(cone, ZeroCone):
        return "zero", cone.dim
    if isinstance(cone, NonnegativeCone):
        return "nonnegative", cone.dim
    if isinstance(cone, SecondOrderCone):
        return "second_order", cone.dim
    if isinstance(cone, PositiveSemidefiniteCone):
        return "positive_semidefinite", cone.order
    raise TypeError(f"unsupported cone descriptor {cone!r}")


def _validate_positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
