"""Problem file input."""

from os import PathLike

from . import bindings
from .problem import Problem


def read_mps(path: str | PathLike[str], *, first_named_set: bool = False) -> Problem:
    """Read an MPS or QPS file into a sparse problem.

    Files with several named RHS, RANGES, or BOUNDS sets are rejected unless
    ``first_named_set`` keeps the first set of each section.
    """

    return Problem.wrap(bindings._read_mps(path, first_named_set=first_named_set))
