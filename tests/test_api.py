import cpo3
import pytest
import scipy.sparse
from cpo3 import _core


def test_sparse_formats_are_canonicalized() -> None:
    problem = cpo3.Problem(
        P=scipy.sparse.eye(2, format="csr"),
        c=[1.0, 2.0],
        A=scipy.sparse.eye(2, format="csr"),
        b=[3.0, 4.0],
        cones=[cpo3.NonnegativeCone(2)],
    )
    assert scipy.sparse.isspmatrix_csc(problem.P)
    assert scipy.sparse.isspmatrix_csc(problem.A)
    assert problem.variable_count == 2
    assert problem.constraint_count == 2


def test_psd_cone_uses_packed_dimension() -> None:
    assert cpo3.PositiveSemidefiniteCone(3).vector_dim == 6


@pytest.mark.parametrize(
    "p,cones,message",
    [
        ([[1, 0], [0, 1]], [cpo3.PositiveSemidefiniteCone(2)], "cone dimensions"),
        ([[1, 1], [0, 1]], [cpo3.NonnegativeCone(2)], "P must be symmetric"),
    ],
)
def test_invalid_problem(p, cones, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cpo3.Problem(
            P=scipy.sparse.csc_matrix(p),
            c=[1, 2],
            A=scipy.sparse.eye(2, format="csc"),
            b=[3, 4],
            cones=cones,
        )


def test_default_settings_match_in_rust_and_python() -> None:
    py_settings = cpo3.Settings()
    assert _core._default_settings_match(py_settings)


def test_negative_threads_rejected() -> None:
    with pytest.raises(OverflowError):
        cpo3.Settings(threads=-1)
