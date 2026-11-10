import re
from pathlib import Path

import cpo3
import numpy as np
import pytest
import scipy.sparse

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("suffix,n,m", [("mps", 6, 11), ("qps", 2, 3)])
def test_read_mps(suffix: str, n: int, m: int) -> None:
    problem = cpo3.io.read_mps(FIXTURES / f"problem_data.{suffix}")

    assert isinstance(problem, cpo3.Problem)
    match suffix:
        case "mps":
            assert problem.P is None
        case "qps":
            assert scipy.sparse.isspmatrix_csc(problem.P)
    assert scipy.sparse.isspmatrix_csc(problem.A)
    assert problem.variable_count == n
    assert problem.constraint_count == m


ROW_MODEL = "NAME ROWS\nROWS\n N COST\n N UNUSED\n L R\nCOLUMNS\n X COST 1 R 1\n"


def read_model(tmp_path: Path, source: str, **kwargs) -> cpo3.Problem:
    path = tmp_path / "model.mps"
    path.write_text(source)
    return cpo3.io.read_mps(path, **kwargs)


def test_missing_file_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.mps"
    with pytest.raises(OSError, match=re.escape(str(path))):
        cpo3.io.read_mps(path)


def test_discrete_bounds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match='bound type BV for variable "X1" is not continuous'
    ):
        read_model(
            tmp_path,
            "NAME INTEGER\nROWS\n N COST\nCOLUMNS\n X1 COST 1\n"
            "BOUNDS\n BV BND1 X1\nENDATA\n",
        )


def test_rhs_and_ranges_define_interval_and_ignore_n_rows(tmp_path: Path) -> None:
    p = read_model(
        tmp_path,
        ROW_MODEL + "RHS\n SET COST 10 R 5\n SET COST 20 UNUSED 30\n"
        "RANGES\n SET UNUSED 10 R -2\n SET COST 20 UNUSED 30\n"
        "BOUNDS\n FR BND X\nENDATA\n",
    )
    np.testing.assert_array_equal(p.b, [-3, 5])
    np.testing.assert_array_equal(p.A.toarray(), [[-1], [1]])
    np.testing.assert_array_equal(p.c, [1])


@pytest.mark.parametrize("section", ["RHS", "RANGES"])
def test_first_named_set(tmp_path: Path, section: str) -> None:
    source = (
        ROW_MODEL + f"{section}\n FIRST COST 10\n SECOND MISSING 99\n FIRST R 2\n"
        "BOUNDS\n FR BND X\nENDATA\n"
    )
    with pytest.raises(
        ValueError, match=f"the {section} section contains more than one named set"
    ):
        read_model(tmp_path, source)
    p = read_model(tmp_path, source, first_named_set=True)
    np.testing.assert_array_equal(p.b, [2] if section == "RHS" else [2, 0])


@pytest.mark.parametrize("section", ["RHS", "RANGES"])
@pytest.mark.parametrize(
    "entries,message",
    [
        (" SET R 0 R 0", 'defines "R" more than once'),
        (" SET MISSING 1", 'row "MISSING" is referenced but not defined'),
        (" SET COST inf", 'non-finite value for "COST"'),
    ],
)
def test_invalid_rhs_and_ranges(
    tmp_path: Path, section: str, entries: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_model(tmp_path, ROW_MODEL + f"{section}\n{entries}\nENDATA\n")
