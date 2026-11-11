"""Run the suite outside the checkout using an installed, non-editable wheel."""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cpo3
import cpo3.bindings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("cpo3") == args.version
    for module in (cpo3, cpo3.bindings):
        assert module.__file__ is not None
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(Path(sys.prefix).resolve()), (
            f"Expected an installed wheel in {sys.prefix}, imported {path}"
        )
    package = Path(cpo3.__file__).parent
    assert (package / "py.typed").is_file()
    assert (package / "bindings.pyi").is_file()
    source = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="cpo3-wheel-test-") as directory:
        shutil.copytree(source / "tests", Path(directory) / "tests")
        subprocess.run(
            [sys.executable, "-m", "pytest", "-ra", "tests"],
            cwd=directory,
            check=True,
        )
