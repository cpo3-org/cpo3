"""Check release archive contents and versions before uploading them."""

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

SOURCE_FILES = {
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "build.rs",
    "src/lib.rs",
    "src/linear_solvers/accelerate.c",
    "src/linear_solvers/cholmod.c",
    "python/cpo3/__init__.py",
    "python/cpo3/bindings.pyi",
    "python/cpo3/py.typed",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
}


def check(path: Path, version: str) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {
                "cpo3/__init__.py",
                "cpo3/bindings.pyi",
                "cpo3/py.typed",
            }
            assert required <= names, f"{path}: missing {required - names}"
            extensions = [
                name
                for name in names
                if name.startswith("cpo3/bindings.") and name.endswith((".so", ".pyd"))
            ]
            assert len(extensions) == 1, f"{path}: expected one native extension"
            # Optional runtime libraries must remain external to the wheel.
            native = [
                name
                for name in names
                if ".so" in Path(name).name or name.endswith((".dylib", ".dll", ".pyd"))
            ]
            assert native == extensions, (
                f"{path}: unexpected bundled libraries {native}"
            )
            metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
            assert len(metadata_paths) == 1, f"{path}: expected one METADATA file"
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            assert metadata["Name"] == "cpo3"
            assert metadata["Version"] == version
            assert any(n.endswith("/LICENSE") for n in names), "missing license"
    else:
        with tarfile.open(path) as archive:
            prefix = f"cpo3-{version}/"
            names = {n.removeprefix(prefix) for n in archive.getnames()}
            assert SOURCE_FILES <= names, f"{path}: missing {SOURCE_FILES - names}"
            if path.name.endswith(".tar.gz"):
                assert "tests/test_solver.py" in names, "sdist missing tests"
                assert any(n.startswith("tests/fixtures/") for n in names)
            assert not any(
                "__pycache__" in n or n.endswith((".so", ".pyd", ".pyc")) for n in names
            ), f"{path}: source archive contains build artifacts"
    print(f"Checked {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    for artifact in args.archives:
        check(artifact, args.version)
