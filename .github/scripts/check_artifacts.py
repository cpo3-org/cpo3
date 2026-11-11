"""Ensure wheels contain only the extension, with no bundled runtime libraries."""

import argparse
import zipfile
from pathlib import Path


def check(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        extensions = [
            name
            for name in names
            if name.startswith("cpo3/bindings.") and name.endswith((".so", ".pyd"))
        ]
        assert len(extensions) == 1, f"{path}: expected one native extension"
        native = [
            name
            for name in names
            if ".so" in Path(name).name or name.endswith((".dylib", ".dll", ".pyd"))
        ]
        assert native == extensions, f"{path}: unexpected bundled libraries {native}"
    print(f"Checked {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path, nargs="+")
    args = parser.parse_args()
    for wheel in args.wheels:
        check(wheel)
