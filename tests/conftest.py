"""Select the short benchmark suite by default, or the full suite with --full."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--full", action="store_true")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--full"):
        return
    deselected = [item for item in items if item.get_closest_marker("full")]
    items[:] = [item for item in items if not item.get_closest_marker("full")]
    config.hook.pytest_deselected(items=deselected)
