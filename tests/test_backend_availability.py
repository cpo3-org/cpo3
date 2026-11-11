"""CI backend requirements must fail rather than silently skip solver tests."""

import os

import cpo3
import pytest


@pytest.mark.parametrize(
    "backend", os.environ.get("CPO3_REQUIRED_BACKENDS", "faer").split(",")
)
def test_required_backend(backend: str) -> None:
    assert cpo3.LinearSolver(backend).is_available(), (
        f"{backend} is required in this job"
    )


@pytest.mark.parametrize(
    "backend",
    list(filter(None, os.environ.get("CPO3_UNAVAILABLE_BACKENDS", "").split(","))),
)
def test_unavailable_backend(backend: str) -> None:
    assert not cpo3.LinearSolver(backend).is_available(), (
        f"{backend} should be unavailable without its runtime"
    )
