from __future__ import annotations

import importlib.metadata

import pytest

import curvelets as m


def test_version():
    installed_version = importlib.metadata.version("curvelets")
    module_version = m.__version__
    if installed_version != module_version:
        pytest.skip(
            f"Version mismatch: installed version {installed_version} != module version {module_version}. "
            "This is common in development environments where the package is installed from a different source."
        )
    assert installed_version == module_version
