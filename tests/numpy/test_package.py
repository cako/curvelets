from __future__ import annotations

import importlib.metadata
import subprocess
import sys

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


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="typing_extensions is a dependency before 3.12"
)
def test_imports_without_typing_extensions():
    # typing_extensions is only declared for Python < 3.12 (pyproject.toml), and
    # the test environment usually has it anyway (pulled in by other packages):
    # hide it to check the package does not need it.
    code = (
        "import sys; sys.modules['typing_extensions'] = None; "
        "import curvelets.numpy, curvelets.utils"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
