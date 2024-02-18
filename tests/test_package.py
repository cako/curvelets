from __future__ import annotations

import importlib.metadata

import curvelets as m


def test_version():
    assert importlib.metadata.version("curvelets") == m.__version__
