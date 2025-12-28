"""Shared fixtures for all test modules."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return np.random.default_rng(42)
