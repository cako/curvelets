"""Round-trip tests for ucurv implementation only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import get_test_configs, get_test_shapes, setup_ucurv_transform


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_ucurv_round_trip_absolute(dim, rng):
    """
    Test ucurv implementation round-trip with absolute tolerance.

    Note: ucurv uses hardcoded alpha=0.1 and r values, so we cannot control these parameters.
    """
    transform = setup_ucurv_transform(dim)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-20
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_ucurv_round_trip_relative(dim, rng):
    """
    Test ucurv implementation round-trip with relative tolerance.

    Note: ucurv uses hardcoded alpha=0.1 and r values, so we cannot control these parameters.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_ucurv_transform(dim, shape_idx=shape_idx, cfg_idx=cfg_idx)

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)
    atol = 1e-8 if dim == 2 else 0.005
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0, 1])
def test_ucurv_round_trip_parametrized(dim, shape_idx, rng):
    """Test ucurv implementation round-trip with parametrized shapes and configs."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_ucurv_transform(dim, shape_idx=shape_idx)
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-8
    np.testing.assert_allclose(data, recon, atol=atol * data.max())
