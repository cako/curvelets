"""Round-trip tests for NumPy UDCT implementation only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import get_test_configs, get_test_shapes, setup_numpy_transform


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_absolute(dim, rng):
    """
    Test NumPy implementation round-trip with absolute tolerance.

    For specific parameters, we can guarantee an absolute precision of approximately 1e-4.
    """
    transform = setup_numpy_transform(dim)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # 4D tests need slightly relaxed tolerance due to numerical precision
    atol = 2e-4 if dim == 4 else 1e-4
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_relative(dim, rng):
    """
    Test NumPy implementation round-trip with relative tolerance.

    For random parameters in the range below, we can guarantee a relative precision of
    approximately 0.5% of the maximum amplitude in the original image.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_numpy_transform(dim, shape_idx=shape_idx, cfg_idx=cfg_idx)

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    np.testing.assert_allclose(data, recon, atol=0.005 * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0, 1])
def test_numpy_round_trip_parametrized(dim, shape_idx, rng):
    """Test NumPy implementation round-trip with parametrized shapes and configs."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_numpy_transform(dim, shape_idx=shape_idx)
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    np.testing.assert_allclose(data, recon, atol=1e-3)
