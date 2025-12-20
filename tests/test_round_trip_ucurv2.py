"""Round-trip tests for ucurv2 implementation only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import get_test_configs, get_test_shapes, setup_ucurv2_transform


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "curvelet"),
        (3, "curvelet"),
        (4, "curvelet"),
        (2, "wavelet"),
        (3, "wavelet"),
        (4, "wavelet"),
    ],
)
def test_ucurv2_round_trip_absolute(dim, high, rng):
    """
    Test ucurv2 implementation round-trip with absolute tolerance.

    Note: ucurv2 uses hardcoded alpha=0.1 and r values, so we cannot control these parameters.
    """
    transform = setup_ucurv2_transform(dim, high=high)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-5
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "curvelet"),
        (3, "curvelet"),
        (4, "curvelet"),
        (2, "wavelet"),
        (3, "wavelet"),
        (4, "wavelet"),
    ],
)
def test_ucurv2_round_trip_relative(dim, high, rng):
    """
    Test ucurv2 implementation round-trip with relative tolerance.

    Note: ucurv2 uses hardcoded alpha=0.1 and r values, so we cannot control these parameters.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_ucurv2_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, high=high
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,shape_idx,high",
    [
        (2, 0, "curvelet"),
        (2, 1, "curvelet"),
        (3, 0, "curvelet"),
        (3, 1, "curvelet"),
        (4, 0, "curvelet"),
    ],
)
def test_ucurv2_round_trip_parametrized(dim, shape_idx, high, rng):
    """Test ucurv2 implementation round-trip with parametrized shapes and configs."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_ucurv2_transform(dim, shape_idx=shape_idx, high=high)
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol)


# ============================================================================
# Complex-valued input array tests (requires complex=True)
# ============================================================================


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2])
def test_ucurv2_round_trip_complex_input_absolute(dim, rng):
    """
    Test ucurv2 implementation round-trip with complex-valued input arrays.

    Complex-valued inputs require complex=True to preserve both real and
    imaginary components through the round-trip.

    NOTE: Only 2D is tested because ucurv2's complex mode has a bug for 3D/4D
    where the backward function tries to access keys that don't exist.
    """
    transform = setup_ucurv2_transform(dim, complex=True)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    # Create complex-valued input
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify output is complex
    assert np.iscomplexobj(recon), "Output should be complex for complex=True"

    atol = 1e-5
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2])
def test_ucurv2_round_trip_complex_input_relative(dim, rng):
    """
    Test ucurv2 round-trip with complex-valued input using relative tolerance.

    Tests with random shapes and configs.

    NOTE: Only 2D is tested because ucurv2's complex mode has a bug for 3D/4D.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_ucurv2_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, complex=True
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * np.abs(data).max())
