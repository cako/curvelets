"""Round-trip tests for NumPy UDCT implementation only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numpy.conftest import (
    get_test_configs,
    get_test_shapes,
    setup_numpy_transform,
)


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

    atol = 1e-4 if dim in {2, 3} else 1e-3
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

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


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

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_wavelet_mode(dim, rng):
    """
    Test NumPy implementation round-trip with new "wavelet" mode.

    Wavelet mode sums all windows at the highest scale into a single window
    with decimation=1 (no decimation).
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Create transform with num_scales=3 and wavelet mode
    transform = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify structure: should have 3 scales
    assert len(coeffs) == 3, f"Expected 3 scales, got {len(coeffs)}"

    # Verify highest scale has single window (1 direction, 1 wedge)
    highest_scale_idx = 2
    assert len(coeffs[highest_scale_idx]) == 1, (
        f"Expected 1 direction at highest scale, got {len(coeffs[highest_scale_idx])}"
    )
    assert len(coeffs[highest_scale_idx][0]) == 1, (
        f"Expected 1 wedge at highest scale, got {len(coeffs[highest_scale_idx][0])}"
    )

    # Verify decimation=1 (coefficient shape should match internal shape)
    highest_coeff = coeffs[highest_scale_idx][0][0]
    # For wavelet mode at highest scale, decimation=1, so shape should match parameters.shape
    expected_shape = transform.parameters.shape
    assert highest_coeff.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {highest_coeff.shape}"
    )

    # Verify windows structure
    assert len(transform.windows[highest_scale_idx]) == 1, (
        "Expected 1 direction in windows at highest scale"
    )
    assert len(transform.windows[highest_scale_idx][0]) == 1, (
        "Expected 1 wedge in windows at highest scale"
    )

    # Verify decimation ratio is 1
    assert np.all(transform.decimation_ratios[highest_scale_idx] == 1), (
        "Expected decimation=1 at highest scale"
    )

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol)


# ============================================================================
# Complex transform tests (separate +/- frequency bands)
# ============================================================================


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_absolute(dim, rng):
    """
    Test NumPy implementation round-trip with complex transform using absolute tolerance.

    Complex transform separates positive and negative frequency components
    into different bands, each scaled by sqrt(0.5).
    """
    transform = setup_numpy_transform(dim, transform_kind="complex")
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_relative(dim, rng):
    """
    Test NumPy implementation round-trip with complex transform using relative tolerance.

    Tests with random shapes and configs.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, transform_kind="complex"
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0, 1])
def test_numpy_round_trip_complex_parametrized(dim, shape_idx, rng):
    """Test NumPy implementation round-trip with complex transform and parametrized shapes."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, transform_kind="complex"
    )
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_wavelet_mode(dim, rng):
    """
    Test NumPy implementation round-trip with complex transform in "wavelet" mode.

    Combines complex transform (separate +/- frequency bands) with wavelet mode
    (single ring-shaped window at highest scale).

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
    ...     high_frequency_mode="wavelet",
    ...     transform_kind="complex"
    ... )
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon.real, atol=1e-4)
    True
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Create transform with num_scales=3, wavelet mode, and complex transform
    transform = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
        transform_kind="complex",
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify structure: should have 3 scales
    assert len(coeffs) == 3, f"Expected 3 scales, got {len(coeffs)}"

    # Verify highest scale has 2*ndim directions (ndim for positive, ndim for negative)
    highest_scale_idx = 2
    assert len(coeffs[highest_scale_idx]) == 2 * transform.parameters.ndim, (
        f"Expected {2 * transform.parameters.ndim} directions at highest scale, "
        f"got {len(coeffs[highest_scale_idx])}"
    )

    # Verify windows structure: should have 1 direction at highest scale
    assert len(transform.windows[highest_scale_idx]) == 1, (
        "Expected 1 direction in windows at highest scale"
    )

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon.real, atol=atol)


# ============================================================================
# Complex-valued input array tests (requires transform_kind="complex")
# ============================================================================


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_input_absolute(dim, rng):
    """
    Test NumPy implementation round-trip with complex-valued input arrays.

    Complex-valued inputs require transform_kind="complex" to preserve both real and
    imaginary components through the round-trip.
    """
    transform = setup_numpy_transform(dim, transform_kind="complex")
    shapes = get_test_shapes(dim)
    size = shapes[0]

    # Create complex-valued input
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify output is complex
    assert np.iscomplexobj(recon), (
        "Output should be complex for transform_kind='complex'"
    )

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_input_relative(dim, rng):
    """
    Test NumPy round-trip with complex-valued input using relative tolerance.

    Tests with random shapes and configs.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, transform_kind="complex"
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * np.abs(data).max())


# ============================================================================
# Backward transform edge cases: decimation ratio handling
# ============================================================================


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_backward_decimation_ratio_single_direction(dim, rng):
    """
    Test backward transform when decimation_ratios has shape[0] == 1.

    This occurs in "wavelet" mode at the highest scale where all directions
    share the same decimation ratio (decimation=1).

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
    ...     high_frequency_mode="wavelet"
    ... )
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Create transform with wavelet mode (triggers shape[0] == 1 at highest scale)
    transform = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
    )

    # Verify decimation_ratios at highest scale has shape[0] == 1
    highest_scale_idx = 2
    assert transform.decimation_ratios[highest_scale_idx].shape[0] == 1, (
        "Expected decimation_ratios[highest_scale_idx].shape[0] == 1 in wavelet mode"
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_backward_complex_decimation_ratio_single(dim, rng):
    """
    Test complex backward transform with single decimation ratio (shape[0] == 1).

    This tests the edge case in complex transform mode where decimation_ratios
    has shape[0] == 1, which occurs in wavelet mode at the highest scale.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
    ...     high_frequency_mode="wavelet",
    ...     transform_kind="complex"
    ... )
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon.real, atol=1e-4)
    True
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Create transform with wavelet mode and complex transform
    transform = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
        transform_kind="complex",
    )

    # Verify decimation_ratios at highest scale has shape[0] == 1
    highest_scale_idx = 2
    assert transform.decimation_ratios[highest_scale_idx].shape[0] == 1, (
        "Expected decimation_ratios[highest_scale_idx].shape[0] == 1 in wavelet mode"
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon.real, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_backward_complex_decimation_ratio_multi(dim, rng):
    """
    Test complex backward transform with multiple decimation ratios.

    This tests the normal case where decimation_ratios has shape[0] > 1,
    meaning different directions have different decimation ratios.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    rng : numpy.random.Generator
        Random number generator fixture.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import UDCT
    >>> transform = UDCT(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
    ...     transform_kind="complex"
    ... )
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon.real, atol=1e-4)
    True
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Create transform with complex transform (normal mode, not wavelet)
    transform = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind="complex",
    )

    # Verify decimation_ratios at scale 1 has shape[0] > 1 (multiple directions)
    scale_idx = 1
    assert transform.decimation_ratios[scale_idx].shape[0] > 1, (
        "Expected decimation_ratios[scale_idx].shape[0] > 1 in normal mode"
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify reconstruction accuracy
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon.real, atol=atol)
