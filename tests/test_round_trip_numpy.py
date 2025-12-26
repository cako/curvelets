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


# ============================================================================
# Meyer mode tests
# ============================================================================


def _get_wavelet_config_idx(dim: int) -> int | None:
    """
    Get the config index for meyer mode tests.

    Meyer mode requires at least 2 scales. Returns None if no suitable config exists.

    Parameters
    ----------
    dim : int
        Dimension.

    Returns
    -------
    int | None
        Config index with at least 2 scales, or None if not available.
    """
    configs = get_test_configs(dim)
    for idx, cfg in enumerate(configs):
        if len(cfg) >= 2:  # nscales >= 2
            return idx
    return None


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "meyer"),
        (3, "meyer"),
        (4, "meyer"),
    ],
)
def test_numpy_round_trip_wavelet_absolute(dim, high, rng):
    """
    Test NumPy implementation round-trip with meyer mode using absolute tolerance.

    Meyer mode applies Meyer wavelet decomposition at the highest scale,
    with curvelet transform on the lowpass component only.
    """
    cfg_idx = _get_wavelet_config_idx(dim)
    if cfg_idx is None:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    transform = setup_numpy_transform(dim, cfg_idx=cfg_idx, high=high)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Meyer mode has slightly higher reconstruction error due to Meyer wavelet
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "meyer"),
        (3, "meyer"),
        (4, "meyer"),
    ],
)
def test_numpy_round_trip_wavelet_relative(dim, high, rng):
    """
    Test NumPy implementation round-trip with meyer mode using relative tolerance.

    Tests with random shapes and configs (that have at least 2 scales for meyer mode).
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    # Filter configs to only those with nscales >= 2 for meyer mode
    valid_cfg_indices = [idx for idx, cfg in enumerate(configs) if len(cfg) >= 2]
    if not valid_cfg_indices:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = valid_cfg_indices[rng.integers(0, len(valid_cfg_indices))]
    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, high=high
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_wavelet_num_scales_2(dim, rng):
    """
    Test NumPy implementation round-trip with meyer mode and num_scales=2.

    This test specifically verifies that num_scales=2 works with meyer mode,
    which should be equivalent to a Meyer wavelet transform (1 lowpass + 1 highpass).

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
    ...     num_scales=2,
    ...     wedges_per_direction=3,
    ...     high_frequency_mode="meyer"
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

    # Create transform with num_scales=2 and meyer mode
    transform = UDCT(
        shape=size,
        num_scales=2,
        wedges_per_direction=3,
        high_frequency_mode="meyer",
    )

    # Test forward and backward transform
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify structure: should have 2 scales (lowpass + 1 high-frequency)
    assert len(coeffs) == 2, f"Expected 2 scales, got {len(coeffs)}"

    # Verify reconstruction accuracy
    # Meyer mode has slightly higher reconstruction error due to Meyer wavelet
    atol = 1e-4 if dim == 2 else 2e-4
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
    transform = setup_numpy_transform(dim, complex=True)
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
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, complex=True
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

    transform = setup_numpy_transform(dim, shape_idx=shape_idx, complex=True)
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "meyer"),
        (3, "meyer"),
        (4, "meyer"),
    ],
)
def test_numpy_round_trip_complex_wavelet_absolute(dim, high, rng):
    """
    Test NumPy implementation round-trip with complex transform in meyer mode.

    Combines complex transform (separate +/- frequency bands) with meyer mode
    (Meyer wavelet at highest scale).
    """
    cfg_idx = _get_wavelet_config_idx(dim)
    if cfg_idx is None:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    transform = setup_numpy_transform(dim, cfg_idx=cfg_idx, high=high, complex=True)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Complex wavelet mode may have slightly higher reconstruction error
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon.real, atol=atol)


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
    ...     use_complex_transform=True
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
        use_complex_transform=True,
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
# Complex-valued input array tests (requires complex=True)
# ============================================================================


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_numpy_round_trip_complex_input_absolute(dim, rng):
    """
    Test NumPy implementation round-trip with complex-valued input arrays.

    Complex-valued inputs require complex=True to preserve both real and
    imaginary components through the round-trip.
    """
    transform = setup_numpy_transform(dim, complex=True)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    # Create complex-valued input
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify output is complex
    assert np.iscomplexobj(recon), "Output should be complex for complex=True"

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
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, complex=True
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    atol = 1e-4
    np.testing.assert_allclose(data, recon, atol=atol * np.abs(data).max())


@pytest.mark.round_trip
@pytest.mark.parametrize(
    "dim,high",
    [
        (2, "meyer"),
        (3, "meyer"),
        (4, "meyer"),
    ],
)
def test_numpy_round_trip_complex_input_wavelet(dim, high, rng):
    """
    Test NumPy round-trip with complex-valued input in meyer mode.

    Combines complex-valued input with meyer mode (Meyer wavelet at highest scale).
    """
    cfg_idx = _get_wavelet_config_idx(dim)
    if cfg_idx is None:
        pytest.skip(f"No config with nscales >= 2 available for dimension {dim}")

    transform = setup_numpy_transform(dim, cfg_idx=cfg_idx, high=high, complex=True)
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    coeffs = transform.forward(data)
    recon = transform.backward(coeffs)

    # Verify output is complex
    assert np.iscomplexobj(recon), "Output should be complex for complex=True"

    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, recon, atol=atol)
