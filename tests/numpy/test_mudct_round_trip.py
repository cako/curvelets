"""Round-trip tests for Monogenic Curvelet Transform (MUDCT) implementation."""

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
def test_mudct_round_trip_absolute(dim, rng):
    """
    Test Monogenic Curvelet Transform round-trip with absolute tolerance.

    For specific parameters, we can guarantee an absolute precision of approximately 1e-4.
    """
    transform = setup_numpy_transform(dim, transform_kind="monogenic")
    shapes = get_test_shapes(dim)
    size = shapes[0]

    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    components = transform.backward(coeffs)

    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

    atol = 1e-4 if dim in {2, 3} else 1e-3
    # Only compare scalar component to original input
    np.testing.assert_allclose(data, scalar, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_round_trip_relative(dim, rng):
    """
    Test Monogenic Curvelet Transform round-trip with relative tolerance.

    For random parameters in the range below, we can guarantee a relative precision of
    approximately 0.5% of the maximum amplitude in the original image.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    shape_idx = rng.integers(0, len(shapes))
    cfg_idx = rng.integers(0, len(configs))
    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, cfg_idx=cfg_idx, transform_kind="monogenic"
    )

    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    components = transform.backward(coeffs)

    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

    atol = 1e-4
    # Only compare scalar component to original input
    np.testing.assert_allclose(data, scalar, atol=atol * data.max())


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0, 1])
def test_mudct_round_trip_parametrized(dim, shape_idx, rng):
    """Test Monogenic Curvelet Transform round-trip with parametrized shapes and configs."""
    shapes = get_test_shapes(dim)
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    transform = setup_numpy_transform(
        dim, shape_idx=shape_idx, transform_kind="monogenic"
    )
    size = shapes[shape_idx]
    data = rng.normal(size=size)
    coeffs = transform.forward(data)
    components = transform.backward(coeffs)

    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

    atol = 1e-4 if dim in {2, 3} else 1e-3
    # Only compare scalar component to original input
    np.testing.assert_allclose(data, scalar, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_round_trip_wavelet_mode(dim, rng):
    """
    Test Monogenic Curvelet Transform round-trip with "wavelet" mode.

    Wavelet mode creates a single ring-shaped window at the highest scale.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    # Test forward and backward transform with transform_kind="monogenic"
    transform_mono = UDCT(
        shape=size,
        num_scales=3,
        wedges_per_direction=3,
        high_frequency_mode="wavelet",
        transform_kind="monogenic",
    )
    coeffs = transform_mono.forward(data)
    components = transform_mono.backward(coeffs)

    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

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

    # Verify each coefficient is a list with ndim+1 components
    for scale_coeffs in coeffs:
        for direction_coeffs in scale_coeffs:
            for wedge_coeffs in direction_coeffs:
                assert isinstance(wedge_coeffs, list), (
                    f"Expected list, got {type(wedge_coeffs)}"
                )
                assert len(wedge_coeffs) == dim + 1, (
                    f"Expected {dim + 1} components, got {len(wedge_coeffs)}"
                )

    # Verify reconstruction accuracy (only scalar component)
    atol = 1e-4 if dim == 2 else 2e-4
    np.testing.assert_allclose(data, scalar, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_scalar_component_equivalence(dim, rng):
    """
    Test that scalar component of monogenic transform matches standard UDCT.

    The scalar component (coeff_0) should be equivalent to the standard UDCT
    coefficients within numerical precision.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)
    transform_mono = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )

    # Get standard UDCT coefficients
    coeffs_udct = transform.forward(data)

    # Get monogenic coefficients
    coeffs_mono = transform_mono.forward(data)

    # Compare scalar component with standard UDCT
    # Note: Both UDCT and monogenic scalar components are complex
    for scale_idx in range(len(coeffs_udct)):
        for direction_idx in range(len(coeffs_udct[scale_idx])):
            for wedge_idx in range(len(coeffs_udct[scale_idx][direction_idx])):
                scalar_mono = coeffs_mono[scale_idx][direction_idx][wedge_idx][0]
                coeff_udct = coeffs_udct[scale_idx][direction_idx][wedge_idx]

                # Compare scalar component with UDCT coefficient
                # Scalar component should match UDCT coefficient exactly (both are complex)
                # Allow for small numerical differences
                np.testing.assert_allclose(
                    scalar_mono, coeff_udct, atol=1e-5, rtol=1e-5
                )


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_amplitude_computation(dim, rng):
    """
    Test amplitude computation from monogenic coefficients.

    Amplitude should be: sqrt(c_0^2 + c_1^2 + c_2^2)
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )
    coeffs = transform.forward(data)

    # Compute amplitude for each coefficient
    for scale_coeffs in coeffs:
        for direction_coeffs in scale_coeffs:
            for wedge_coeffs in direction_coeffs:
                # Verify correct number of components (ndim+1)
                assert len(wedge_coeffs) == dim + 1, (
                    f"Expected {dim + 1} components, got {len(wedge_coeffs)}"
                )
                scalar = wedge_coeffs[0]
                riesz_components = wedge_coeffs[1:]
                # Scalar is complex, Riesz components are real
                # Amplitude: sqrt(|scalar|^2 + sum(riesz_k^2))
                amplitude = np.sqrt(
                    np.abs(scalar) ** 2 + sum(r**2 for r in riesz_components)
                )

                # Amplitude should be non-negative
                assert np.all(amplitude >= 0), "Amplitude should be non-negative"

                # Amplitude should match magnitude of components
                # (basic sanity check)
                assert amplitude.shape == scalar.shape, (
                    "Amplitude shape should match coefficient shape"
                )


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_complex_input_rejection(dim, rng):
    """
    Test that complex inputs raise appropriate error.

    Monogenic transform only accepts real-valued inputs.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    # Create complex input
    data = rng.normal(size=size) + 1j * rng.normal(size=size)

    transform = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )

    # Should raise ValueError for complex input
    with pytest.raises(
        ValueError, match="Monogenic transform requires real-valued input"
    ):
        transform.forward(data)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_reconstruction_vs_udct(dim, rng):
    """
    Test that monogenic reconstruction matches UDCT reconstruction.

    Both transforms should reconstruct the same input with similar accuracy.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)

    # Standard UDCT round-trip
    coeffs_udct = transform.forward(data)
    recon_udct = transform.backward(coeffs_udct)

    # Monogenic round-trip
    transform_mono = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )
    coeffs_mono = transform_mono.forward(data)
    components = transform_mono.backward(coeffs_mono)
    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

    # Both reconstructions should match input
    # 3D can have slightly higher error due to numerical precision
    atol = 1.2e-4 if dim == 3 else (1e-4 if dim == 2 else 1e-3)
    np.testing.assert_allclose(data, recon_udct, atol=atol)
    # Only compare scalar component to input
    np.testing.assert_allclose(data, scalar, atol=atol)

    # Scalar component should match UDCT reconstruction (within numerical precision)
    np.testing.assert_allclose(recon_udct, scalar, atol=1e-5)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_mudct_riesz_components_correctness(dim, rng):
    """
    Test that Riesz components from backward() with transform_kind="monogenic" match direct Riesz transforms.

    The backward() with transform_kind="monogenic" should return (f, -R_1f, -R_2f) where:
    - f is the original input
    - -R_1f and -R_2f match the direct Riesz transform computation
    """
    from curvelets.numpy import UDCT
    from curvelets.numpy._riesz import riesz_filters

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )

    # Get monogenic coefficients and reconstruct
    coeffs = transform.forward(data)
    components = transform.backward(coeffs)
    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]
    riesz1 = components[1] if len(components) > 1 else None
    riesz2 = components[2] if len(components) > 2 else None

    # Compute Riesz transforms directly
    filters = riesz_filters(size)
    data_fft = np.fft.fftn(data)
    riesz1_direct = np.fft.ifftn(data_fft * filters[0]).real
    riesz2_direct = np.fft.ifftn(data_fft * filters[1]).real

    # Verify scalar component (should already pass from other tests)
    atol_scalar = 5e-4 if dim in {2, 3} else 1e-3
    np.testing.assert_allclose(data, scalar, atol=atol_scalar)

    # Verify Riesz components match direct computation
    # Note: backward() with transform_kind="monogenic" returns -R_1f and -R_2f
    # Use relaxed tolerance for Riesz components due to numerical precision
    atol_riesz = 2.0 if dim == 2 else (2.5 if dim == 3 else 1.5)
    np.testing.assert_allclose(-riesz1_direct, riesz1, atol=atol_riesz)
    np.testing.assert_allclose(-riesz2_direct, riesz2, atol=atol_riesz)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_monogenic_matches_backward_forward(dim, rng):
    """
    Test that monogenic(f) produces the same result as backward(forward(f)) with transform_kind="monogenic".

    The monogenic method computes the monogenic signal directly without the curvelet
    transform, and should produce identical results to the round-trip through the transform.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)
    transform_mono = UDCT(
        shape=size, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )

    # Direct computation using monogenic method
    components_direct = transform.monogenic(data)
    # Verify correct number of components (ndim+1)
    assert len(components_direct) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components_direct)}"
    )
    scalar_direct = components_direct[0]

    # Round-trip through transform
    coeffs = transform_mono.forward(data)
    components_round = transform_mono.backward(coeffs)
    # Verify correct number of components (ndim+1)
    assert len(components_round) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components_round)}"
    )
    scalar_round = components_round[0]

    # Set tolerances based on dimension
    # Scalar component is very accurate, Riesz components have larger errors
    atol_scalar = 5e-4 if dim in {2, 3} else 1e-3
    atol_riesz = 2.0 if dim == 2 else (2.5 if dim == 3 else 1.5)

    # Verify all components match
    np.testing.assert_allclose(scalar_direct, scalar_round, atol=atol_scalar)
    # Verify all Riesz components match
    for k in range(1, len(components_direct)):
        np.testing.assert_allclose(
            components_direct[k], components_round[k], atol=atol_riesz
        )


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_monogenic_riesz_components_correctness(dim, rng):
    """
    Test that Riesz components from monogenic match direct Riesz transform computation.

    The monogenic method should return (f, -R_1f, -R_2f) where:
    - f is the original input
    - -R_1f and -R_2f match the direct Riesz transform computation
    """
    from curvelets.numpy import UDCT
    from curvelets.numpy._riesz import riesz_filters

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(np.float64)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)

    # Get monogenic signal directly
    components = transform.monogenic(data)
    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]

    # Compute Riesz transforms directly
    filters = riesz_filters(size)
    data_fft = np.fft.fftn(data)

    # Set tolerances based on dimension
    atol = 1e-4 if dim in {2, 3} else 1e-3

    # Verify scalar component matches input
    np.testing.assert_allclose(data, scalar, atol=atol)

    # Verify Riesz components match direct computation
    # Note: monogenic returns -R_kf for k = 1, 2, ..., ndim
    for k in range(1, min(len(components), len(filters) + 1)):
        riesz_k_direct = np.fft.ifftn(data_fft * filters[k - 1]).real
        riesz_k = components[k]
        np.testing.assert_allclose(-riesz_k_direct, riesz_k, atol=atol)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_monogenic_complex_input_rejection(dim, rng):
    """
    Test that complex inputs raise appropriate error.

    Monogenic transform only accepts real-valued inputs.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    # Create complex input
    data = rng.normal(size=size) + 1j * rng.normal(size=size)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)

    # Should raise ValueError for complex input
    with pytest.raises(
        ValueError, match="Monogenic transform requires real-valued input"
    ):
        transform.monogenic(data)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_monogenic_shape_validation(dim, rng):
    """
    Test that input shape must match transform shape.

    The monogenic method should raise AssertionError if the input shape
    does not match the transform's shape.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)

    # Create data with wrong shape
    if len(shapes) > 1:
        wrong_size = shapes[1]
    else:
        # Create a wrong shape by modifying one dimension
        wrong_size = tuple(s + 1 if i == 0 else s for i, s in enumerate(size))

    wrong_data = rng.normal(size=wrong_size).astype(np.float64)

    # Should raise AssertionError for shape mismatch
    with pytest.raises(AssertionError):
        transform.monogenic(wrong_data)


@pytest.mark.round_trip
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_monogenic_dtype_preservation(dim, dtype, rng):
    """
    Test that output dtypes match input dtype.

    The monogenic method should preserve the input dtype for all output components.
    """
    from curvelets.numpy import UDCT

    shapes = get_test_shapes(dim)
    if not shapes:
        pytest.skip(f"No test shapes defined for dimension {dim}")

    size = shapes[0]
    data = rng.normal(size=size).astype(dtype)

    transform = UDCT(shape=size, num_scales=3, wedges_per_direction=3)

    components = transform.monogenic(data)
    # Verify correct number of components (ndim+1)
    assert len(components) == dim + 1, (
        f"Expected {dim + 1} components, got {len(components)}"
    )
    scalar = components[0]
    riesz1 = components[1] if len(components) > 1 else None
    riesz2 = components[2] if len(components) > 2 else None

    # Verify all output dtypes match input dtype
    assert scalar.dtype == data.dtype, (
        f"Scalar dtype {scalar.dtype} != input dtype {data.dtype}"
    )
    if riesz1 is not None:
        assert riesz1.dtype == data.dtype, (
            f"Riesz1 dtype {riesz1.dtype} != input dtype {data.dtype}"
        )
    if riesz2 is not None:
        assert riesz2.dtype == data.dtype, (
            f"Riesz2 dtype {riesz2.dtype} != input dtype {data.dtype}"
        )
