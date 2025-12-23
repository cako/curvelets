"""Comprehensive comparison tests between original and refactored SimpleUDCT implementations."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import SimpleUDCT as SimpleUDCT_Original
from curvelets.numpy_refactor import UDCT as SimpleUDCT_Refactored

# Powers of 2 shapes
POWER_OF_2_SHAPES_2D = [(32, 32), (64, 64), (128, 128), (256, 256)]
POWER_OF_2_SHAPES_3D = [(32, 32, 32), (64, 64, 64)]
POWER_OF_2_SHAPES_4D = [(16, 16, 16, 16)]

# Non-powers of 2 shapes
NON_POWER_OF_2_SHAPES_2D = [
    (31, 31),
    (33, 33),
    (48, 48),
    (50, 50),
    (63, 63),
    (65, 65),
    (72, 72),
    (96, 96),
    (100, 100),
    (127, 127),
    (64, 32),  # Mixed dimensions
    (128, 64),
    (100, 50),
    (72, 48),
]
NON_POWER_OF_2_SHAPES_3D = [(33, 33, 33), (50, 50, 50), (48, 48, 48)]
NON_POWER_OF_2_SHAPES_4D = [(48, 48, 48, 48)]

ALL_SHAPES_2D = POWER_OF_2_SHAPES_2D + NON_POWER_OF_2_SHAPES_2D
ALL_SHAPES_3D = POWER_OF_2_SHAPES_3D + NON_POWER_OF_2_SHAPES_3D
ALL_SHAPES_4D = POWER_OF_2_SHAPES_4D + NON_POWER_OF_2_SHAPES_4D


def _compare_coefficients(
    coeffs_orig: list, coeffs_ref: list, rtol: float = 1e-5, atol: float = 1e-7
) -> None:
    """
    Compare coefficient structures and values between original and refactored versions.

    Parameters
    ----------
    coeffs_orig : list
        Coefficients from original implementation.
    coeffs_ref : list
        Coefficients from refactored implementation.
    rtol : float, optional
        Relative tolerance for comparison. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for comparison. Default is 1e-7.
    """
    assert len(coeffs_orig) == len(coeffs_ref), "Number of scales must match"
    for ires in range(len(coeffs_orig)):
        assert len(coeffs_orig[ires]) == len(coeffs_ref[ires]), (
            f"Number of directions at scale {ires} must match"
        )
        for idir in range(len(coeffs_orig[ires])):
            assert len(coeffs_orig[ires][idir]) == len(coeffs_ref[ires][idir]), (
                f"Number of wedges at scale {ires}, direction {idir} must match"
            )
            for iang in range(len(coeffs_orig[ires][idir])):
                np.testing.assert_allclose(
                    coeffs_orig[ires][idir][iang],
                    coeffs_ref[ires][idir][iang],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Coef mismatch at scale {ires}, dir {idir}, wedge {iang}",
                )


@pytest.mark.slow
@pytest.mark.parametrize("shape", ALL_SHAPES_2D)
@pytest.mark.parametrize("nscales", [2, 3, 4])
@pytest.mark.parametrize("nbands_per_direction", [3, 4, 5])
@pytest.mark.parametrize("high", ["curvelet", "wavelet"])
@pytest.mark.parametrize("complex_transform", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_forward_coefficients_match(
    shape, nscales, nbands_per_direction, high, complex_transform, dtype, rng
):
    """
    Test that forward transform coefficients match exactly.

    Parameters
    ----------
    shape : tuple
        Shape of input array.
    nscales : int
        Number of scales.
    nbands_per_direction : int
        Number of bands per direction.
    high : str
        High frequency mode ("curvelet" or "wavelet").
    complex_transform : bool
        Whether to use complex transform.
    dtype : numpy dtype
        Data type of input array.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    # Skip wavelet mode if nscales < 2
    if high == "wavelet" and nscales < 2:
        pytest.skip("Wavelet mode requires nscales >= 2")

    # Skip complex dtypes if complex_transform is False (for now, test separately)
    if not complex_transform and np.iscomplexobj(np.array([1], dtype=dtype)):
        pytest.skip("Complex dtype requires complex_transform=True")

    try:
        # Create both transforms
        transform_orig = SimpleUDCT_Original(
            shape=shape,
            nscales=nscales,
            nbands_per_direction=nbands_per_direction,
            high=high,
            complex=complex_transform,
        )
        transform_ref = SimpleUDCT_Refactored(
            shape=shape,
            num_scales=nscales,
            wedges_per_direction=nbands_per_direction,
            high_frequency_mode=high,
            use_complex_transform=complex_transform,
        )

        # Generate test data
        if np.iscomplexobj(np.array([1], dtype=dtype)):
            data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
        else:
            data = rng.normal(size=shape).astype(dtype)

        # Compute forward transforms
        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        # Compare coefficients
        _compare_coefficients(coeffs_orig, coeffs_ref)

    except (ValueError, AssertionError) as e:
        # If original raises an error, refactored should raise the same error
        with pytest.raises(type(e)):
            transform_ref = SimpleUDCT_Refactored(
                shape=shape,
                num_scales=nscales,
                wedges_per_direction=nbands_per_direction,
                high_frequency_mode=high,
                use_complex_transform=complex_transform,
            )
            if np.iscomplexobj(np.array([1], dtype=dtype)):
                data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
                    dtype
                )
            else:
                data = rng.normal(size=shape).astype(dtype)
            transform_ref.forward(data)


@pytest.mark.slow
@pytest.mark.parametrize("shape", ALL_SHAPES_2D)
@pytest.mark.parametrize("nscales", [2, 3])
@pytest.mark.parametrize("nbands_per_direction", [3, 4])
@pytest.mark.parametrize("high", ["curvelet", "wavelet"])
@pytest.mark.parametrize("complex_transform", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reconstruction_match(
    shape, nscales, nbands_per_direction, high, complex_transform, dtype, rng
):
    """
    Test that backward transform reconstruction matches.

    Parameters
    ----------
    shape : tuple
        Shape of input array.
    nscales : int
        Number of scales.
    nbands_per_direction : int
        Number of bands per direction.
    high : str
        High frequency mode ("curvelet" or "wavelet").
    complex_transform : bool
        Whether to use complex transform.
    dtype : numpy dtype
        Data type of input array.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    # Skip wavelet mode if nscales < 2
    if high == "wavelet" and nscales < 2:
        pytest.skip("Wavelet mode requires nscales >= 2")

    try:
        # Create both transforms
        transform_orig = SimpleUDCT_Original(
            shape=shape,
            nscales=nscales,
            nbands_per_direction=nbands_per_direction,
            high=high,
            complex=complex_transform,
        )
        transform_ref = SimpleUDCT_Refactored(
            shape=shape,
            num_scales=nscales,
            wedges_per_direction=nbands_per_direction,
            high_frequency_mode=high,
            use_complex_transform=complex_transform,
        )

        # Generate test data
        data = rng.normal(size=shape).astype(dtype)

        # Compute forward and backward transforms
        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        recon_orig = transform_orig.backward(coeffs_orig)
        recon_ref = transform_ref.backward(coeffs_ref)

        # Compare reconstructions
        np.testing.assert_allclose(
            recon_orig,
            recon_ref,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Reconstruction mismatch between original and refactored",
        )

        # Compare round-trip errors
        error_orig = np.abs(data - recon_orig).max()
        error_ref = np.abs(data - recon_ref).max()
        # Round-trip errors should be similar (within factor of 2)
        assert abs(error_orig - error_ref) < max(error_orig, error_ref) * 0.1, (
            f"Round-trip errors differ significantly: orig={error_orig}, ref={error_ref}"
        )

    except (ValueError, AssertionError) as e:
        # If original raises an error, refactored should raise the same error
        with pytest.raises(type(e)):
            transform_ref = SimpleUDCT_Refactored(
                shape=shape,
                num_scales=nscales,
                wedges_per_direction=nbands_per_direction,
                high_frequency_mode=high,
                use_complex_transform=complex_transform,
            )
            data = rng.normal(size=shape).astype(dtype)
            coeffs_ref = transform_ref.forward(data)
            transform_ref.backward(coeffs_ref)


@pytest.mark.slow
@pytest.mark.parametrize("shape", POWER_OF_2_SHAPES_2D[:2])  # Smaller subset for speed
@pytest.mark.parametrize("nscales", [2, 3])
@pytest.mark.parametrize("nbands_per_direction", [3, 4])
@pytest.mark.parametrize("alpha", [None, 0.15, 0.3, 0.5])
@pytest.mark.parametrize("winthresh", [1e-5, 1e-6, 1e-4])
def test_parameter_combinations(
    shape, nscales, nbands_per_direction, alpha, winthresh, rng
):
    """
    Test various parameter combinations.

    Parameters
    ----------
    shape : tuple
        Shape of input array.
    nscales : int
        Number of scales.
    nbands_per_direction : int
        Number of bands per direction.
    alpha : float or None
        Window overlap parameter.
    winthresh : float
        Window threshold.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    try:
        transform_orig = SimpleUDCT_Original(
            shape=shape,
            nscales=nscales,
            nbands_per_direction=nbands_per_direction,
            alpha=alpha,
            winthresh=winthresh,
        )
        transform_ref = SimpleUDCT_Refactored(
            shape=shape,
            num_scales=nscales,
            wedges_per_direction=nbands_per_direction,
            window_overlap=alpha,
            window_threshold=winthresh,
        )

        data = rng.normal(size=shape).astype(np.float64)

        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        _compare_coefficients(coeffs_orig, coeffs_ref)

        recon_orig = transform_orig.backward(coeffs_orig)
        recon_ref = transform_ref.backward(coeffs_ref)

        np.testing.assert_allclose(recon_orig, recon_ref, rtol=1e-10, atol=1e-12)

    except (ValueError, AssertionError) as e:
        with pytest.raises(type(e)):
            transform_ref = SimpleUDCT_Refactored(
                shape=shape,
                num_scales=nscales,
                wedges_per_direction=nbands_per_direction,
                window_overlap=alpha,
                window_threshold=winthresh,
            )
            data = rng.normal(size=shape).astype(np.float64)
            transform_ref.forward(data)


@pytest.mark.parametrize("shape", [(8, 8), (16, 16), (32, 32)])  # Small shapes
@pytest.mark.parametrize("nscales", [2, 3])
def test_edge_cases_small_arrays(shape, nscales, rng):
    """Test edge cases with very small arrays."""
    try:
        transform_orig = SimpleUDCT_Original(
            shape=shape, nscales=nscales, nbands_per_direction=3
        )
        transform_ref = SimpleUDCT_Refactored(
            shape=shape, num_scales=nscales, wedges_per_direction=3
        )

        data = rng.normal(size=shape).astype(np.float64)

        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        _compare_coefficients(coeffs_orig, coeffs_ref)

        recon_orig = transform_orig.backward(coeffs_orig)
        recon_ref = transform_ref.backward(coeffs_ref)

        np.testing.assert_allclose(recon_orig, recon_ref, rtol=1e-10, atol=1e-12)

    except (ValueError, AssertionError) as e:
        with pytest.raises(type(e)):
            transform_ref = SimpleUDCT_Refactored(
                shape=shape, num_scales=nscales, wedges_per_direction=3
            )
            data = rng.normal(size=shape).astype(np.float64)
            transform_ref.forward(data)


def test_error_cases_match():
    """Test that error cases raise the same errors in both versions."""
    # Test invalid nscales
    with pytest.raises(AssertionError):
        SimpleUDCT_Original(shape=(64, 64), nscales=1, nbands_per_direction=3)

    with pytest.raises(ValueError):
        SimpleUDCT_Refactored(shape=(64, 64), num_scales=1, wedges_per_direction=3)

    # Test invalid nbands_per_direction
    with pytest.raises(AssertionError):
        SimpleUDCT_Original(shape=(64, 64), nscales=2, nbands_per_direction=2)

    with pytest.raises(ValueError):
        SimpleUDCT_Refactored(shape=(64, 64), num_scales=2, wedges_per_direction=2)

    # Test wavelet mode with insufficient scales
    with pytest.raises(ValueError, match="Wavelet mode requires"):
        SimpleUDCT_Original(
            shape=(64, 64), nscales=2, nbands_per_direction=3, high="wavelet"
        )

    with pytest.raises(ValueError, match="Wavelet mode requires"):
        SimpleUDCT_Refactored(
            shape=(64, 64),
            num_scales=2,
            wedges_per_direction=3,
            high_frequency_mode="wavelet",
        )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("complex_transform", [True, False])
def test_complex_input_arrays(shape, complex_transform, rng):
    """
    Test with complex-valued input arrays.

    Parameters
    ----------
    shape : tuple
        Shape of input array.
    complex_transform : bool
        Whether to use complex transform.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    if not complex_transform:
        pytest.skip("Complex input arrays require complex_transform=True")

    try:
        transform_orig = SimpleUDCT_Original(
            shape=shape, nscales=3, nbands_per_direction=3, complex=True
        )
        transform_ref = SimpleUDCT_Refactored(
            shape=shape,
            num_scales=3,
            wedges_per_direction=3,
            use_complex_transform=True,
        )

        # Create complex-valued input
        data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        )

        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        _compare_coefficients(coeffs_orig, coeffs_ref)

        recon_orig = transform_orig.backward(coeffs_orig)
        recon_ref = transform_ref.backward(coeffs_ref)

        # Both should be complex
        assert np.iscomplexobj(recon_orig), "Original should return complex output"
        assert np.iscomplexobj(recon_ref), "Refactored should return complex output"

        np.testing.assert_allclose(recon_orig, recon_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(data, recon_orig, atol=1e-4)
        np.testing.assert_allclose(data, recon_ref, atol=1e-4)

    except (ValueError, AssertionError) as e:
        with pytest.raises(type(e)):
            transform_ref = SimpleUDCT_Refactored(
                shape=shape,
                num_scales=3,
                wedges_per_direction=3,
                use_complex_transform=True,
            )
            data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
                np.complex128
            )
            transform_ref.forward(data)
