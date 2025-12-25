"""Comprehensive comparison tests between original and refactored UDCT implementations using cfg parameter."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT as UDCT_Original
from curvelets.numpy_refactor import UDCT as UDCT_Refactored
from tests.conftest import COMMON_R, get_test_configs, get_test_shapes


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


def _create_equivalent_transforms(
    cfg: np.ndarray,
    shape: tuple[int, ...],
    alpha: float = 0.15,
    winthresh: float = 1e-5,
    high: str = "curvelet",
    complex_transform: bool = False,
    r: tuple[float, float, float, float] | None = None,
) -> tuple[UDCT_Original, UDCT_Refactored]:
    """
    Create both original and refactored UDCT instances with equivalent parameters.

    Parameters
    ----------
    cfg : np.ndarray
        Configuration array with shape (nscales, dim).
    shape : tuple[int, ...]
        Shape of the input data.
    alpha : float, optional
        Window overlap parameter. Default is 0.15.
    winthresh : float, optional
        Window threshold. Default is 1e-5.
    high : str, optional
        High frequency mode ("curvelet" or "wavelet"). Default is "curvelet".
    complex_transform : bool, optional
        Whether to use complex transform. Default is False.
    r : tuple[float, float, float, float] | None, optional
        Radial frequency parameters. Default is None (uses COMMON_R).

    Returns
    -------
    tuple[UDCT_Original, UDCT_Refactored]
        Tuple of (original_transform, refactored_transform).
    """
    if r is None:
        r = COMMON_R

    transform_orig = UDCT_Original(
        shape=shape,
        cfg=cfg,
        alpha=alpha,
        r=r,
        winthresh=winthresh,
        high=high,
        complex=complex_transform,
    )

    transform_ref = UDCT_Refactored(
        shape=shape,
        angular_wedges_config=cfg,
        window_overlap=alpha,
        radial_frequency_params=r,
        window_threshold=winthresh,
        high_frequency_mode=high,
        use_complex_transform=complex_transform,
    )

    return transform_orig, transform_ref


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("cfg_idx", [0])
@pytest.mark.parametrize("shape_idx", [0])
@pytest.mark.parametrize("alpha", [0.15])
@pytest.mark.parametrize("winthresh", [1e-5])
@pytest.mark.parametrize("high", ["curvelet", "wavelet"])
@pytest.mark.parametrize("complex_transform", [True, False])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_udct_cfg_forward_coefficients_match(
    dim, cfg_idx, shape_idx, alpha, winthresh, high, complex_transform, dtype, rng
):
    """
    Test that forward transform coefficients match between original and refactored UDCT.

    Parameters
    ----------
    dim : int
        Dimension (2 or 3).
    cfg_idx : int
        Index into get_test_configs(dim).
    shape_idx : int
        Index into get_test_shapes(dim).
    alpha : float
        Window overlap parameter.
    winthresh : float
        Window threshold.
    high : str
        High frequency mode ("curvelet" or "wavelet").
    complex_transform : bool
        Whether to use complex transform.
    dtype : numpy dtype
        Data type of input array.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    # Get test configurations
    configs = get_test_configs(dim)
    shapes = get_test_shapes(dim)

    # Skip if indices are out of range
    if cfg_idx >= len(configs):
        pytest.skip(f"Config index {cfg_idx} out of range for dimension {dim}")
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    cfg = configs[cfg_idx]
    shape = shapes[shape_idx]

    # Skip wavelet mode if nscales < 2
    if high == "wavelet" and len(cfg) < 2:
        pytest.skip("Wavelet mode requires nscales >= 2")

    # Skip complex dtypes if complex_transform is False
    if not complex_transform and np.iscomplexobj(np.array([1], dtype=dtype)):
        pytest.skip("Complex dtype requires complex_transform=True")

    # NOTE: Known bug in refactored 3D implementation:
    # Investigation shows that for 3D transforms, the refactored implementation
    # produces different window values than the original (28% of values differ
    # significantly, max diff ~1.0). The low-frequency window computation
    # matches before normalization, suggesting the issue is in:
    # 1. High-frequency window computation for 3D
    # 2. Window normalization logic for 3D
    # 3. How windows are accumulated/flipped during normalization for 3D
    # 
    # 2D transforms work correctly, indicating this is specific to 3D handling.
    # This causes coefficient mismatch and reconstruction failure (round-trip
    # error ~3.25 vs ~6e-5 for original). The bug needs to be fixed in the
    # window computation/normalization code in numpy_refactor.

    try:
        # Create both transforms
        transform_orig, transform_ref = _create_equivalent_transforms(
            cfg=cfg,
            shape=shape,
            alpha=alpha,
            winthresh=winthresh,
            high=high,
            complex_transform=complex_transform,
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
            transform_ref = UDCT_Refactored(
                shape=shape,
                angular_wedges_config=cfg,
                window_overlap=alpha,
                window_threshold=winthresh,
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


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("cfg_idx", [0])
@pytest.mark.parametrize("shape_idx", [0])
@pytest.mark.parametrize("alpha", [0.15])
@pytest.mark.parametrize("winthresh", [1e-5])
@pytest.mark.parametrize("high", ["curvelet", "wavelet"])
@pytest.mark.parametrize("complex_transform", [True, False])
@pytest.mark.parametrize("dtype", [np.float64])
def test_udct_cfg_reconstruction_match(
    dim, cfg_idx, shape_idx, alpha, winthresh, high, complex_transform, dtype, rng
):
    """
    Test that backward transform reconstruction matches between original and refactored UDCT.

    Parameters
    ----------
    dim : int
        Dimension (2 or 3).
    cfg_idx : int
        Index into get_test_configs(dim).
    shape_idx : int
        Index into get_test_shapes(dim).
    alpha : float
        Window overlap parameter.
    winthresh : float
        Window threshold.
    high : str
        High frequency mode ("curvelet" or "wavelet").
    complex_transform : bool
        Whether to use complex transform.
    dtype : numpy dtype
        Data type of input array.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    # Get test configurations
    configs = get_test_configs(dim)
    shapes = get_test_shapes(dim)

    # Skip if indices are out of range
    if cfg_idx >= len(configs):
        pytest.skip(f"Config index {cfg_idx} out of range for dimension {dim}")
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    cfg = configs[cfg_idx]
    shape = shapes[shape_idx]

    # Skip wavelet mode if nscales < 2
    if high == "wavelet" and len(cfg) < 2:
        pytest.skip("Wavelet mode requires nscales >= 2")

    # NOTE: Known bug in refactored 3D implementation:
    # See test_udct_cfg_forward_coefficients_match for detailed explanation.
    # The same window computation bug affects reconstruction.

    try:
        # Create both transforms
        transform_orig, transform_ref = _create_equivalent_transforms(
            cfg=cfg,
            shape=shape,
            alpha=alpha,
            winthresh=winthresh,
            high=high,
            complex_transform=complex_transform,
        )

        # Generate test data
        data = rng.normal(size=shape).astype(dtype)

        # Compute forward and backward transforms
        coeffs_orig = transform_orig.forward(data)
        coeffs_ref = transform_ref.forward(data)

        recon_orig = transform_orig.backward(coeffs_orig)
        recon_ref = transform_ref.backward(coeffs_ref)

        # Compare reconstructions
        # Use dtype-appropriate tolerances: float32 has ~1e-7 precision
        if dtype == np.float32:
            rtol = 1e-5
            atol = 1e-6
        else:  # float64
            rtol = 1e-10
            atol = 1e-12
        np.testing.assert_allclose(
            recon_orig,
            recon_ref,
            rtol=rtol,
            atol=atol,
            err_msg="Reconstruction mismatch between original and refactored",
        )

        # Compare round-trip errors
        error_orig = np.abs(data - recon_orig).max()
        error_ref = np.abs(data - recon_ref).max()
        # Round-trip errors should be similar
        # For float32, use more lenient tolerance due to precision limits
        # For complex transforms, errors can be slightly larger
        if dtype == np.float32:
            # Allow up to 50% difference for float32 (more lenient for complex transforms)
            max_allowed_diff = max(error_orig, error_ref) * 0.5
            # Also ensure absolute difference is reasonable (< 1e-6 for float32)
            max_abs_diff = 1e-6
        else:  # float64
            # Stricter tolerance for float64
            max_allowed_diff = max(error_orig, error_ref) * 0.1
            max_abs_diff = 1e-10
        assert abs(error_orig - error_ref) < max(max_allowed_diff, max_abs_diff), (
            f"Round-trip errors differ significantly: orig={error_orig}, ref={error_ref}"
        )

    except (ValueError, AssertionError) as e:
        # If original raises an error, refactored should raise the same error
        with pytest.raises(type(e)):
            transform_ref = UDCT_Refactored(
                shape=shape,
                angular_wedges_config=cfg,
                window_overlap=alpha,
                window_threshold=winthresh,
                high_frequency_mode=high,
                use_complex_transform=complex_transform,
            )
            data = rng.normal(size=shape).astype(dtype)
            coeffs_ref = transform_ref.forward(data)
            transform_ref.backward(coeffs_ref)


@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("cfg_idx", [0])
@pytest.mark.parametrize("shape_idx", [0])
@pytest.mark.parametrize("alpha", [0.15, 0.3])
@pytest.mark.parametrize("winthresh", [1e-5, 1e-6])
def test_udct_cfg_parameter_combinations(
    dim, cfg_idx, shape_idx, alpha, winthresh, rng
):
    """
    Test various parameter combinations between original and refactored UDCT.

    Parameters
    ----------
    dim : int
        Dimension (2).
    cfg_idx : int
        Index into get_test_configs(dim).
    shape_idx : int
        Index into get_test_shapes(dim).
    alpha : float or None
        Window overlap parameter.
    winthresh : float
        Window threshold.
    rng : numpy.random.Generator
        Random number generator fixture.
    """
    # Get test configurations
    configs = get_test_configs(dim)
    shapes = get_test_shapes(dim)

    # Skip if indices are out of range
    if cfg_idx >= len(configs):
        pytest.skip(f"Config index {cfg_idx} out of range for dimension {dim}")
    if shape_idx >= len(shapes):
        pytest.skip(f"Shape index {shape_idx} out of range for dimension {dim}")

    cfg = configs[cfg_idx]
    shape = shapes[shape_idx]

    # Use default alpha if None
    alpha_val = alpha if alpha is not None else 0.15

    try:
        transform_orig, transform_ref = _create_equivalent_transforms(
            cfg=cfg,
            shape=shape,
            alpha=alpha_val,
            winthresh=winthresh,
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
            transform_ref = UDCT_Refactored(
                shape=shape,
                angular_wedges_config=cfg,
                window_overlap=alpha_val,
                window_threshold=winthresh,
            )
            data = rng.normal(size=shape).astype(np.float64)
            transform_ref.forward(data)


def test_udct_cfg_error_cases_match():
    """Test that error cases raise the same errors in both versions."""
    shape = (64, 64)

    # Test wavelet mode with insufficient scales
    cfg_insufficient = np.array([[3, 3]])  # Only 1 scale
    with pytest.raises(ValueError, match="Wavelet mode requires"):
        UDCT_Original(shape=shape, cfg=cfg_insufficient, high="wavelet")

    with pytest.raises(ValueError, match="Wavelet mode requires"):
        UDCT_Refactored(
            shape=shape, angular_wedges_config=cfg_insufficient, high_frequency_mode="wavelet"
        )

    # Test with valid cfg for wavelet mode (should work)
    cfg_valid = np.array([[3, 3], [6, 6]])  # 2 scales
    transform_orig = UDCT_Original(shape=shape, cfg=cfg_valid, high="wavelet")
    transform_ref = UDCT_Refactored(
        shape=shape, angular_wedges_config=cfg_valid, high_frequency_mode="wavelet"
    )
    # Should not raise errors
    assert transform_orig is not None
    assert transform_ref is not None

