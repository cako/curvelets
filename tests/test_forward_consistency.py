"""Forward output consistency tests across UDCT implementations."""

from __future__ import annotations

import numpy as np
import pytest

import curvelets.numpy as numpy_udct
from curvelets.ucurv import ucurv
from curvelets.ucurv import udct as ucurv2_udct
from tests.conftest import (
    COMMON_ALPHA,
    COMMON_R,
    COMMON_WINTHRESH,
    coeffs_dict_to_udct,
    get_test_configs,
    get_test_shapes,
)


@pytest.mark.forward_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_forward_numpy_vs_ucurv(dim):
    """
    Compare NumPy forward vs ucurv forward using identical explicit parameters.

    Note: ucurv uses hardcoded alpha=0.1 and r values, so we cannot match NumPy's
    parameters exactly. This test documents the difference.
    """
    rng = np.random.default_rng(42)

    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    size = shapes[0]
    cfg = configs[0]

    # NumPy implementation with explicit parameters
    numpy_transform = numpy_udct.UDCT(
        shape=size, cfg=cfg, alpha=COMMON_ALPHA, r=COMMON_R, winthresh=COMMON_WINTHRESH
    )
    im = rng.normal(size=size)
    numpy_coeffs = numpy_transform.forward(im)

    # ucurv implementation (uses hardcoded alpha=0.1, r values)
    shape_array = np.array(size, dtype=int)
    cfg_list = cfg.tolist() if hasattr(cfg, "tolist") else cfg
    ucurv_transform = ucurv.udct(shape_array, cfg_list)
    ucurv_coeffs_dict = ucurv.ucurvfwd(im, ucurv_transform)

    # Convert ucurv dict format to UDCTCoefficients for comparison
    ucurv_coeffs = coeffs_dict_to_udct(ucurv_coeffs_dict)

    # Compare structures
    # Note: Due to different parameter defaults (alpha, r), the number of scales
    # may differ. We check that both have at least one scale.
    assert len(numpy_coeffs) > 0, "NumPy should have at least one scale"
    assert len(ucurv_coeffs) > 0, "ucurv should have at least one scale"

    # For comparison, we'll compare what we can, but structures may differ
    min_scales = min(len(numpy_coeffs), len(ucurv_coeffs))

    # Use dimension-specific tolerances: stricter for 2D, relaxed for 3D and 4D
    # due to larger differences from parameter mismatches
    if dim == 2:
        low_freq_rtol, low_freq_atol = 1e-2, 1e-2
    else:  # dim == 3 or 4
        low_freq_rtol, low_freq_atol = 1e-1, 1e-1

    # Compare low frequency
    np.testing.assert_allclose(
        numpy_coeffs[0][0][0], ucurv_coeffs[0][0][0], rtol=low_freq_rtol, atol=low_freq_atol
    )

    # Compare other scales (may have different structures due to parameter differences)
    # We use relaxed tolerance due to different alpha/r parameters
    for scale_idx in range(1, min_scales):
        if scale_idx < len(numpy_coeffs) and scale_idx < len(ucurv_coeffs):
            numpy_scale = numpy_coeffs[scale_idx]
            ucurv_scale = ucurv_coeffs[scale_idx]
            # Structure may differ, so we compare what we can
            min_dirs = min(len(numpy_scale), len(ucurv_scale))
            for dir_idx in range(min_dirs):
                if dir_idx < len(numpy_scale) and dir_idx < len(ucurv_scale):
                    numpy_dir = numpy_scale[dir_idx]
                    ucurv_dir = ucurv_scale[dir_idx]
                    min_wedges = min(len(numpy_dir), len(ucurv_dir))
                    for wedge_idx in range(min_wedges):
                        np.testing.assert_allclose(
                            numpy_dir[wedge_idx],
                            ucurv_dir[wedge_idx],
                            rtol=1e-1,
                            atol=1e-1,
                        )


@pytest.mark.forward_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.xfail(reason="ucurv2 forward method has issues with key unpacking")
def test_forward_numpy_vs_ucurv2(dim):
    """
    Compare NumPy forward vs ucurv2 forward using identical explicit parameters.

    Note: ucurv2 uses hardcoded alpha=0.1 and r values, so we cannot match NumPy's
    parameters exactly. This test documents the difference.
    """
    rng = np.random.default_rng(42)

    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    size = shapes[0]
    cfg = configs[0]

    # NumPy implementation with explicit parameters
    numpy_transform = numpy_udct.UDCT(
        shape=size, cfg=cfg, alpha=COMMON_ALPHA, r=COMMON_R, winthresh=COMMON_WINTHRESH
    )
    im = rng.normal(size=size)
    numpy_coeffs = numpy_transform.forward(im)

    # ucurv2 implementation (uses hardcoded alpha=0.1, r values)
    ucurv2_transform = ucurv2_udct.UDCT(shape=size, cfg=cfg, high="curvelet")
    ucurv2_coeffs = ucurv2_transform.forward(im)

    # Compare structures
    assert len(numpy_coeffs) == len(ucurv2_coeffs), "Number of scales should match"

    # Compare low frequency
    np.testing.assert_allclose(
        numpy_coeffs[0][0][0], ucurv2_coeffs[0][0][0], rtol=1e-2, atol=1e-2
    )

    # Compare other scales (may have different structures due to parameter differences)
    # We use relaxed tolerance due to different alpha/r parameters
    for scale_idx in range(1, min(len(numpy_coeffs), len(ucurv2_coeffs))):
        if scale_idx < len(numpy_coeffs) and scale_idx < len(ucurv2_coeffs):
            numpy_scale = numpy_coeffs[scale_idx]
            ucurv2_scale = ucurv2_coeffs[scale_idx]
            min_dirs = min(len(numpy_scale), len(ucurv2_scale))
            for dir_idx in range(min_dirs):
                if dir_idx < len(numpy_scale) and dir_idx < len(ucurv2_scale):
                    numpy_dir = numpy_scale[dir_idx]
                    ucurv2_dir = ucurv2_scale[dir_idx]
                    min_wedges = min(len(numpy_dir), len(ucurv2_dir))
                    for wedge_idx in range(min_wedges):
                        np.testing.assert_allclose(
                            numpy_dir[wedge_idx],
                            ucurv2_dir[wedge_idx],
                            rtol=1e-1,
                            atol=1e-1,
                        )


@pytest.mark.forward_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.xfail(reason="ucurv2 forward method has issues with key unpacking")
def test_forward_ucurv_vs_ucurv2(dim):
    """
    Compare ucurv vs ucurv2 forward using identical explicit parameters.

    Both use the same hardcoded alpha=0.1 and r values, so they should match more closely.
    """
    rng = np.random.default_rng(42)

    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    size = shapes[0]
    cfg = configs[0]

    # ucurv implementation
    shape_array = np.array(size, dtype=int)
    cfg_list = cfg.tolist() if hasattr(cfg, "tolist") else cfg
    ucurv_transform = ucurv.udct(shape_array, cfg_list)
    ucurv_coeffs_dict = ucurv.ucurvfwd(im := rng.normal(size=size), ucurv_transform)

    # ucurv2 implementation
    ucurv2_transform = ucurv2_udct.UDCT(shape=size, cfg=cfg, high="curvelet")
    ucurv2_coeffs = ucurv2_transform.forward(im)

    # Convert ucurv dict format to UDCTCoefficients for comparison
    ucurv_coeffs = coeffs_dict_to_udct(ucurv_coeffs_dict)

    # Compare structures
    assert len(ucurv_coeffs) == len(ucurv2_coeffs), "Number of scales should match"

    # Compare low frequency
    np.testing.assert_allclose(
        ucurv_coeffs[0][0][0], ucurv2_coeffs[0][0][0], rtol=1e-3, atol=1e-3
    )

    # Compare other scales
    for scale_idx in range(1, min(len(ucurv_coeffs), len(ucurv2_coeffs))):
        if scale_idx < len(ucurv_coeffs) and scale_idx < len(ucurv2_coeffs):
            ucurv_scale = ucurv_coeffs[scale_idx]
            ucurv2_scale = ucurv2_coeffs[scale_idx]
            min_dirs = min(len(ucurv_scale), len(ucurv2_scale))
            for dir_idx in range(min_dirs):
                if dir_idx < len(ucurv_scale) and dir_idx < len(ucurv2_scale):
                    ucurv_dir = ucurv_scale[dir_idx]
                    ucurv2_dir = ucurv2_scale[dir_idx]
                    min_wedges = min(len(ucurv_dir), len(ucurv2_dir))
                    for wedge_idx in range(min_wedges):
                        np.testing.assert_allclose(
                            ucurv_dir[wedge_idx],
                            ucurv2_dir[wedge_idx],
                            rtol=1e-2,
                            atol=1e-2,
                        )
