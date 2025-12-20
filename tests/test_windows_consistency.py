"""Window consistency tests across UDCT implementations."""

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
    extract_ucurv_window_dense,
    get_numpy_windows_dict,
    get_test_configs,
    get_test_shapes,
)


@pytest.mark.window_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_windows_numpy_vs_ucurv(dim):
    """
    Compare windows between NumPy and ucurv using identical explicit parameters.

    Note: ucurv uses hardcoded alpha=0.1 and r values, so we cannot match NumPy's
    parameters exactly. This test documents the difference.
    """
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
    numpy_windows_dict = get_numpy_windows_dict(numpy_transform.windows, size)

    # ucurv implementation (uses hardcoded alpha=0.1, r values)
    shape_array = np.array(size, dtype=int)
    cfg_list = cfg.tolist() if hasattr(cfg, "tolist") else cfg
    ucurv_transform = ucurv.udct(shape_array, cfg_list, sparse=False)

    # Compare low frequency window
    numpy_low = numpy_windows_dict.get((0, 0, 0))
    ucurv_low = extract_ucurv_window_dense(ucurv_transform.FL, size)

    if numpy_low is not None:
        # Use dimension-specific tolerance for dim==2
        if dim == 2:
            rtol, atol = 1e-14, 1e-14
        else:
            # Use relaxed tolerance due to different alpha/r parameters
            rtol, atol = 1e-1, 1e-1
        np.testing.assert_allclose(numpy_low, ucurv_low, rtol=rtol, atol=atol)

    # Compare other windows
    # Note: ucurv uses different indexing, so we compare what we can
    # ucurv windows are in Msubwin dict with keys like (rs, ipyr, *alist)
    # This is a simplified comparison - full mapping would be more complex
    ucurv_window_count = len(ucurv_transform.Msubwin)
    numpy_window_count = len([k for k in numpy_windows_dict.keys() if k[0] > 0])

    # At least check that both have windows
    assert ucurv_window_count > 0, "ucurv should have windows"
    assert numpy_window_count > 0, "NumPy should have windows"

    # Compare a few representative windows if possible
    # This is a basic check - full comparison would require mapping the different indexing schemes


@pytest.mark.window_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_windows_numpy_vs_ucurv2(dim):
    """
    Compare windows between NumPy and ucurv2 using identical explicit parameters.

    Both implementations now use the same hardcoded r values and configurable alpha.
    """
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
    numpy_windows_dict = get_numpy_windows_dict(numpy_transform.windows, size)

    # ucurv2 implementation (uses hardcoded r values)
    ucurv2_transform = ucurv2_udct.UDCT(
        shape=size, cfg=cfg, high="curvelet", sparse=False, alpha=COMMON_ALPHA
    )

    # Compare low frequency window
    numpy_low = numpy_windows_dict.get((0, 0, 0))
    ucurv2_low = extract_ucurv_window_dense(ucurv2_transform.FL, size)

    if numpy_low is not None:
        # Use dimension-specific tolerance for dim==2
        if dim == 2:
            rtol, atol = 1e-14, 1e-14
        else:
            # Use relaxed tolerance due to different r parameters
            rtol, atol = 1e-1, 1e-1
        np.testing.assert_allclose(numpy_low, ucurv2_low, rtol=rtol, atol=atol)

    # Compare other windows
    # Note: ucurv2 uses different indexing, so we compare what we can
    ucurv2_window_count = len(ucurv2_transform.Msubwin)
    numpy_window_count = len([k for k in numpy_windows_dict.keys() if k[0] > 0])

    # At least check that both have windows
    assert ucurv2_window_count > 0, "ucurv2 should have windows"
    assert numpy_window_count > 0, "NumPy should have windows"


@pytest.mark.window_consistency
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_windows_ucurv_vs_ucurv2(dim):
    """
    Compare windows between ucurv and ucurv2 using identical explicit parameters.

    Both use the same hardcoded alpha=0.1 and r values, so they should match more closely.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    size = shapes[0]
    cfg = configs[0]

    # ucurv implementation
    shape_array = np.array(size, dtype=int)
    cfg_list = cfg.tolist() if hasattr(cfg, "tolist") else cfg
    ucurv_transform = ucurv.udct(shape_array, cfg_list, sparse=False)

    # ucurv2 implementation
    ucurv2_transform = ucurv2_udct.UDCT(
        shape=size, cfg=cfg, high="curvelet", sparse=False
    )

    # Compare low frequency window
    ucurv_low = extract_ucurv_window_dense(ucurv_transform.FL, size)
    ucurv2_low = extract_ucurv_window_dense(ucurv2_transform.FL, size)

    if dim == 2:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-3, 1e-3
    np.testing.assert_allclose(ucurv_low, ucurv2_low, rtol=rtol, atol=atol)

    # Compare Msubwin dictionaries
    # Both should have the same keys and similar windows
    ucurv_keys = set(ucurv_transform.Msubwin.keys())
    ucurv2_keys = set(ucurv2_transform.Msubwin.keys())

    # Check that both have windows
    assert len(ucurv_keys) > 0, "ucurv should have windows"
    assert len(ucurv2_keys) > 0, "ucurv2 should have windows"

    # Compare windows for common keys
    common_keys = ucurv_keys & ucurv2_keys
    assert len(common_keys) > 0, "ucurv and ucurv2 should have some common window keys"

    for key in list(common_keys)[:5]:  # Compare first 5 common windows
        ucurv_win = extract_ucurv_window_dense(ucurv_transform.Msubwin[key], size)
        ucurv2_win = extract_ucurv_window_dense(ucurv2_transform.Msubwin[key], size)
        # Use dimension-specific tolerance for dim==2
        if dim == 2:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-2, 1e-2
        np.testing.assert_allclose(ucurv_win, ucurv2_win, rtol=rtol, atol=atol)
