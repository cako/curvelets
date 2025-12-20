"""Shared fixtures and utilities for UDCT test files."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy.typing import UDCTCoefficients, UDCTWindows
from curvelets.numpy.utils import from_sparse_new

# Common test parameters
COMMON_ALPHA = 0.15
COMMON_R = tuple(np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3)
COMMON_WINTHRESH = 1e-5

# Note: ucurv and ucurv2 use hardcoded alpha=0.1 and r values
# For comparison tests, we use NumPy's defaults when possible
# or document the differences


def get_test_shapes(dim: int) -> list[tuple[int, ...]]:
    """
    Get test shapes for a given dimension.

    Parameters
    ----------
    dim : int
        Dimension of the test shapes (2, 3, or 4).

    Returns
    -------
    list[tuple[int, ...]]
        List of test shapes for the given dimension.

    Examples
    --------
    >>> shapes_2d = get_test_shapes(2)
    >>> len(shapes_2d) > 0
    True
    >>> shapes_2d[0]
    (64, 64)
    """
    if dim == 2:
        return [(64, 64), (128, 128), (256, 256)]
    elif dim == 3:
        return [(32, 32, 32), (64, 64, 64)]
    elif dim == 4:
        return [(16, 16, 16, 16)]
    else:
        return []


def get_test_configs(dim: int) -> list[np.ndarray]:
    """
    Get test configurations for a given dimension.

    Parameters
    ----------
    dim : int
        Dimension of the test configurations (2, 3, or 4).

    Returns
    -------
    list[np.ndarray]
        List of test configurations (cfg arrays) for the given dimension.

    Examples
    --------
    >>> configs_2d = get_test_configs(2)
    >>> len(configs_2d) > 0
    True
    >>> configs_2d[0].shape
    (1, 2)
    """
    if dim == 2:
        return [
            np.array([[3, 3]]),
            np.array([[6, 6]]),
            np.array([[3, 3], [6, 6]]),
            np.array([[3, 3], [6, 6], [12, 12]]),
        ]
    elif dim == 3:
        return [
            np.array([[3, 3, 3]]),
            np.array([[6, 6, 6]]),
            np.array([[3, 3, 3], [6, 6, 6]]),
        ]
    elif dim == 4:
        return [
            np.array([[3, 3, 3, 3]]),
        ]
    else:
        return []


def coeffs_dict_to_udct(
    coeffs_dict: dict[tuple[int, ...], np.ndarray],
) -> UDCTCoefficients:
    """
    Convert ucurv dict format to UDCTCoefficients format.

    Parameters
    ----------
    coeffs_dict : dict[tuple[int, ...], np.ndarray]
        Dictionary with keys like (scale, dir, wedge) for 2D or (scale, dir0, dir1, wedge) for 3D, etc.
        Note: ucurv uses scale 0 for curvelet coefficients, which are mapped to scale 1
        in the output to match ucurv2's structure.

    Returns
    -------
    UDCTCoefficients
        Nested list format: [[[low_freq]], [[dir0_wedge0, dir0_wedge1, ...], [dir1_wedge0, ...]], ...]

    Examples
    --------
    >>> import numpy as np
    >>> coeffs_dict = {(0,): np.array([1.0]), (0, 0, 0): np.array([2.0])}
    >>> coeffs = coeffs_dict_to_udct(coeffs_dict)
    >>> len(coeffs) > 0
    True
    """
    # Find all curvelet coefficient keys (length > 1, excluding low freq key (0,))
    curvelet_keys = [k for k in coeffs_dict.keys() if len(k) > 1]
    
    if not curvelet_keys:
        # No curvelet coefficients, just return low frequency
        coeffs: UDCTCoefficients = [[[coeffs_dict[(0,)]]]]
        return coeffs
    
    # Determine key structure: for ucurv, keys are (scale, *dir_indices, wedge)
    # The last element is the wedge index, everything between scale and wedge are direction indices
    first_key = curvelet_keys[0]
    num_dir_indices = len(first_key) - 2  # Total length - scale - wedge
    
    # Find max direction index (for 2D: k[1], for 3D: max(k[1], k[2]), etc.)
    # We'll use the first direction index to group by direction
    max_dir = max(k[1] for k in curvelet_keys)
    
    # Initialize structure with low frequency
    coeffs: UDCTCoefficients = [[[coeffs_dict[(0,)]]]]
    
    # Check if curvelet coefficients are at scale 0 (ucurv format)
    scales_in_keys = set(k[0] for k in curvelet_keys)
    
    if scales_in_keys == {0}:
        # ucurv format: curvelet coefficients are at scale 0, map them to scale 1 in output
        # ucurv2 groups by scale, then direction (k[1]), then sorts by remaining indices (k[2:])
        coeffs.append([])
        for dir_idx in range(max_dir + 1):
            coeffs[1].append([])
            # Find all keys for this direction (scale is 0 in input)
            # Sort by remaining indices (k[2:]) to match ucurv2's behavior
            scale_dir_keys = [
                k
                for k in curvelet_keys
                if k[0] == 0 and k[1] == dir_idx
            ]
            scale_dir_keys_sorted = sorted(
                scale_dir_keys,
                key=lambda k: k[2:],  # Sort by all indices after scale and direction
            )
            # Extract coefficients in sorted order
            for key in scale_dir_keys_sorted:
                coeffs[1][dir_idx].append(coeffs_dict[key])
    else:
        # Other formats: process each scale >= 1
        for scale in sorted(scales_in_keys):
            if scale == 0:
                continue  # Skip scale 0, it's only for low frequency
            coeffs.append([])
            for dir_idx in range(max_dir + 1):
                coeffs[scale].append([])
                # Find all keys for this scale and direction
                # Sort by remaining indices (k[2:]) to match ucurv2's behavior
                scale_dir_keys = [
                    k
                    for k in curvelet_keys
                    if k[0] == scale and k[1] == dir_idx
                ]
                scale_dir_keys_sorted = sorted(
                    scale_dir_keys,
                    key=lambda k: k[2:],  # Sort by all indices after scale and direction
                )
                # Extract coefficients in sorted order
                for key in scale_dir_keys_sorted:
                    coeffs[scale][dir_idx].append(coeffs_dict[key])

    return coeffs


def coeffs_udct_to_dict(
    coeffs: UDCTCoefficients,
) -> dict[tuple[int, ...], np.ndarray]:
    """
    Convert UDCTCoefficients format to ucurv dict format.

    Parameters
    ----------
    coeffs : UDCTCoefficients
        Nested list format from NumPy implementation.

    Returns
    -------
    dict[tuple[int, ...], np.ndarray]
        Dictionary with keys like (scale, dir, wedge) and coefficient arrays as values.

    Examples
    --------
    >>> import numpy as np
    >>> coeffs = [[[np.array([1.0])]], [[[np.array([2.0])]]]]
    >>> coeffs_dict = coeffs_udct_to_dict(coeffs)
    >>> (0,) in coeffs_dict
    True
    """
    coeffs_dict: dict[tuple[int, ...], np.ndarray] = {}

    # Low frequency (scale 0)
    if len(coeffs) > 0 and len(coeffs[0]) > 0 and len(coeffs[0][0]) > 0:
        coeffs_dict[(0,)] = coeffs[0][0][0]

    # Process other scales
    for scale_idx, scale_coeffs in enumerate(coeffs[1:], start=1):
        for dir_idx, dir_coeffs in enumerate(scale_coeffs):
            for wedge_idx, wedge_coeff in enumerate(dir_coeffs):
                coeffs_dict[(scale_idx, dir_idx, wedge_idx)] = wedge_coeff

    return coeffs_dict


def extract_numpy_window_dense(
    window_sparse: tuple[np.ndarray, np.ndarray], size: tuple[int, ...]
) -> np.ndarray:
    """
    Extract a dense window from NumPy's sparse format.

    Parameters
    ----------
    window_sparse : tuple[np.ndarray, np.ndarray]
        Sparse window format (indices, values).
    size : tuple[int, ...]
        Size of the full window array.

    Returns
    -------
    np.ndarray
        Dense window array.
    """
    idx, val = from_sparse_new(window_sparse)
    win_dense = np.zeros(size, dtype=val.dtype)
    win_dense.flat[idx] = val
    return win_dense


def extract_ucurv_window_dense(
    window: np.ndarray | tuple[np.ndarray, np.ndarray], size: tuple[int, ...]
) -> np.ndarray:
    """
    Extract a dense window from ucurv format (can be sparse or dense).

    Parameters
    ----------
    window : np.ndarray | tuple[np.ndarray, np.ndarray]
        Window in ucurv format (either dense array or sparse tuple).
    size : tuple[int, ...]
        Size of the full window array.

    Returns
    -------
    np.ndarray
        Dense window array.
    """
    if isinstance(window, tuple):
        # Sparse format
        idx, val = window
        win_dense = np.zeros(size, dtype=val.dtype)
        win_dense.flat[idx] = val
        return win_dense
    else:
        # Dense format
        return window


def get_numpy_windows_dict(
    windows: UDCTWindows, size: tuple[int, ...]
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Convert NumPy windows to a dictionary format for easier comparison.

    Parameters
    ----------
    windows : UDCTWindows
        NumPy windows in nested list format.
    size : tuple[int, ...]
        Size of the full window array.

    Returns
    -------
    dict[tuple[int, int, int], np.ndarray]
        Dictionary with keys (scale, dir, wedge) and dense window arrays as values.
    """
    windows_dict: dict[tuple[int, int, int], np.ndarray] = {}

    # Low frequency (scale 0, dir 0, wedge 0)
    if len(windows) > 0 and len(windows[0]) > 0 and len(windows[0][0]) > 0:
        windows_dict[(0, 0, 0)] = extract_numpy_window_dense(windows[0][0][0], size)

    # Other scales
    for scale_idx, scale_windows in enumerate(windows[1:], start=1):
        for dir_idx, dir_windows in enumerate(scale_windows):
            for wedge_idx, wedge_window in enumerate(dir_windows):
                windows_dict[(scale_idx, dir_idx, wedge_idx)] = (
                    extract_numpy_window_dense(wedge_window, size)
                )

    return windows_dict


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return np.random.default_rng(42)


# Transform fixtures for round-trip tests
# These create a uniform interface so tests look identical across implementations


class TransformWrapper:
    """Wrapper to provide uniform interface across implementations."""

    def __init__(self, transform_obj, forward_fn, backward_fn):
        self._obj = transform_obj
        self._forward = forward_fn
        self._backward = backward_fn

    def forward(self, data: np.ndarray) -> np.ndarray | UDCTCoefficients:
        """Forward transform."""
        return self._forward(data)

    def backward(self, coeffs: np.ndarray | UDCTCoefficients) -> np.ndarray:
        """Backward transform."""
        return self._backward(coeffs)


def _create_numpy_transform(
    size: tuple[int, ...],
    cfg: np.ndarray,
    high: str = "curvelet",
    alpha: float = COMMON_ALPHA,
) -> TransformWrapper:
    """Create NumPy UDCT transform."""
    import curvelets.numpy as numpy_udct

    transform_obj = numpy_udct.UDCT(
        shape=size,
        cfg=cfg,
        alpha=alpha,
        r=COMMON_R,
        winthresh=COMMON_WINTHRESH,
        high=high,
    )

    def forward(data):
        return transform_obj.forward(data)

    def backward(coeffs):
        return transform_obj.backward(coeffs)

    return TransformWrapper(transform_obj, forward, backward)


def _create_ucurv_transform(size: tuple[int, ...], cfg: np.ndarray) -> TransformWrapper:
    """Create ucurv transform."""
    from curvelets.ucurv import ucurv

    shape_array = np.array(size, dtype=int)
    cfg_list = cfg.tolist() if hasattr(cfg, "tolist") else cfg
    transform_obj = ucurv.udct(shape_array, cfg_list)

    def forward(data):
        return ucurv.ucurvfwd(data, transform_obj)

    def backward(coeffs):
        return ucurv.ucurvinv(coeffs, transform_obj)

    return TransformWrapper(transform_obj, forward, backward)


def _create_ucurv2_transform(
    size: tuple[int, ...], cfg: np.ndarray, high: str = "curvelet", alpha: float = COMMON_ALPHA
) -> TransformWrapper:
    """Create ucurv2 transform."""
    from curvelets.ucurv import udct as ucurv2_udct

    transform_obj = ucurv2_udct.UDCT(shape=size, cfg=cfg, high=high, alpha=alpha)

    def forward(data):
        return transform_obj.forward(data)

    def backward(coeffs):
        return transform_obj.backward(coeffs)

    return TransformWrapper(transform_obj, forward, backward)


def setup_numpy_transform(
    dim: int,
    shape_idx: int = 0,
    cfg_idx: int = 0,
    high: str = "curvelet",
    alpha: float = COMMON_ALPHA,
) -> TransformWrapper:
    """
    Set up NumPy UDCT transform for round-trip tests.

    Parameters
    ----------
    dim : int
        Dimension.
    shape_idx : int, optional
        Index into shapes list. Default is 0.
    cfg_idx : int, optional
        Index into configs list. Default is 0.
    high : str, optional
        High frequency mode ("curvelet" or "wavelet"). Default is "curvelet".
    alpha : float, optional
        Alpha parameter for window overlap. Default is COMMON_ALPHA.

    Returns
    -------
    TransformWrapper
        Transform with forward/backward methods.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")
    if shape_idx >= len(shapes) or cfg_idx >= len(configs):
        pytest.skip(f"Index out of range for dimension {dim}")

    size = shapes[shape_idx]
    cfg = configs[cfg_idx]
    return _create_numpy_transform(size, cfg, high=high, alpha=alpha)


def setup_ucurv_transform(
    dim: int, shape_idx: int = 0, cfg_idx: int = 0
) -> TransformWrapper:
    """
    Set up ucurv transform for round-trip tests.

    Parameters
    ----------
    dim : int
        Dimension.
    shape_idx : int, optional
        Index into shapes list. Default is 0.
    cfg_idx : int, optional
        Index into configs list. Default is 0.

    Returns
    -------
    TransformWrapper
        Transform with forward/backward methods.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")
    if shape_idx >= len(shapes) or cfg_idx >= len(configs):
        pytest.skip(f"Index out of range for dimension {dim}")

    size = shapes[shape_idx]
    cfg = configs[cfg_idx]
    return _create_ucurv_transform(size, cfg)


def setup_ucurv2_transform(
    dim: int, shape_idx: int = 0, cfg_idx: int = 0, high: str = "curvelet", alpha: float = COMMON_ALPHA
) -> TransformWrapper:
    """
    Set up ucurv2 transform for round-trip tests.

    Parameters
    ----------
    dim : int
        Dimension.
    shape_idx : int, optional
        Index into shapes list. Default is 0.
    cfg_idx : int, optional
        Index into configs list. Default is 0.
    high : str, optional
        High frequency mode. Default is "curvelet".
    alpha : float, optional
        Alpha parameter. Default is COMMON_ALPHA.

    Returns
    -------
    TransformWrapper
        Transform with forward/backward methods.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)
    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")
    if shape_idx >= len(shapes) or cfg_idx >= len(configs):
        pytest.skip(f"Index out of range for dimension {dim}")

    size = shapes[shape_idx]
    cfg = configs[cfg_idx]
    return _create_ucurv2_transform(size, cfg, high, alpha)
