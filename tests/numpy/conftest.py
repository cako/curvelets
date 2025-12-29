"""NumPy-specific fixtures and utilities for UDCT test files."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest

from curvelets.numpy._typing import UDCTCoefficients
from curvelets.numpy._utils import from_sparse_new

# Common test parameters
COMMON_ALPHA = 0.15
COMMON_R = tuple(np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3)
COMMON_WINTHRESH = 1e-5


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
    if dim == 3:
        return [(32, 32, 32), (64, 64, 64)]
    if dim == 4:
        return [(16, 16, 16, 16)]
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
    if dim == 3:
        return [
            np.array([[3, 3, 3]]),
            np.array([[6, 6, 6]]),
            np.array([[3, 3, 3], [6, 6, 6]]),
        ]
    if dim == 4:
        return [
            np.array([[3, 3, 3, 3]]),
        ]
    return []


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

    Examples
    --------
    >>> import numpy as np
    >>> from tests.numpy.conftest import extract_numpy_window_dense
    >>> idx = np.array([0, 1, 2])
    >>> val = np.array([1.0, 2.0, 3.0])
    >>> win = extract_numpy_window_dense((idx, val), (4,))
    >>> win.shape
    (4,)
    """
    idx, val = from_sparse_new(window_sparse)
    win_dense = np.zeros(size, dtype=val.dtype)
    win_dense.flat[idx] = val
    return win_dense


# Transform fixtures for round-trip tests
# These create a uniform interface so tests look identical across implementations


class TransformWrapper:
    """Wrapper to provide uniform interface across implementations."""

    def __init__(
        self,
        transform_obj: Any,
        forward_fn: Callable[[np.ndarray], np.ndarray | UDCTCoefficients],
        backward_fn: Callable[[np.ndarray | UDCTCoefficients], np.ndarray],
    ) -> None:
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
    transform_kind: Literal["real", "complex", "monogenic"] = "real",
) -> TransformWrapper:
    """Create NumPy UDCT transform."""
    import curvelets.numpy as numpy_udct

    transform_obj = numpy_udct.UDCT(
        shape=size,
        angular_wedges_config=cfg,
        window_overlap=alpha,
        radial_frequency_params=COMMON_R,
        window_threshold=COMMON_WINTHRESH,
        high_frequency_mode=high,  # type: ignore[arg-type]
        transform_kind=transform_kind,
    )

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
    transform_kind: Literal["real", "complex", "monogenic"] = "real",
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
    transform_kind : str, optional
        Type of transform ("real", "complex", or "monogenic").
        Default is "real".

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
    return _create_numpy_transform(
        size, cfg, high=high, alpha=alpha, transform_kind=transform_kind
    )
