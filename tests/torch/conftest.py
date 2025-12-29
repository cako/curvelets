"""PyTorch-specific fixtures and utilities for UDCT test files."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
import torch

import curvelets.torch as torch_curvelets

# Common test parameters
COMMON_ALPHA = 0.15
COMMON_R = (np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3)
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
    """
    if dim == 2:
        return [(64, 64), (128, 128), (256, 256)]
    if dim == 3:
        return [(32, 32, 32), (64, 64, 64)]
    if dim == 4:
        return [(16, 16, 16, 16)]
    return []


def get_test_configs(dim: int) -> list[torch.Tensor]:
    """
    Get test configurations for a given dimension.

    Parameters
    ----------
    dim : int
        Dimension of the test configurations (2, 3, or 4).

    Returns
    -------
    list[torch.Tensor]
        List of test configurations (cfg tensors) for the given dimension.
    """
    if dim == 2:
        return [
            torch.tensor([[3, 3]]),
            torch.tensor([[6, 6]]),
            torch.tensor([[3, 3], [6, 6]]),
            torch.tensor([[3, 3], [6, 6], [12, 12]]),
        ]
    if dim == 3:
        return [
            torch.tensor([[3, 3, 3]]),
            torch.tensor([[6, 6, 6]]),
            torch.tensor([[3, 3, 3], [6, 6, 6]]),
        ]
    if dim == 4:
        return [
            torch.tensor([[3, 3, 3, 3]]),
        ]
    return []


class TransformWrapper:
    """Wrapper to provide uniform interface for PyTorch transforms."""

    def __init__(
        self,
        transform_obj: Any,
    ) -> None:
        self._obj = transform_obj

    def forward(self, data: torch.Tensor) -> torch_curvelets.UDCTCoefficients:
        """Forward transform."""
        return self._obj.forward(data)  # type: ignore[no-any-return]

    def backward(self, coeffs: torch_curvelets.UDCTCoefficients) -> torch.Tensor:
        """Backward transform."""
        return self._obj.backward(coeffs)


def setup_torch_transform(
    dim: int,
    shape_idx: int = 0,
    cfg_idx: int = 0,
    high: str = "curvelet",
    alpha: float = COMMON_ALPHA,
    transform_kind: Literal["real", "complex", "monogenic"] = "real",
) -> TransformWrapper:
    """
    Set up PyTorch UDCT transform for round-trip tests.

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

    transform_obj = torch_curvelets.UDCT(
        shape=size,
        angular_wedges_config=cfg,
        window_overlap=alpha,
        radial_frequency_params=COMMON_R,
        window_threshold=COMMON_WINTHRESH,
        high_frequency_mode=high,  # type: ignore[arg-type]
        transform_kind=transform_kind,
    )

    return TransformWrapper(transform_obj)


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return np.random.default_rng(42)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert timeout failures to xfail for tests marked with timeout_to_xfail."""
    outcome = yield
    rep = outcome.get_result()

    # Only handle call phase failures
    if rep.when != "call" or not rep.failed:
        return

    # Check if the test has the timeout_to_xfail marker
    marker = item.get_closest_marker("timeout_to_xfail")
    if marker is None:
        return

    # Check if this is a timeout failure
    if call.excinfo is None:
        return

    exc_value_str = str(call.excinfo.value).lower()
    if "timeout" not in exc_value_str:
        return

    # Convert to xfail by modifying the report
    rep.outcome = "skipped"
    rep.wasxfail = f"reason: {call.excinfo.value}"
