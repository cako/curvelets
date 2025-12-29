"""NumPy vs PyTorch comparison tests for UDCT windows."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvelets.numpy._udct_windows import UDCTWindow as NpUDCTWindow
from curvelets.numpy._utils import ParamUDCT as NpParamUDCT
from curvelets.torch._udct_windows import UDCTWindow as TorchUDCTWindow
from curvelets.torch._utils import ParamUDCT as TorchParamUDCT

from .conftest import TEST_SHAPES


def create_params(shape: tuple[int, ...], angular_config: np.ndarray):
    """Create ParamUDCT for both NumPy and PyTorch."""
    np_params = NpParamUDCT(
        shape=shape,
        angular_wedges_config=angular_config,
        window_overlap=0.15,
        radial_frequency_params=(
            np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 3,
            4 * np.pi / 3,
        ),
        window_threshold=1e-5,
    )
    torch_params = TorchParamUDCT(
        shape=shape,
        angular_wedges_config=torch.from_numpy(angular_config),
        window_overlap=0.15,
        radial_frequency_params=(
            np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 3,
            4 * np.pi / 3,
        ),
        window_threshold=1e-5,
    )
    return np_params, torch_params


@pytest.mark.parametrize("ndim", [2, 3])
def test_udct_window_compute_matches_numpy(ndim):
    """Test UDCTWindow.compute() produces same results as NumPy."""
    shape = TEST_SHAPES[ndim]

    # Create angular config based on dimension
    if ndim == 2:
        angular_config = np.array([[3, 3]])
    elif ndim == 3:
        angular_config = np.array([[3, 3, 3]])
    else:
        angular_config = np.array([[3] * ndim])

    np_params, torch_params = create_params(shape, angular_config)

    np_window = NpUDCTWindow(np_params)
    torch_window = TorchUDCTWindow(torch_params)

    np_windows, np_dec_ratios, np_indices = np_window.compute()
    torch_windows, torch_dec_ratios, torch_indices = torch_window.compute()

    # Check same number of scales
    assert len(np_windows) == len(torch_windows)

    # Check decimation ratios match
    for scale_idx in range(len(np_dec_ratios)):
        np.testing.assert_array_equal(
            np_dec_ratios[scale_idx], torch_dec_ratios[scale_idx].numpy()
        )

    # Check windows match (compare sparse format)
    for scale_idx in range(len(np_windows)):
        assert len(np_windows[scale_idx]) == len(torch_windows[scale_idx])
        for dir_idx in range(len(np_windows[scale_idx])):
            assert len(np_windows[scale_idx][dir_idx]) == len(
                torch_windows[scale_idx][dir_idx]
            )
            for wedge_idx in range(len(np_windows[scale_idx][dir_idx])):
                np_idx, np_val = np_windows[scale_idx][dir_idx][wedge_idx]
                torch_idx, torch_val = torch_windows[scale_idx][dir_idx][wedge_idx]

                # Values should match (within tolerance)
                # Use atol=1e-5 due to floating point precision differences
                np.testing.assert_allclose(
                    np_val.flatten(), torch_val.numpy().flatten(), atol=1e-5
                )


@pytest.mark.parametrize("ndim", [2, 3])
def test_decimation_ratios_match_numpy(ndim):
    """Test that decimation ratios match between NumPy and PyTorch."""
    shape = TEST_SHAPES[ndim]

    if ndim == 2:
        angular_config = np.array([[3, 3], [6, 6]])
    elif ndim == 3:
        angular_config = np.array([[3, 3, 3]])
    else:
        angular_config = np.array([[3] * ndim])

    np_params, torch_params = create_params(shape, angular_config)

    np_window = NpUDCTWindow(np_params)
    torch_window = TorchUDCTWindow(torch_params)

    _, np_dec_ratios, _ = np_window.compute()
    _, torch_dec_ratios, _ = torch_window.compute()

    for scale_idx in range(len(np_dec_ratios)):
        np.testing.assert_array_equal(
            np_dec_ratios[scale_idx], torch_dec_ratios[scale_idx].numpy()
        )
