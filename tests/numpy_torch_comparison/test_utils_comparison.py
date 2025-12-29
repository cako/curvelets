"""NumPy vs PyTorch comparison tests for utility functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import curvelets.numpy._utils as np_utils
import curvelets.torch._utils as torch_utils

from .conftest import TEST_SHAPES


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_circular_shift_matches_numpy(ndim):
    """Test circular_shift produces same results as NumPy."""
    rng = np.random.default_rng(42)
    shape = TEST_SHAPES[ndim]
    arr_np = rng.normal(size=shape)
    arr_torch = torch.from_numpy(arr_np)
    shift = tuple(s // 3 for s in shape)

    result_np = np_utils.circular_shift(arr_np, shift)
    result_torch = torch_utils.circular_shift(arr_torch, shift)

    np.testing.assert_allclose(result_np, result_torch.numpy(), atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_downsample_matches_numpy(ndim):
    """Test downsample produces same results as NumPy."""
    rng = np.random.default_rng(42)
    shape = TEST_SHAPES[ndim]
    arr_np = rng.normal(size=shape)
    arr_torch = torch.from_numpy(arr_np)
    decimation_np = np.array([2] * ndim)
    decimation_torch = torch.tensor([2] * ndim)

    result_np = np_utils.downsample(arr_np, decimation_np)
    result_torch = torch_utils.downsample(arr_torch, decimation_torch)

    np.testing.assert_allclose(result_np, result_torch.numpy(), atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_upsample_matches_numpy(ndim):
    """Test upsample produces same results as NumPy."""
    rng = np.random.default_rng(42)
    shape = TEST_SHAPES[ndim]
    # Use smaller shape for upsampling to avoid memory issues
    small_shape = tuple(s // 2 for s in shape)
    arr_np = rng.normal(size=small_shape)
    arr_torch = torch.from_numpy(arr_np)
    decimation_np = np.array([2] * ndim)
    decimation_torch = torch.tensor([2] * ndim)

    result_np = np_utils.upsample(arr_np, decimation_np)
    result_torch = torch_utils.upsample(arr_torch, decimation_torch)

    np.testing.assert_allclose(result_np, result_torch.numpy(), atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_flip_fft_all_axes_matches_numpy(ndim):
    """Test flip_fft_all_axes produces same results as NumPy."""
    rng = np.random.default_rng(42)
    shape = TEST_SHAPES[ndim]
    arr_np = rng.normal(size=shape)
    arr_torch = torch.from_numpy(arr_np)

    result_np = np_utils.flip_fft_all_axes(arr_np)
    result_torch = torch_utils.flip_fft_all_axes(arr_torch)

    np.testing.assert_allclose(result_np, result_torch.numpy(), atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_meyer_window_matches_numpy(ndim):
    """Test meyer_window produces same results as NumPy."""
    # Create frequency array
    shape = TEST_SHAPES[ndim]
    freq_np = np.linspace(-1.5 * np.pi, 0.5 * np.pi, shape[0], endpoint=False)
    freq_torch = torch.from_numpy(freq_np)

    # Standard Meyer wavelet parameters
    params = (np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3)

    result_np = np_utils.meyer_window(freq_np, *params)
    result_torch = torch_utils.meyer_window(freq_torch, *params)

    np.testing.assert_allclose(result_np, result_torch.numpy(), atol=1e-10)
