"""Tests for equivalence between UDCTWindow staticmethods and original numpy functions.

This module verifies that the refactored UDCTWindow staticmethods produce
equivalent results to the original functions in curvelets.numpy for 2D, 3D, and 4D cases.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

# Original implementations
from curvelets.numpy import udctmdwin
from curvelets.numpy.udctmdwin import (
    _calculate_decimation_ratios_with_lowest,
    _create_angle_info,
    _create_bandpass_windows,
    _create_mdirs,
    _inplace_normalize_windows,
    _inplace_sort_windows,
    _nchoosek,
)
from curvelets.numpy.utils import (
    ParamUDCT as ParamUDCTOriginal,
    adapt_grid,
    angle_fun,
    angle_kron,
    fftflip,
    to_sparse_new,
)

# Refactored implementations
from curvelets.numpy_refactor._udct_windows import UDCTWindow
from curvelets.numpy_refactor._utils import ParamUDCT as ParamUDCTRefactor

# Common test parameters
COMMON_ALPHA = 0.15
COMMON_R = tuple(np.pi * np.array([1.0, 2.0, 2.0, 4.0]) / 3)
COMMON_WINTHRESH = 1e-5


# Helper functions for parameter conversion
def create_params_original(
    size: tuple[int, ...],
    dim: int,
    cfg: np.ndarray,
    alpha: float = COMMON_ALPHA,
    r: tuple[float, float, float, float] = COMMON_R,
    winthresh: float = COMMON_WINTHRESH,
) -> ParamUDCTOriginal:
    """Create original ParamUDCT."""
    return ParamUDCTOriginal(
        dim=dim, size=size, cfg=cfg, alpha=alpha, r=r, winthresh=winthresh
    )


def create_params_refactor(
    size: tuple[int, ...],
    dim: int,
    angular_wedges_config: np.ndarray,
    window_overlap: float = COMMON_ALPHA,
    radial_frequency_params: tuple[float, float, float, float] = COMMON_R,
    window_threshold: float = COMMON_WINTHRESH,
) -> ParamUDCTRefactor:
    """Create refactored ParamUDCT."""
    return ParamUDCTRefactor(
        dim=dim,
        size=size,
        angular_wedges_config=angular_wedges_config,
        window_overlap=window_overlap,
        radial_frequency_params=radial_frequency_params,
        window_threshold=window_threshold,
    )


# Test fixtures for dimensions
@pytest.fixture(
    params=[
        ((32, 32), 2, np.array([[3, 3], [6, 6]])),  # 2D
        ((16, 16, 16), 3, np.array([[3, 3, 3], [6, 6, 6]])),  # 3D
        ((8, 8, 8, 8), 4, np.array([[3, 3, 3, 3]])),  # 4D
    ],
    ids=["2D", "3D", "4D"],
)
def dimension_case(request: pytest.FixtureRequest) -> tuple[tuple[int, ...], int, np.ndarray]:
    """Fixture providing size, dimension, and config for parametrized tests."""
    return request.param


# Helper function to compare dictionaries recursively
def compare_dicts(
    dict1: dict[Any, Any], dict2: dict[Any, Any], rtol: float = 1e-10, atol: float = 1e-12
) -> None:
    """Recursively compare two dictionaries with numpy arrays."""
    assert set(dict1.keys()) == set(dict2.keys()), "Dictionary keys don't match"
    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            np.testing.assert_allclose(val1, val2, rtol=rtol, atol=atol)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            compare_dicts(val1, val2, rtol=rtol, atol=atol)
        else:
            assert val1 == val2, f"Values don't match for key {key}"


# Test implementations
@pytest.mark.parametrize(
    "dim,n_elements",
    [
        (2, 4),  # Choose 2 from 4 elements
        (3, 5),  # Choose 2 from 5 elements
        (4, 6),  # Choose 2 from 6 elements
    ],
)
def test_nchoosek_equivalence(dim: int, n_elements: int) -> None:
    """Test that _nchoosek produces equivalent results."""
    n = np.arange(n_elements)
    k = 2

    original_result = _nchoosek(n, k)
    refactor_result = UDCTWindow._nchoosek(n, k)

    np.testing.assert_array_equal(original_result, refactor_result)


@pytest.mark.parametrize(
    "size",
    [
        (32, 32),  # 2D
        (16, 16, 16),  # 3D
        (8, 8, 8, 8),  # 4D
    ],
)
def test_to_sparse_equivalence(size: tuple[int, ...]) -> None:
    """Test that _to_sparse produces equivalent results."""
    # Create test array with some values above threshold
    arr = np.random.rand(*size).astype(np.float64)
    arr[arr < 0.5] = 0.0  # Set some values below threshold
    threshold = 0.3

    original_result = to_sparse_new(arr, threshold)
    refactor_result = UDCTWindow._to_sparse(arr, threshold)

    # Compare indices and values separately
    np.testing.assert_array_equal(original_result[0], refactor_result[0])
    np.testing.assert_allclose(original_result[1], refactor_result[1], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "size",
    [
        (32, 32),  # 2D
        (16, 16, 16),  # 3D
        (8, 8, 8, 8),  # 4D
    ],
)
def test_flip_with_fft_shift_equivalence(size: tuple[int, ...]) -> None:
    """Test that _flip_with_fft_shift produces equivalent results."""
    arr = np.random.rand(*size).astype(np.float64)

    # Test each axis
    for axis in range(len(size)):
        original_result = fftflip(arr, axis)
        refactor_result = UDCTWindow._flip_with_fft_shift(arr, axis)

        np.testing.assert_allclose(original_result, refactor_result, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "size",
    [
        (32, 32),  # 2D
        (16, 16, 16),  # 3D
        (8, 8, 8, 8),  # 4D
    ],
)
def test_create_angle_grids_equivalence(size: tuple[int, ...]) -> None:
    """Test that _create_angle_grids_from_frequency_grids produces equivalent results."""
    # Create frequency grids
    freq_grid_1 = np.linspace(-1.5 * np.pi, 0.5 * np.pi, size[0], endpoint=False)
    freq_grid_2 = np.linspace(-1.5 * np.pi, 0.5 * np.pi, size[1], endpoint=False)

    original_result = adapt_grid(freq_grid_1, freq_grid_2)
    refactor_result = UDCTWindow._create_angle_grids_from_frequency_grids(
        freq_grid_1, freq_grid_2
    )

    # adapt_grid returns (M2, M1), refactor returns (angle_grid_2, angle_grid_1)
    np.testing.assert_allclose(
        original_result[0], refactor_result[0], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        original_result[1], refactor_result[1], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "size,direction,num_wedges",
    [
        ((32, 32), 1, 3),  # 2D
        ((32, 32), 2, 6),  # 2D
        ((16, 16, 16), 1, 3),  # 3D
        ((8, 8, 8, 8), 1, 3),  # 4D
    ],
)
def test_create_angle_functions_equivalence(
    size: tuple[int, ...], direction: int, num_wedges: int
) -> None:
    """Test that _create_angle_functions produces equivalent results."""
    # Create angle grid
    angle_grid = np.linspace(-2, 2, size[0])
    window_overlap = COMMON_ALPHA

    original_result = angle_fun(angle_grid, direction, num_wedges, window_overlap)
    refactor_result = UDCTWindow._create_angle_functions(
        angle_grid, direction, num_wedges, window_overlap
    )

    np.testing.assert_allclose(original_result, refactor_result, rtol=1e-10, atol=1e-12)


def test_compute_angle_kronecker_product_equivalence(dimension_case: tuple) -> None:
    """Test that _compute_angle_kronecker_product produces equivalent results."""
    size, dim, cfg = dimension_case

    # Create test parameters
    params_original = create_params_original(size, dim, cfg)
    params_refactor = create_params_refactor(size, dim, cfg)

    # Get valid dimension permutations (combinations of 2 dimensions, 1-based)
    mperms = _nchoosek(np.arange(dim), 2)
    # Use first valid permutation (convert to 1-based indexing)
    dimension_permutation = mperms[0] + 1

    # Create test angle function with appropriate size
    # The angle function size should match the product of sizes of the two dimensions
    # in the permutation (for the meshgrid created from those two frequency grids)
    dim0_idx = dimension_permutation[0] - 1  # Convert to 0-based
    dim1_idx = dimension_permutation[1] - 1  # Convert to 0-based
    angle_function_size = size[dim0_idx] * size[dim1_idx]
    angle_function_1d = np.random.rand(angle_function_size).astype(np.float64)

    original_result = angle_kron(angle_function_1d, dimension_permutation, params_original)
    refactor_result = UDCTWindow._compute_angle_kronecker_product(
        angle_function_1d, dimension_permutation, params_refactor
    )

    np.testing.assert_allclose(original_result, refactor_result, rtol=1e-10, atol=1e-12)


def test_create_direction_mappings_equivalence(dimension_case: tuple) -> None:
    """Test that _create_direction_mappings produces equivalent results."""
    size, dim, cfg = dimension_case
    num_resolutions = len(cfg)

    original_result = _create_mdirs(dim, num_resolutions)
    refactor_result = UDCTWindow._create_direction_mappings(dim, num_resolutions)

    assert len(original_result) == len(refactor_result)
    for orig, ref in zip(original_result, refactor_result):
        np.testing.assert_array_equal(orig, ref)


def test_create_bandpass_windows_equivalence(dimension_case: tuple) -> None:
    """Test that _create_bandpass_windows produces equivalent results."""
    size, dim, cfg = dimension_case
    num_scales = len(cfg)
    r = COMMON_R

    original_result = _create_bandpass_windows(num_scales, size, r)
    refactor_result = UDCTWindow._create_bandpass_windows(num_scales, size, r)

    # Compare frequency grids
    assert set(original_result[0].keys()) == set(refactor_result[0].keys())
    for key in original_result[0].keys():
        np.testing.assert_allclose(
            original_result[0][key], refactor_result[0][key], rtol=1e-10, atol=1e-12
        )

    # Compare bandpass windows
    assert set(original_result[1].keys()) == set(refactor_result[1].keys())
    for key in original_result[1].keys():
        np.testing.assert_allclose(
            original_result[1][key], refactor_result[1][key], rtol=1e-10, atol=1e-12
        )


def test_create_angle_info_equivalence(dimension_case: tuple) -> None:
    """Test that _create_angle_info produces equivalent results."""
    size, dim, cfg = dimension_case
    num_resolutions = len(cfg)
    alpha = COMMON_ALPHA

    # Create frequency grid (same as in bandpass windows)
    frequency_grid: dict[int, np.ndarray] = {}
    for dimension_idx in range(dim):
        frequency_grid[dimension_idx] = np.linspace(
            -1.5 * np.pi, 0.5 * np.pi, size[dimension_idx], endpoint=False
        )

    original_result = _create_angle_info(frequency_grid, dim, num_resolutions, cfg, alpha)
    refactor_result = UDCTWindow._create_angle_info(
        frequency_grid, dim, num_resolutions, cfg, alpha
    )

    # Compare angle functions dictionaries
    compare_dicts(original_result[0], refactor_result[0], rtol=1e-10, atol=1e-12)

    # Compare angle indices dictionaries
    compare_dicts(original_result[1], refactor_result[1], rtol=1e-10, atol=1e-12)


def test_calculate_decimation_ratios_equivalence(dimension_case: tuple) -> None:
    """Test that _calculate_decimation_ratios_with_lowest produces equivalent results."""
    size, dim, cfg = dimension_case
    num_resolutions = len(cfg)

    # Create direction mappings first
    direction_mappings = _create_mdirs(dim, num_resolutions)

    original_result = _calculate_decimation_ratios_with_lowest(
        num_resolutions, dim, cfg, direction_mappings
    )
    refactor_result = UDCTWindow._calculate_decimation_ratios_with_lowest(
        num_resolutions, dim, cfg, direction_mappings
    )

    assert len(original_result) == len(refactor_result)
    for orig, ref in zip(original_result, refactor_result):
        np.testing.assert_array_equal(orig, ref)


def test_inplace_normalize_windows_equivalence(dimension_case: tuple) -> None:
    """Test that _inplace_normalize_windows produces equivalent results."""
    size, dim, cfg = dimension_case
    num_resolutions = len(cfg)

    # Create parameters
    params_original = create_params_original(size, dim, cfg)
    params_refactor = create_params_refactor(size, dim, cfg)

    # Compute windows using original function
    windows_original, _, _ = udctmdwin(params_original)
    windows_refactor, _, _ = udctmdwin(params_original)  # Use same for comparison

    # Create deep copies for in-place operations
    windows_original_copy = copy.deepcopy(windows_original)
    windows_refactor_copy = copy.deepcopy(windows_refactor)

    # Apply normalization
    _inplace_normalize_windows(
        windows_original_copy, size, dim, num_resolutions
    )
    UDCTWindow._inplace_normalize_windows(
        windows_refactor_copy, size, dim, num_resolutions
    )

    # Compare normalized windows
    # Compare structure
    assert len(windows_original_copy) == len(windows_refactor_copy)
    for scale_idx in range(len(windows_original_copy)):
        assert len(windows_original_copy[scale_idx]) == len(windows_refactor_copy[scale_idx])
        for dir_idx in range(len(windows_original_copy[scale_idx])):
            assert (
                len(windows_original_copy[scale_idx][dir_idx])
                == len(windows_refactor_copy[scale_idx][dir_idx])
            )
            for wedge_idx in range(len(windows_original_copy[scale_idx][dir_idx])):
                orig_idx, orig_val = windows_original_copy[scale_idx][dir_idx][wedge_idx]
                ref_idx, ref_val = windows_refactor_copy[scale_idx][dir_idx][wedge_idx]
                np.testing.assert_array_equal(orig_idx, ref_idx)
                np.testing.assert_allclose(orig_val, ref_val, rtol=1e-10, atol=1e-12)


def test_inplace_sort_windows_equivalence(dimension_case: tuple) -> None:
    """Test that _inplace_sort_windows produces equivalent results."""
    size, dim, cfg = dimension_case
    num_resolutions = len(cfg)

    # Create parameters
    params_original = create_params_original(size, dim, cfg)

    # Compute windows and indices using original function
    windows_original, _, indices_original = udctmdwin(params_original)

    # Create deep copies for in-place operations
    windows_original_copy = copy.deepcopy(windows_original)
    indices_original_copy = copy.deepcopy(indices_original)
    windows_refactor_copy = copy.deepcopy(windows_original)
    indices_refactor_copy = copy.deepcopy(indices_original)

    # Apply sorting
    _inplace_sort_windows(windows_original_copy, indices_original_copy, num_resolutions, dim)
    UDCTWindow._inplace_sort_windows(
        windows_refactor_copy, indices_refactor_copy, num_resolutions, dim
    )

    # Compare sorted windows and indices
    assert len(indices_original_copy) == len(indices_refactor_copy)
    for scale_idx in indices_original_copy.keys():
        if scale_idx == 0:
            continue  # Skip scale 0 (low frequency)
        assert set(indices_original_copy[scale_idx].keys()) == set(
            indices_refactor_copy[scale_idx].keys()
        )
        for dir_idx in indices_original_copy[scale_idx].keys():
            orig_indices = indices_original_copy[scale_idx][dir_idx]
            ref_indices = indices_refactor_copy[scale_idx][dir_idx]
            np.testing.assert_array_equal(orig_indices, ref_indices)

            # Compare corresponding windows
            orig_windows = windows_original_copy[scale_idx][dir_idx]
            ref_windows = windows_refactor_copy[scale_idx][dir_idx]
            assert len(orig_windows) == len(ref_windows)
            for wedge_idx in range(len(orig_windows)):
                orig_idx, orig_val = orig_windows[wedge_idx]
                ref_idx, ref_val = ref_windows[wedge_idx]
                np.testing.assert_array_equal(orig_idx, ref_idx)
                np.testing.assert_allclose(orig_val, ref_val, rtol=1e-10, atol=1e-12)

