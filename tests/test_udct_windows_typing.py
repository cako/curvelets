"""Type hint validation tests for UDCT windows module."""

from __future__ import annotations

import typing

import numpy as np

from curvelets.numpy_refactor._typing import (
    FloatingNDArray,
    IntegerNDArray,
    IntpNDArray,
)
from curvelets.numpy_refactor._udct_windows import UDCTWindow
from curvelets.numpy_refactor._utils import ParamUDCT


def test_type_aliases_exist() -> None:
    """Test that all type aliases are properly defined."""
    assert FloatingNDArray is not None
    assert IntegerNDArray is not None
    assert IntpNDArray is not None


def test_compute_angle_component_type_hints() -> None:
    """Test that _compute_angle_component has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._compute_angle_component)
    assert "x_primary" in hints
    assert "x_secondary" in hints
    assert "return" in hints


def test_create_angle_grids_type_hints() -> None:
    """Test that _create_angle_grids_from_frequency_grids has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._create_angle_grids_from_frequency_grids)
    assert "frequency_grid_1" in hints
    assert "frequency_grid_2" in hints
    assert "return" in hints


def test_create_angle_functions_type_hints() -> None:
    """Test that _create_angle_functions has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._create_angle_functions)
    assert "angle_grid" in hints
    assert "direction" in hints
    assert "num_angular_wedges" in hints
    assert "window_overlap" in hints
    assert "return" in hints


def test_compute_angle_kronecker_product_type_hints() -> None:
    """Test that _compute_angle_kronecker_product has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._compute_angle_kronecker_product)
    assert "angle_function_1d" in hints
    assert "dimension_permutation" in hints
    assert "param_udct" in hints
    assert "return" in hints


def test_flip_with_fft_shift_type_hints() -> None:
    """Test that _flip_with_fft_shift has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._flip_with_fft_shift)
    assert "input_array" in hints
    assert "axis" in hints
    assert "return" in hints


def test_to_sparse_type_hints() -> None:
    """Test that _to_sparse has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._to_sparse)
    assert "arr" in hints
    assert "threshold" in hints
    assert "return" in hints


def test_nchoosek_type_hints() -> None:
    """Test that _nchoosek has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._nchoosek)
    assert "n" in hints
    assert "k" in hints
    assert "return" in hints


def test_create_bandpass_windows_type_hints() -> None:
    """Test that _create_bandpass_windows has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._create_bandpass_windows)
    assert "num_scales" in hints
    assert "shape" in hints
    assert "radial_frequency_params" in hints
    assert "return" in hints


def test_create_direction_mappings_type_hints() -> None:
    """Test that _create_direction_mappings has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._create_direction_mappings)
    assert "dimension" in hints
    assert "num_resolutions" in hints
    assert "return" in hints


def test_create_angle_info_type_hints() -> None:
    """Test that _create_angle_info has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._create_angle_info)
    assert "frequency_grid" in hints
    assert "dimension" in hints
    assert "num_resolutions" in hints
    assert "angular_wedges_config" in hints
    assert "window_overlap" in hints
    assert "return" in hints


def test_calculate_decimation_ratios_type_hints() -> None:
    """Test that _calculate_decimation_ratios_with_lowest has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._calculate_decimation_ratios_with_lowest)
    assert "num_resolutions" in hints
    assert "dimension" in hints
    assert "angular_wedges_config" in hints
    assert "direction_mappings" in hints
    assert "return" in hints


def test_inplace_sort_windows_type_hints() -> None:
    """Test that _inplace_sort_windows has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow._inplace_sort_windows)
    assert "windows" in hints
    assert "indices" in hints
    assert "num_resolutions" in hints
    assert "dimension" in hints
    assert "return" in hints


def test_compute_type_hints() -> None:
    """Test that compute has correct type hints."""
    hints = typing.get_type_hints(UDCTWindow.compute)
    assert "parameters" in hints
    assert "return" in hints


def test_compute_returns_correct_types() -> None:
    """Test that compute returns the correct types."""
    params = ParamUDCT(
        size=(64, 64),
        dim=2,
        angular_wedges_config=np.array(
            [[3, 3], [6, 6], [12, 12]]
        ),  # Shape: (num_scales, dim)
        window_overlap=0.15,
        window_threshold=1e-5,
        radial_frequency_params=(
            np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 3,
            4 * np.pi / 3,
        ),
    )
    windows, decimation_ratios, indices = UDCTWindow.compute(params)

    # Check types
    assert isinstance(windows, list)
    assert isinstance(decimation_ratios, list)
    assert isinstance(indices, dict)

    # Check structure
    assert (
        len(windows) == 4
    )  # 0 + res scales (res is computed from angular_wedges_config)
    assert len(decimation_ratios) == 4
    assert len(indices) == 4


def test_type_aliases_work_with_arrays() -> None:
    """Test that type aliases work correctly with actual arrays."""
    # FloatingNDArray should accept float32 and float64
    arr32: FloatingNDArray = np.array([1.0, 2.0], dtype=np.float32)
    arr64: FloatingNDArray = np.array([1.0, 2.0], dtype=np.float64)

    assert arr32.dtype == np.float32
    assert arr64.dtype == np.float64

    # IntegerNDArray should accept int arrays
    int_arr: IntegerNDArray = np.array([1, 2, 3], dtype=np.int_)
    assert int_arr.dtype == np.int_


def test_complex_dtype_promotion_numpy2_compatible() -> None:
    """Test that complex dtype promotion works correctly (NumPy 2.0 compatible)."""
    # Test float32 → complex64
    real_dtype = np.float32
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    assert complex_dtype == np.complex64

    # Test float64 → complex128
    real_dtype = np.float64
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    assert complex_dtype == np.complex128

    # Verify the pattern works with actual arrays
    arr32 = np.array([1.0], dtype=np.float32)
    result32 = np.zeros(
        (10,), dtype=np.complex64 if arr32.dtype == np.float32 else np.complex128
    )
    assert result32.dtype == np.complex64

    arr64 = np.array([1.0], dtype=np.float64)
    result64 = np.zeros(
        (10,), dtype=np.complex64 if arr64.dtype == np.float32 else np.complex128
    )
    assert result64.dtype == np.complex128
