"""Simple tests for dtype conversion functions in _typing module."""

from __future__ import annotations

import numpy as np

from curvelets.numpy_refactor._typing import _to_complex_dtype, _to_real_dtype


def test_to_real_dtype_float32():
    """Test _to_real_dtype with np.float32."""
    result = _to_real_dtype(np.float32)
    assert result == np.float32


def test_to_real_dtype_float64():
    """Test _to_real_dtype with np.float64."""
    result = _to_real_dtype(np.float64)
    assert result == np.float64


def test_to_real_dtype_complex64():
    """Test _to_real_dtype with np.complex64."""
    result = _to_real_dtype(np.complex64)
    assert result == np.float32


def test_to_real_dtype_complex128():
    """Test _to_real_dtype with np.complex128."""
    result = _to_real_dtype(np.complex128)
    assert result == np.float64


def test_to_real_dtype_dtype_object():
    """Test _to_real_dtype with dtype object."""
    dtype_obj = np.dtype(np.float32)
    result = _to_real_dtype(dtype_obj)
    assert result == np.float32


def test_to_complex_dtype_float32():
    """Test _to_complex_dtype with np.float32."""
    result = _to_complex_dtype(np.float32)
    assert result == np.complex64


def test_to_complex_dtype_float64():
    """Test _to_complex_dtype with np.float64."""
    result = _to_complex_dtype(np.float64)
    assert result == np.complex128


def test_to_complex_dtype_complex64():
    """Test _to_complex_dtype with np.complex64 (unchanged)."""
    result = _to_complex_dtype(np.complex64)
    assert result == np.complex64


def test_to_complex_dtype_complex128():
    """Test _to_complex_dtype with np.complex128 (unchanged)."""
    result = _to_complex_dtype(np.complex128)
    assert result == np.complex128


def test_to_complex_dtype_dtype_object():
    """Test _to_complex_dtype with dtype object."""
    dtype_obj = np.dtype(np.float32)
    result = _to_complex_dtype(dtype_obj)
    assert result == np.complex64
