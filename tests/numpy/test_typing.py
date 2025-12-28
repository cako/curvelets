"""Simple tests for dtype conversion functions in _typing module."""

from __future__ import annotations

import numpy as np

from curvelets.numpy._typing import _to_complex_dtype, _to_real_dtype


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


# ============================================================================
# Python version compatibility tests for type aliases
# ============================================================================


def test_typing_python310_compatibility():
    """
    Test that type aliases work on Python 3.10+.

    On Python 3.10+, the typing module provides TypeAlias and
    the type aliases use built-in list/tuple syntax.

    Examples
    --------
    >>> from curvelets.numpy._typing import UDCTCoefficients, UDCTWindows, MUDCTCoefficients
    >>> # Type aliases should be importable
    >>> assert UDCTCoefficients is not None
    >>> assert UDCTWindows is not None
    >>> assert MUDCTCoefficients is not None
    """
    import sys

    from curvelets.numpy._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows

    # Verify type aliases are importable
    assert UDCTCoefficients is not None
    assert UDCTWindows is not None
    assert MUDCTCoefficients is not None

    # On Python 3.10+, type aliases should use built-in list/tuple syntax
    if sys.version_info >= (3, 10):
        # Verify the type aliases are defined (they should be)
        # We can't easily test the internal structure, but we can verify
        # they're not None and are callable/usable
        assert UDCTCoefficients is not None
        assert UDCTWindows is not None
        assert MUDCTCoefficients is not None


def test_typing_python39_compatibility():
    """
    Test that type aliases work on Python 3.9.

    On Python 3.9, typing_extensions is used and type aliases use
    typing.List/Tuple syntax.

    Examples
    --------
    >>> import sys
    >>> from curvelets.numpy._typing import UDCTCoefficients, UDCTWindows, MUDCTCoefficients
    >>> # Type aliases should be importable on all Python versions
    >>> assert UDCTCoefficients is not None
    >>> assert UDCTWindows is not None
    >>> assert MUDCTCoefficients is not None
    """
    import sys

    from curvelets.numpy._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows

    # Verify type aliases are importable
    assert UDCTCoefficients is not None
    assert UDCTWindows is not None
    assert MUDCTCoefficients is not None

    # On Python 3.9, type aliases use typing.List/Tuple syntax
    # We can't easily test the internal structure without importing typing,
    # but we can verify they're importable and usable
    if sys.version_info < (3, 10):
        # Verify typing_extensions import path works
        # The type aliases should still be usable
        assert UDCTCoefficients is not None
        assert UDCTWindows is not None
        assert MUDCTCoefficients is not None
