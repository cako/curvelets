"""Type definitions for NumPy UDCT implementation.

This module provides type aliases for UDCT coefficients and windows.

Public Types
------------
UDCTCoefficients
    Generic type alias for UDCT coefficient structure.
UDCTWindows
    Generic type alias for UDCT window structure.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeVar

    from typing_extensions import TypeAliasType
else:
    from typing_extensions import TypeAlias, TypeAliasType, TypeVar

import numpy as np
import numpy.typing as npt

__all__ = [
    "UDCTCoefficients",
    "UDCTWindows",
]

# =============================================================================
# Private TypeVars (internal use only)
# =============================================================================

# _F: Real floating point types (np.floating and all subtypes)
_F = TypeVar("_F", bound=np.floating)

# _C: Complex floating point types (np.complexfloating and all subtypes)
_C = TypeVar("_C", bound=np.complexfloating)

# _T: Any scalar type (real or complex) for generic coefficient type
# np.inexact is the common base class for np.floating and np.complexfloating
_T = TypeVar("_T", bound=np.inexact)

# _A: Any numpy array type (includes all numpy scalar types: floating, complex, integer, bool, etc.)
_A = TypeVar("_A", bound=np.generic)

# =============================================================================
# Public Type Aliases
# =============================================================================

# Generic UDCT coefficients parameterized by scalar dtype
# Usage:
#   - UDCTCoefficients[np.floating] or UDCTCoefficients[_F] for monogenic transform
#   - UDCTCoefficients[np.complexfloating] or UDCTCoefficients[_C] for real/complex transforms
UDCTCoefficients = TypeAliasType(
    "UDCTCoefficients",
    list[list[list[npt.NDArray[_T]]]],
    type_params=(_T,),
)

# Generic UDCT windows parameterized by real floating dtype (no complex allowed)
# Usage: UDCTWindows[np.floating], UDCTWindows[np.float32], UDCTWindows[np.float64]
# Structure: list[list[list[tuple[indices, values]]]] where values are real floats
UDCTWindows = TypeAliasType(
    "UDCTWindows",
    list[list[list[tuple[npt.NDArray[np.intp], npt.NDArray[_F]]]]],
    type_params=(_F,),
)

# =============================================================================
# Private Type Aliases (internal use only)
# =============================================================================

# Type aliases for common NDArray types
# Note: These are kept for backward compatibility but new code should use TypeVars directly
_FloatingNDArray: TypeAlias = npt.NDArray[np.floating]
_IntegerNDArray: TypeAlias = npt.NDArray[np.int_]
_IntpNDArray: TypeAlias = npt.NDArray[np.intp]
_BoolNDArray: TypeAlias = npt.NDArray[np.bool_]

# =============================================================================
# Private Helper Functions
# =============================================================================


def _to_real_dtype(dtype: npt.DTypeLike) -> npt.DTypeLike:
    """
    Extract the real dtype from any dtype.

    If the input is a complex dtype (e.g., `np.complex64`), returns the
    corresponding real dtype (e.g., `np.float32`). If the input is already
    a real dtype, returns it unchanged.

    Parameters
    ----------
    dtype : npt.DTypeLike
        Input dtype. Can be a dtype object, dtype string, or type object.

    Returns
    -------
    npt.DTypeLike
        Real dtype corresponding to the input dtype. Always returns a
        `np.dtype` instance at runtime.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.typing import _to_real_dtype
    >>> _to_real_dtype(np.float32)
    dtype('float32')
    >>> _to_real_dtype(np.float64)
    dtype('float64')
    >>> _to_real_dtype(np.complex64)
    dtype('float32')
    >>> _to_real_dtype(np.complex128)
    dtype('float64')
    """
    return np.real(np.empty(0, dtype=dtype)).dtype


def _to_complex_dtype(dtype: npt.DTypeLike) -> npt.DTypeLike:
    """
    Convert any dtype to its corresponding complex dtype.

    If the input is a real dtype (e.g., `np.float32`), returns the
    corresponding complex dtype (e.g., `np.complex64`). If the input is
    already a complex dtype, returns it unchanged.

    Parameters
    ----------
    dtype : npt.DTypeLike
        Input dtype. Can be a dtype object, dtype string, or type object.

    Returns
    -------
    npt.DTypeLike
        Complex dtype corresponding to the input dtype. Always returns a
        `np.dtype` instance at runtime.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy.typing import _to_complex_dtype
    >>> _to_complex_dtype(np.float32)
    dtype('complex64')
    >>> _to_complex_dtype(np.float64)
    dtype('complex128')
    >>> _to_complex_dtype(np.complex64)
    dtype('complex64')
    >>> _to_complex_dtype(np.complex128)
    dtype('complex128')
    """
    return np.result_type(dtype, 1j)
