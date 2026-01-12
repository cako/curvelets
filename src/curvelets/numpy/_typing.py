from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeVar
else:
    from typing_extensions import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAliasType

# TypeVars for numpy array dtypes (using bound per NumPy recommendation)
# F: Real floating point types (np.floating and all subtypes)
F = TypeVar("F", bound=np.floating)

# C: Complex floating point types (np.complexfloating and all subtypes)
C = TypeVar("C", bound=np.complexfloating)

# T: Any scalar type (real or complex) for generic coefficient type
# np.inexact is the common base class for np.floating and np.complexfloating
T = TypeVar("T", bound=np.inexact)

# Generic UDCT coefficients parameterized by scalar dtype
# Usage:
#   - UDCTCoefficients[np.floating] or UDCTCoefficients[F] for monogenic transform
#   - UDCTCoefficients[np.complexfloating] or UDCTCoefficients[C] for real/complex transforms
UDCTCoefficients = TypeAliasType(
    "UDCTCoefficients",
    list[list[list[npt.NDArray[T]]]],
    type_params=(T,),
)

# Generic UDCT windows parameterized by real floating dtype (no complex allowed)
# Usage: UDCTWindows[np.floating], UDCTWindows[np.float32], UDCTWindows[np.float64]
# Structure: list[list[list[tuple[indices, values]]]] where values are real floats
UDCTWindows = TypeAliasType(
    "UDCTWindows",
    list[list[list[tuple[npt.NDArray[np.intp], npt.NDArray[F]]]]],
    type_params=(F,),
)

# A: Any numpy array type (includes all numpy scalar types: floating, complex, integer, bool, etc.)
A = TypeVar("A", bound=np.generic)

# Type aliases for common NDArray types
# Note: These are kept for backward compatibility but new code should use TypeVars directly
FloatingNDArray: TypeAlias = npt.NDArray[np.floating]
IntegerNDArray: TypeAlias = npt.NDArray[np.int_]
IntpNDArray: TypeAlias = npt.NDArray[np.intp]
BoolNDArray: TypeAlias = npt.NDArray[np.bool_]


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
    >>> from curvelets.numpy._typing import _to_real_dtype
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
    >>> from curvelets.numpy._typing import _to_complex_dtype
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
