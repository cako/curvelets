from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeVar
else:
    from typing_extensions import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAliasType

# TypeVars for numpy array dtypes
# F: Real floating point types
F = TypeVar("F", np.float16, np.float32, np.float64, np.longdouble)

# C: Complex floating point types
# Note: complex256 is available on some platforms but not others (e.g., not on macOS/Apple Silicon)
# We define C with the common types; complex256 can be used directly when available
C = TypeVar("C", np.complex64, np.complex128)

# T: Any scalar type (real or complex) for generic coefficient type
T = TypeVar(
    "T",
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
    np.complex64,
    np.complex128,
)

# Generic UDCT coefficients parameterized by scalar dtype
# Usage:
#   - UDCTCoefficients[np.float64] for monogenic transform (real dtype, ndim+2 channels)
#   - UDCTCoefficients[np.complex128] for real/complex transforms (complex dtype)
UDCTCoefficients = TypeAliasType(
    "UDCTCoefficients",
    list[list[list[npt.NDArray[T]]]],
    type_params=(T,),
)

# Generic UDCT windows parameterized by real floating dtype (no complex allowed)
# Usage: UDCTWindows[np.float32], UDCTWindows[np.float64]
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

# Coefficient type usage with UDCTCoefficients[T]:
# - Real/Complex transforms: UDCTCoefficients[C] (complex dtype)
#   - NDArray[np.float32] input -> UDCTCoefficients[np.complex64]
#   - NDArray[np.float64] input -> UDCTCoefficients[np.complex128]
#   - NDArray[np.complex64] input -> UDCTCoefficients[np.complex64]
#   - NDArray[np.complex128] input -> UDCTCoefficients[np.complex128]
# - Monogenic transform: UDCTCoefficients[F] (real dtype) with ndim+2 channels
#   - NDArray[np.float32] input -> UDCTCoefficients[np.float32]
#   - NDArray[np.float64] input -> UDCTCoefficients[np.float64]
#   - Channels: [scalar.real, scalar.imag, riesz_1, riesz_2, ..., riesz_ndim]
#   - Complex scalar reconstructed via .view(complex_dtype) on first 2 channels


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
