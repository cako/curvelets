from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeVar
else:
    from typing_extensions import TypeAlias, TypeVar


import numpy as np
import numpy.typing as npt

# TypeVars for numpy array dtypes
# F: Real floating point types
F = TypeVar("F", np.float16, np.float32, np.float64, np.longdouble)

if sys.version_info < (3, 10):
    from typing import List, Tuple, Union  # noqa: UP035

    UDCTCoefficients: TypeAlias = List[List[List[npt.NDArray[np.complexfloating]]]]  # noqa: UP006
    UDCTWindows: TypeAlias = List[  # noqa: UP006
        List[List[Tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]]]  # noqa: UP006
    ]
    # MUDCTCoefficients: variable-length list structure
    # Each coefficient is: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
    # where scalar is complex, all riesz components are real
    MUDCTCoefficients: TypeAlias = List[  # noqa: UP006
        List[List[List[npt.NDArray[Union[np.complexfloating, F]]]]]  # noqa: UP006
    ]
else:
    from typing import Union

    UDCTCoefficients: TypeAlias = list[list[list[npt.NDArray[np.complexfloating]]]]
    UDCTWindows: TypeAlias = list[
        list[list[tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]]]
    ]
    # MUDCTCoefficients: variable-length list structure
    # Each coefficient is: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
    # where scalar is complex, all riesz components are real
    MUDCTCoefficients: TypeAlias = list[
        list[list[list[npt.NDArray[Union[np.complexfloating, F]]]]]
    ]

# C: Complex floating point types
# Note: complex256 is available on some platforms but not others (e.g., not on macOS/Apple Silicon)
# We define C with the common types; complex256 can be used directly when available
C = TypeVar("C", np.complex64, np.complex128)

# A: Any numpy array type (includes all numpy scalar types: floating, complex, integer, bool, etc.)
A = TypeVar("A", bound=np.generic)

# Type aliases for common NDArray types
# Note: These are kept for backward compatibility but new code should use TypeVars directly
FloatingNDArray: TypeAlias = npt.NDArray[np.floating]
IntegerNDArray: TypeAlias = npt.NDArray[np.int_]
IntpNDArray: TypeAlias = npt.NDArray[np.intp]
BoolNDArray: TypeAlias = npt.NDArray[np.bool_]

# Note: UDCTCoefficients is always complex dtype (even in "real" transform mode)
# For generic usage, use list[list[list[npt.NDArray[C]]]] directly in function signatures
# where C is the complex TypeVar matching the input dtype precision:
# - NDArray[np.float32] -> list[list[list[npt.NDArray[np.complex64]]]]
# - NDArray[np.float64] -> list[list[list[npt.NDArray[np.complex128]]]]
# - NDArray[np.complex64] -> list[list[list[npt.NDArray[np.complex64]]]]
# - NDArray[np.complex128] -> list[list[list[npt.NDArray[np.complex128]]]]


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
