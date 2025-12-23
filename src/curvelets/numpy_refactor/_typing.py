from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeGuard
else:
    from typing_extensions import TypeAlias, TypeGuard


import numpy as np
import numpy.typing as npt

if sys.version_info <= (3, 9):
    from typing import List, Tuple  # noqa: UP035

    UDCTCoefficients: TypeAlias = List[List[List[npt.NDArray[np.complexfloating]]]]  # noqa: UP006
    UDCTWindows: TypeAlias = List[  # noqa: UP006
        List[List[Tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]]]  # noqa: UP006
    ]
else:
    UDCTCoefficients: TypeAlias = list[list[list[npt.NDArray[np.complexfloating]]]]
    UDCTWindows: TypeAlias = list[
        list[list[tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]]]
    ]

# Type aliases for common NDArray types
# Note: np.floating does NOT include np.complexfloating - they are separate base classes
FloatingNDArray: TypeAlias = npt.NDArray[np.floating]
ComplexFloatingNDArray: TypeAlias = npt.NDArray[np.complexfloating]
IntegerNDArray: TypeAlias = npt.NDArray[np.int_]
IntpNDArray: TypeAlias = npt.NDArray[np.intp]
BoolNDArray: TypeAlias = npt.NDArray[np.bool_]


def _is_complex_array(
    image: FloatingNDArray | ComplexFloatingNDArray,
) -> TypeGuard[ComplexFloatingNDArray]:
    """
    Type guard to check if an array is complex-valued.

    Parameters
    ----------
    image : FloatingNDArray | ComplexFloatingNDArray
        Array to check.

    Returns
    -------
    TypeGuard[ComplexFloatingNDArray]
        True if the array is complex-valued, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._typing import _is_complex_array
    >>> arr_complex = np.array([1+2j, 3+4j])
    >>> arr_real = np.array([1.0, 2.0])
    >>> _is_complex_array(arr_complex)
    True
    >>> _is_complex_array(arr_real)
    False
    """
    return np.iscomplexobj(image)


def _is_floating_array(
    image: FloatingNDArray | ComplexFloatingNDArray,
) -> TypeGuard[FloatingNDArray]:
    """
    Type guard to check if an array is real-valued (floating point).

    Parameters
    ----------
    image : FloatingNDArray | ComplexFloatingNDArray
        Array to check.

    Returns
    -------
    TypeGuard[FloatingNDArray]
        True if the array is real-valued, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._typing import _is_floating_array
    >>> arr_real = np.array([1.0, 2.0])
    >>> arr_complex = np.array([1+2j, 3+4j])
    >>> _is_floating_array(arr_real)
    True
    >>> _is_floating_array(arr_complex)
    False
    """
    return not np.iscomplexobj(image)
