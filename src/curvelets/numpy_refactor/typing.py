from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


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
