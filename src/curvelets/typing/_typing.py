from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# FDCTStructLike: TypeAlias = list[list[NDArray]]
AnyNDArray: TypeAlias = NDArray[np.generic]
ComplexNDArray: TypeAlias = Union[NDArray[np.complex64], NDArray[np.complex128]]
RecursiveListAnyNDArray: TypeAlias = Union[
    list[AnyNDArray], list["RecursiveListAnyNDArray"]
]
UDCTCoefficients: TypeAlias = list[list[list[ComplexNDArray]]]
