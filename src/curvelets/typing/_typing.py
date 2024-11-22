from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

DTypeG = TypeVar("DTypeG", bound=np.generic)
DTypeI = TypeVar("DTypeI", np.intp, np.int8, np.int16, np.int32, np.int64, np.int128)
DTypeF = TypeVar("DTypeF", np.float16, np.float64, np.float128)
DTypeC = TypeVar("DTypeC", np.complex64, np.complex128, np.complex256)

# FDCTStructLike: TypeAlias = list[list[NDArray]]
AnyNDArray: TypeAlias = NDArray[np.generic]
IntNDArray: TypeAlias = NDArray[np.int_]
ComplexNDArray: TypeAlias = Union[NDArray[np.complex64], NDArray[np.complex128]]
RecursiveListAnyNDArray: TypeAlias = Union[
    list[AnyNDArray], list["RecursiveListAnyNDArray"]
]
UDCTCoefficients: TypeAlias = list[list[list[ComplexNDArray]]]
