from __future__ import annotations

import sys
from typing import TypeVar, Union

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

DTypeG = TypeVar("DTypeG", bound=np.generic)
DTypeI = TypeVar(
    "DTypeI",
    np.intp,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)
DTypeF = TypeVar(
    "DTypeF",
    np.float32,
    np.float64,
    np.float128,
)
DTypeC = TypeVar(
    "DTypeC",
    np.complex64,
    np.complex128,
    np.complex256,
)

# FDCTStructLike: TypeAlias = list[list[NDArray]]
AnyNDArray: TypeAlias = NDArray[np.generic]
IntNDArray: TypeAlias = Union[
    NDArray[np.int_],
    NDArray[np.int8],
    NDArray[np.int16],
    NDArray[np.int64],
]
FloatNDArray: TypeAlias = Union[
    NDArray[np.float32],
    NDArray[np.float64],
    NDArray[np.float128],
]
ComplexNDArray: TypeAlias = Union[
    NDArray[np.complex64],
    NDArray[np.complex128],
    NDArray[np.complex256],
]
RecursiveListAnyNDArray: TypeAlias = Union[
    list[AnyNDArray], list["RecursiveListAnyNDArray"]
]
UDCTCoefficients: TypeAlias = list[list[list[ComplexNDArray]]]
UDCTWindows: TypeAlias = list[list[list[list[FloatNDArray]]]]
