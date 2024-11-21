from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# FDCTStructLike: TypeAlias = list[list[NDArray]]
AnyNDArray: TypeAlias = NDArray[np.generic]
RecursiveListAnyNDArray: TypeAlias = Union[
    list[AnyNDArray], list["RecursiveListAnyNDArray"]
]
