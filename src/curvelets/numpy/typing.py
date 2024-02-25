from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt

UDCTCoefficients: TypeAlias = list[list[list[npt.NDArray[np.complexfloating]]]]

UDCTWindows: TypeAlias = list[list[list[list[npt.NDArray[np.floating]]]]]
