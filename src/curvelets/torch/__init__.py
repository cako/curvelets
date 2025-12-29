"""PyTorch implementation of Uniform Discrete Curvelet Transform (UDCT)."""

from __future__ import annotations

from ._meyerwavelet import MeyerWavelet
from ._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows
from ._udct import UDCT
from ._udct_module import UDCTModule

__all__ = [
    "UDCT",
    "UDCTModule",
    "MeyerWavelet",
    "UDCTCoefficients",
    "UDCTWindows",
    "MUDCTCoefficients",
]
