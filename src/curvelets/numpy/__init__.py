from __future__ import annotations

# pylint: disable=duplicate-code
# Similar __all__ to torch.__init__ is intentional - both modules export the same API
__all__ = [
    "UDCT",
    "MeyerWavelet",
    "UDCTCoefficients",
    "UDCTWindows",
]

from ._meyerwavelet import MeyerWavelet
from ._typing import UDCTCoefficients, UDCTWindows
from ._udct import UDCT
