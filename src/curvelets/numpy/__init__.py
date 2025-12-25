from __future__ import annotations

__all__ = ["UDCT", "MeyerWavelet", "ParamUDCT", "UDCTCoefficients", "UDCTWindows", "SimpleUDCT"]

from ._udct import UDCT
from ._meyerwavelet import MeyerWavelet
from ._utils import ParamUDCT
from ._typing import UDCTCoefficients, UDCTWindows

# Compatibility alias: SimpleUDCT uses num_scales and wedges_per_direction
# instead of nscales and nbands_per_direction
SimpleUDCT = UDCT
