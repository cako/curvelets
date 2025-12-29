"""PyTorch implementation of Uniform Discrete Curvelet Transform (UDCT)."""

from __future__ import annotations

from .._internal import TORCH_ENABLED

if not TORCH_ENABLED:
    _error_msg = (
        "PyTorch is not installed. To use the torch implementation of curvelets, "
        "please install PyTorch by running:\n\n"
        "    pip install curvelets[torch]\n\n"
        "or\n\n"
        "    pip install torch\n\n"
        "For more information, see: https://pytorch.org/get-started/locally/"
    )
    raise ImportError(_error_msg)

from ._meyerwavelet import MeyerWavelet
from ._typing import UDCTCoefficients, UDCTWindows
from ._udct import UDCT
from ._udct_module import UDCTModule

__all__ = [
    "UDCT",
    "MeyerWavelet",
    "UDCTCoefficients",
    "UDCTModule",
    "UDCTWindows",
]
