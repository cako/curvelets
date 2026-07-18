"""Type definitions for PyTorch UDCT implementation.

This module provides type aliases for UDCT coefficients and windows.

Public Types
------------
UDCTCoefficients
    Type alias for UDCT coefficient structure.
UDCTWindows
    Type alias for UDCT window structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

__all__ = [
    "UDCTCoefficients",
    "UDCTWindows",
]

# =============================================================================
# Public Type Aliases
# =============================================================================


# Simple type aliases - every array is just a Tensor
# Structure: coefficients[scale][direction][wedge] = Tensor
# For monogenic transforms, each wedge tensor has shape (*wedge_shape, ndim+1)
UDCTCoefficients = list[list[list[torch.Tensor]]]

if TYPE_CHECKING:
    from ._sparse_window import SparseWindow
else:
    SparseWindow = Any

# Structure: windows[scale][direction][wedge] = SparseWindow
UDCTWindows = list[list[list[SparseWindow]]]

# =============================================================================
# Private Type Aliases (internal use only)
# =============================================================================

# Integer tensor type alias for decimation ratios, indices, etc.
_IntegerTensor = torch.Tensor
