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

# Structure: windows[scale][direction][wedge] = (indices, values) tuple
UDCTWindows = list[list[list[tuple[torch.Tensor, torch.Tensor]]]]

# =============================================================================
# Private Type Aliases (internal use only)
# =============================================================================

# Integer tensor type alias for decimation ratios, indices, etc.
_IntegerTensor = torch.Tensor
