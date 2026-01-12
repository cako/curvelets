"""Type definitions for PyTorch UDCT implementation.

This module provides simple type aliases using torch.Tensor everywhere.
Unlike the NumPy version, we don't use complex TypeVars since PyTorch
tensors handle dtype internally.
"""

from __future__ import annotations

import torch

# Simple type aliases - every array is just a Tensor
# Structure: coefficients[scale][direction][wedge] = Tensor
# For monogenic transforms, each wedge tensor has shape (*wedge_shape, ndim+1)
UDCTCoefficients = list[list[list[torch.Tensor]]]

# Structure: windows[scale][direction][wedge] = (indices, values) tuple
UDCTWindows = list[list[list[tuple[torch.Tensor, torch.Tensor]]]]

# Integer tensor type alias for decimation ratios, indices, etc.
IntegerTensor = torch.Tensor
