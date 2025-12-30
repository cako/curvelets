"""Type definitions for PyTorch UDCT implementation.

This module provides simple type aliases using torch.Tensor everywhere.
Unlike the NumPy version, we don't use complex TypeVars since PyTorch
tensors handle dtype internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._sparse_window import SparseWindow


# Simple type aliases - every array is just a Tensor
# Structure: coefficients[scale][direction][wedge] = Tensor
UDCTCoefficients = list[list[list[torch.Tensor]]]

# MUDCTCoefficients: Monogenic UDCT coefficients
# Structure: coefficients[scale][direction][wedge] = list[Tensor]
# Each wedge contains ndim+1 components: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
MUDCTCoefficients = list[list[list[list[torch.Tensor]]]]

# Structure: windows[scale][direction][wedge] = SparseWindow
UDCTWindows = list[list[list["SparseWindow"]]]

# Integer tensor type alias for decimation ratios, indices, etc.
IntegerTensor = torch.Tensor
