"""Precision handling for the torch UDCT (its fixed ``dtype``)."""

from __future__ import annotations

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float64)


def resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Real precision for a ``dtype`` argument (default: torch's default dtype).

    A complex dtype names the precision of its parts: ``complex64`` means
    ``float32``.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype.is_complex:
        dtype = dtype.to_real()
    if dtype not in SUPPORTED_DTYPES:
        msg = f"dtype must be torch.float32 or torch.float64, got {dtype}"
        raise ValueError(msg)
    return dtype


def check_dtype(tensor: torch.Tensor, dtype: torch.dtype, what: str) -> None:
    """Reject an ``input`` or ``coefficients`` tensor of another precision."""
    if tensor.dtype in (dtype, dtype.to_complex()):
        return
    msg = f"UDCT was built for {dtype} but got {tensor.dtype} {what}."
    if what == "input" and tensor.dtype in SUPPORTED_DTYPES:
        msg += (
            f" Cast the input, or build the transform with dtype={tensor.dtype}"
            " (or call .to(dtype) on it)."
        )
    raise TypeError(msg)
