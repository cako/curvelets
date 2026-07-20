"""Periodized (folded / tiled) sparse FFT primitives for PyTorch UDCT wedges.

These replace full-size ``torch.fft.fftn`` / ``torch.fft.ifftn`` on mostly-zero
frequency bands with mathematically equivalent small FFTs via Poisson summation / aliasing:

- Forward: ``downsample(ifftn(X), d) == ifftn(fold(X)) / prod(d)``
- Backward: ``fftn(upsample(y, d)) == tile(fftn(y), d)``
- Flip: ``flip_fft_all_axes`` is index remap ``coord -> (-coord) mod N``

Higher-level fold / analyze / synthesize operations live on
:class:`~curvelets.torch._sparse_window.SparseWindow`.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
from typing import Any

import torch


def _unravel_index(
    indices: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Helper to unravel flat indices into multidimensional coordinates."""
    if hasattr(torch, "unravel_index"):
        res: tuple[torch.Tensor, ...] = torch.unravel_index(indices, shape)
        return res
    coords = []
    rem = indices
    for s in reversed(shape[1:]):
        coords.append(rem % s)
        rem = rem // s
    coords.append(rem)
    coords.reverse()
    return tuple(coords)


def _ravel_multi_index(
    coords: tuple[torch.Tensor, ...], shape: tuple[int, ...]
) -> torch.Tensor:
    """Helper to ravel multidimensional coordinates into flat indices."""
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    strides.reverse()
    flat = torch.zeros_like(coords[0])
    for c, s in zip(coords, strides):
        flat = flat + c * s
    return flat


def flip_fft_indices(
    indices: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """
    Map flat indices through ``flip_fft_all_axes``: coord -> (-coord) mod N.

    Parameters
    ----------
    indices : torch.Tensor
        Flat indices into an array/tensor of shape ``shape``.
    shape : tuple of int
        Full tensor shape.

    Returns
    -------
    torch.Tensor
        Flat indices after FFT-axis flip on every axis.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch._periodized_fft import flip_fft_indices
    >>> from curvelets.torch._utils import flip_fft_all_axes
    >>> shape = (4, 4)
    >>> dense = torch.zeros(shape)
    >>> dense[1, 2] = 1.0
    >>> idx = torch.tensor([1 * shape[1] + 2], dtype=torch.long)
    >>> flipped_idx = flip_fft_indices(idx, shape)
    >>> dense_flip = flip_fft_all_axes(dense)
    >>> dense_flip.flatten()[flipped_idx[0]].item()
    1.0
    """
    if indices.numel() == 0:
        return indices.clone()
    coords = _unravel_index(indices, shape)
    flipped = tuple((-c) % n for c, n in zip(coords, shape))
    return _ravel_multi_index(flipped, shape).to(indices.dtype)


def decimated_shape(
    shape: tuple[int, ...],
    decimation: Any,
) -> tuple[int, ...]:
    """
    Compute decimated (small) shape from full shape and decimation ratios.

    Parameters
    ----------
    shape : tuple of int
        Full tensor shape.
    decimation : array_like or torch.Tensor
        Per-axis decimation ratios.

    Returns
    -------
    tuple of int
        Shape after decimation (``s // d`` per axis).

    Examples
    --------
    >>> from curvelets.torch._periodized_fft import decimated_shape
    >>> decimated_shape((256, 256), [4, 4])
    (64, 64)
    """
    if isinstance(decimation, torch.Tensor):
        dec = [int(x.item()) for x in decimation.flatten()]
    else:
        dec = [int(x) for x in decimation]
    if len(shape) != len(dec):
        msg = "shape and decimation must have equal length"
        raise ValueError(msg)
    return tuple(int(s // int(d)) for s, d in zip(shape, dec))


def compute_folded_indices(
    indices: torch.Tensor,
    shape: tuple[int, ...],
    decimation: Any,
) -> torch.Tensor:
    """
    Map full-grid flat indices to flat indices in the decimated (folded) grid.

    Parameters
    ----------
    indices : torch.Tensor
        Flat indices into the full-size tensor.
    shape : tuple of int
        Full tensor shape.
    decimation : array_like or torch.Tensor
        Per-axis decimation ratios.

    Returns
    -------
    torch.Tensor
        Flat indices into the decimated tensor of shape
        ``decimated_shape(shape, decimation)``.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch._periodized_fft import compute_folded_indices
    >>> shape = (8, 8)
    >>> idx = torch.tensor([5 * shape[1] + 6], dtype=torch.long)
    >>> compute_folded_indices(idx, shape, [2, 2])
    tensor([6])
    """
    out_shape = decimated_shape(shape, decimation)
    unravel = _unravel_index(indices, shape)
    folded = tuple(u % mi for u, mi in zip(unravel, out_shape))
    return _ravel_multi_index(folded, out_shape).to(indices.dtype)
