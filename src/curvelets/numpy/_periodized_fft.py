"""Periodized (folded / tiled) sparse FFT primitives for UDCT wedges.

These replace full-size ``fftn`` / ``ifftn`` on mostly-zero frequency bands
with mathematically equivalent small FFTs via Poisson summation / aliasing:

- Forward: ``downsample(ifftn(X), d) == ifftn(fold(X)) / prod(d)``
- Backward: ``fftn(upsample(y, d)) == tile(fftn(y), d)``
- Flip: ``flip_fft_all_axes`` is index remap ``coord -> (-coord) mod N``

Higher-level fold / analyze / synthesize operations live on
:class:`~curvelets.numpy._sparse_window.SparseWindow`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .typing import _IntpNDArray


def flip_fft_indices(
    indices: npt.NDArray[np.intp],
    shape: tuple[int, ...],
) -> _IntpNDArray:
    """
    Map flat indices through ``flip_fft_all_axes``: coord -> (-coord) mod N.

    Parameters
    ----------
    indices : ndarray of intp
        Flat indices into an array of shape ``shape``.
    shape : tuple of int
        Full array shape.

    Returns
    -------
    ndarray of intp
        Flat indices after FFT-axis flip on every axis.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._periodized_fft import flip_fft_indices
    >>> from curvelets.numpy._utils import flip_fft_all_axes
    >>> shape = (4, 4)
    >>> dense = np.zeros(shape)
    >>> dense[1, 2] = 1.0
    >>> idx = np.array([np.ravel_multi_index((1, 2), shape)], dtype=np.intp)
    >>> flipped_idx = flip_fft_indices(idx, shape)
    >>> dense_flip = flip_fft_all_axes(dense)
    >>> dense_flip.flat[flipped_idx[0]]
    1.0
    """
    coords = np.unravel_index(indices, shape)
    flipped = tuple((-c) % n for c, n in zip(coords, shape, strict=True))
    return np.asarray(np.ravel_multi_index(flipped, shape), dtype=np.intp)


def decimated_shape(
    shape: tuple[int, ...],
    decimation: npt.ArrayLike,
) -> tuple[int, ...]:
    """
    Compute decimated (small) shape from full shape and decimation ratios.

    Parameters
    ----------
    shape : tuple of int
        Full array shape.
    decimation : array_like
        Per-axis decimation ratios.

    Returns
    -------
    tuple of int
        Shape after decimation (``s // d`` per axis).

    Examples
    --------
    >>> from curvelets.numpy._periodized_fft import decimated_shape
    >>> decimated_shape((256, 256), [4, 4])
    (64, 64)
    """
    dec = np.asarray(decimation, dtype=np.intp).ravel()
    return tuple(int(s // int(d)) for s, d in zip(shape, dec, strict=True))


def compute_folded_indices(
    indices: npt.NDArray[np.intp],
    shape: tuple[int, ...],
    decimation: npt.ArrayLike,
) -> _IntpNDArray:
    """
    Map full-grid flat indices to flat indices in the decimated (folded) grid.

    Parameters
    ----------
    indices : ndarray of intp
        Flat indices into the full-size array.
    shape : tuple of int
        Full array shape.
    decimation : array_like
        Per-axis decimation ratios.

    Returns
    -------
    ndarray of intp
        Flat indices into the decimated array of shape
        ``decimated_shape(shape, decimation)``.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._periodized_fft import compute_folded_indices
    >>> shape = (8, 8)
    >>> idx = np.array([np.ravel_multi_index((5, 6), shape)], dtype=np.intp)
    >>> compute_folded_indices(idx, shape, [2, 2])
    array([6])
    """
    out_shape = decimated_shape(shape, decimation)
    unravel = np.unravel_index(indices, shape)
    folded = tuple(u % mi for u, mi in zip(unravel, out_shape, strict=True))
    return np.asarray(np.ravel_multi_index(folded, out_shape), dtype=np.intp)
