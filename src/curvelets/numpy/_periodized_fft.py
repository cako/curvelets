"""Periodized (folded / tiled) sparse FFT primitives for UDCT wedges.

These replace full-size ``fftn`` / ``ifftn`` on mostly-zero frequency bands
with mathematically equivalent small FFTs via Poisson summation / aliasing:

- Forward: ``downsample(ifftn(X), d) == ifftn(fold(X)) / prod(d)``
- Backward: ``fftn(upsample(y, d)) == tile(fftn(y), d)``
- Flip: ``flip_fft_all_axes`` is index remap ``coord -> (-coord) mod N``
- Half spectrum (real signals): the forward gathers full-grid samples from the
  ``rfftn`` half spectrum (``F[k] = conj(F[-k])`` for samples outside it), and
  the backward accumulates the Hermitian part ``H[k] = (F[k] + conj(F[-k])) / 2``
  on the half grid, whose ``irfftn`` equals ``ifftn(F).real``.

Higher-level fold / analyze / synthesize operations live on
:class:`~curvelets.numpy._sparse_window.SparseWindow`.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .typing import _FloatingNDArray, _IntpNDArray


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
    if len(shape) != len(dec):
        msg = "shape and decimation must have equal length"
        raise ValueError(msg)
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


def half_spectrum_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Shape of the ``rfftn`` half spectrum of a real array of shape ``shape``.

    Parameters
    ----------
    shape : tuple of int
        Full (real-space) array shape.

    Returns
    -------
    tuple of int
        ``(*shape[:-1], shape[-1] // 2 + 1)``.

    Examples
    --------
    >>> from curvelets.numpy._periodized_fft import half_spectrum_shape
    >>> half_spectrum_shape((8, 7))
    (8, 4)
    """
    return (*shape[:-1], shape[-1] // 2 + 1)


@dataclass(frozen=True)
class HalfSpectrumMaps:
    """
    Index maps of one window onto the ``rfftn`` half spectrum.

    Attributes
    ----------
    source : ndarray of intp
        Forward gather source of every window sample in the doubled half
        spectrum ``[F_half, conj(F_half)]`` (see :func:`half_spectrum_source`).
    own_indices, own_folded, own_values : ndarray
        Backward contributions whose own frequency lies in the half spectrum:
        half-grid target, folded source and weight (window value / 2).
    mirror_indices, mirror_folded, mirror_values : ndarray
        Backward contributions whose negated frequency lies in the half spectrum;
        they are accumulated separately and conjugated (see
        :func:`hermitian_from_half`).
    """

    source: _IntpNDArray
    own_indices: _IntpNDArray
    own_folded: _IntpNDArray
    own_values: _FloatingNDArray
    mirror_indices: _IntpNDArray
    mirror_folded: _IntpNDArray
    mirror_values: _FloatingNDArray


def compute_half_spectrum_maps(
    indices: npt.NDArray[np.intp],
    folded_indices: npt.NDArray[np.intp],
    values: npt.NDArray[np.floating],
    shape: tuple[int, ...],
) -> HalfSpectrumMaps:
    """
    Build the half-spectrum gather / scatter maps of a window.

    Parameters
    ----------
    indices : ndarray of intp
        Flat full-grid indices of the window samples.
    folded_indices : ndarray of intp
        Flat indices of the same samples in the decimated grid.
    values : ndarray of float
        Window values.
    shape : tuple of int
        Full array shape.

    Returns
    -------
    HalfSpectrumMaps
        Forward source and backward own / mirror contributions.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._periodized_fft import compute_half_spectrum_maps
    >>> idx = np.arange(16, dtype=np.intp)
    >>> maps = compute_half_spectrum_maps(idx, idx % 4, np.ones(16), (4, 4))
    >>> maps.source.shape, maps.own_indices.size, maps.mirror_indices.size
    ((16,), 12, 12)
    """
    half_shape = half_spectrum_shape(shape)
    half_numel = math.prod(half_shape)
    half_last = half_shape[-1]
    coords = tuple(np.asarray(c) for c in np.unravel_index(indices, shape))
    neg = tuple((-c) % n for c, n in zip(coords, shape, strict=True))
    own_in = coords[-1] < half_last
    neg_in = neg[-1] < half_last
    own = np.ravel_multi_index(
        (*coords[:-1], np.where(own_in, coords[-1], 0)), half_shape
    )
    mirror = np.ravel_multi_index((*neg[:-1], np.where(neg_in, neg[-1], 0)), half_shape)
    # Every sample is in the half or has its mirror there; mirrored ones are
    # read from the conjugated second half of [F_half, conj(F_half)].
    source = np.where(own_in, own, mirror + half_numel).astype(np.intp)
    return HalfSpectrumMaps(
        source=source,
        own_indices=own[own_in].astype(np.intp),
        own_folded=np.asarray(folded_indices[own_in], dtype=np.intp),
        own_values=0.5 * values[own_in],
        mirror_indices=mirror[neg_in].astype(np.intp),
        mirror_folded=np.asarray(folded_indices[neg_in], dtype=np.intp),
        mirror_values=0.5 * values[neg_in],
    )


def half_spectrum_source(
    half_spectrum: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.complexfloating]:
    """
    Flat ``[F_half, conj(F_half)]`` buffer gathered by :attr:`HalfSpectrumMaps.source`.

    Parameters
    ----------
    half_spectrum : ndarray of complex
        ``rfftn`` of a real array.

    Returns
    -------
    ndarray of complex
        1-D array of twice the half-spectrum size.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._periodized_fft import half_spectrum_source
    >>> half_spectrum_source(np.array([[1 + 2j]])).tolist()
    [(1+2j), (1-2j)]
    """
    flat = half_spectrum.reshape(-1)
    out = np.empty(2 * flat.size, dtype=half_spectrum.dtype)
    out[: flat.size] = flat
    np.conjugate(flat, out=out[flat.size :])
    return out


def hermitian_from_half(
    own: npt.NDArray[np.complexfloating],
    mirror: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.complexfloating]:
    """
    Combine half-grid accumulators into ``own + conj(mirror)`` (in place on ``own``).

    Parameters
    ----------
    own : ndarray of complex
        Accumulator of the own-frequency contributions (overwritten).
    mirror : ndarray of complex
        Accumulator of the mirrored contributions (overwritten).

    Returns
    -------
    ndarray of complex
        ``own``, now holding the Hermitian half spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._periodized_fft import hermitian_from_half
    >>> hermitian_from_half(np.array([1 + 1j]), np.array([1 + 1j])).tolist()
    [(2+0j)]
    """
    np.conjugate(mirror, out=mirror)
    own += mirror
    return own
