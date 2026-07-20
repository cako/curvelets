"""Sparse window implementation for NumPy UDCT."""

from __future__ import annotations

# pylint: disable=duplicate-code
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ._periodized_fft import (
    compute_folded_indices,
    decimated_shape,
    flip_fft_indices,
)
from .typing import _F, _FloatingNDArray, _IntpNDArray


@dataclass
class SparseWindow:
    """
    Sparse frequency-domain window for efficient curvelet operations.

    This class represents a sparse window as a collection of indices and values,
    and owns periodized (folded / tiled) FFT operations used by UDCT wedges.

    Parameters
    ----------
    indices : npt.NDArray[np.intp]
        Flat indices into array where window has non-zero values.
    values : npt.NDArray[np.floating]
        Window values at those indices.
    shape : tuple[int, ...]
        Original array shape (for to_dense conversion).
    folded_indices : npt.NDArray[np.intp], optional
        Flat indices into the decimated grid for periodized FFTs.
        Attached via :meth:`attach_periodized`.
    flipped_indices : npt.NDArray[np.intp], optional
        Flat indices after FFT-axis flip (complex negative-frequency wedges).
    flipped_folded_indices : npt.NDArray[np.intp], optional
        Folded indices corresponding to ``flipped_indices``.
    decimation : npt.NDArray[np.intp], optional
        Per-axis decimation ratios attached with periodized maps.
    out_shape : tuple of int, optional
        Decimated coefficient shape ``s // d`` per axis.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import SparseWindow
    >>> arr = np.array([[0.1, 0.9], [0.2, 0.8]])
    >>> window = SparseWindow.from_dense(arr, threshold=0.5)
    >>> window.size
    2
    >>> dense = window.to_dense()
    >>> dense.shape
    (2, 2)
    """

    indices: _IntpNDArray
    values: _FloatingNDArray
    shape: tuple[int, ...]
    folded_indices: _IntpNDArray | None = field(default=None, repr=False)
    flipped_indices: _IntpNDArray | None = field(default=None, repr=False)
    flipped_folded_indices: _IntpNDArray | None = field(default=None, repr=False)
    decimation: _IntpNDArray | None = field(default=None, repr=False)
    out_shape: tuple[int, ...] | None = field(default=None, repr=False)

    @property
    def size(self) -> int:
        """
        Number of non-zero elements.

        Returns
        -------
        int
            Number of non-zero elements in the sparse window.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.1, 0.9, 0.2]), threshold=0.5)
        >>> window.size
        1
        """
        return len(self.indices)

    @classmethod
    def from_dense(
        cls,
        arr: npt.NDArray[_F],
        threshold: float,
    ) -> SparseWindow:
        """
        Create SparseWindow from dense array using threshold.

        Parameters
        ----------
        arr : npt.NDArray[F]
            Input dense array.
        threshold : float
            Threshold for sparse storage (values above threshold are kept).

        Returns
        -------
        SparseWindow
            Sparse window containing only values above threshold.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> arr = np.array([[0.1, 0.9], [0.2, 0.8]])
        >>> window = SparseWindow.from_dense(arr, threshold=0.5)
        >>> window.size
        2
        """
        arr_flat = arr.ravel()
        indices = np.argwhere(arr_flat > threshold).ravel()
        values = arr_flat[indices]
        return cls(indices=indices, values=values, shape=arr.shape)

    def attach_periodized(
        self,
        decimation: npt.ArrayLike,
        *,
        with_flip: bool = False,
    ) -> None:
        """
        Attach decimation and precomputed folded (and optional flipped) index maps.

        Parameters
        ----------
        decimation : array_like
            Per-axis decimation ratios for this window.
        with_flip : bool, optional
            If True, also attach flipped index maps for complex mode.
            Default is False.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2], with_flip=True)
        >>> window.folded_indices is not None
        True
        >>> window.flipped_indices is not None
        True
        >>> window.out_shape
        (4, 4)
        """
        self.decimation = np.asarray(decimation, dtype=np.intp).ravel()
        self.out_shape = decimated_shape(self.shape, self.decimation)
        self.folded_indices = compute_folded_indices(
            self.indices, self.shape, self.decimation
        )
        if with_flip:
            self.flipped_indices = flip_fft_indices(self.indices, self.shape)
            self.flipped_folded_indices = compute_folded_indices(
                self.flipped_indices, self.shape, self.decimation
            )

    def resolve_indices(
        self,
        *,
        flip: bool = False,
        decimation: npt.ArrayLike | None = None,
    ) -> tuple[_IntpNDArray, _IntpNDArray]:
        """
        Return ``(indices, folded_indices)``, computing on the fly if needed.

        Parameters
        ----------
        flip : bool, optional
            If True, use flipped indices (complex negative-frequency wedges).
        decimation : array_like, optional
            Used only when folded maps are not yet attached.

        Returns
        -------
        indices : ndarray of intp
            Flat indices into the full-size frequency grid.
        folded_indices : ndarray of intp
            Flat indices into the decimated grid.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> idx, folded = window.resolve_indices()
        >>> len(idx) == len(folded) == window.size
        True
        """
        if flip:
            if self.flipped_indices is not None:
                indices = self.flipped_indices
            else:
                indices = flip_fft_indices(self.indices, self.shape)
            if self.flipped_folded_indices is not None:
                return indices, self.flipped_folded_indices
            dec = decimation if decimation is not None else self.decimation
            if dec is None:
                msg = "decimation required when flipped_folded_indices are not attached"
                raise ValueError(msg)
            return indices, compute_folded_indices(indices, self.shape, dec)

        indices = self.indices
        if self.folded_indices is not None:
            return indices, self.folded_indices
        dec = decimation if decimation is not None else self.decimation
        if dec is None:
            msg = "decimation required when folded_indices are not attached"
            raise ValueError(msg)
        return indices, compute_folded_indices(indices, self.shape, dec)

    def _resolved_out_shape(
        self, decimation: npt.ArrayLike | None = None
    ) -> tuple[int, ...]:
        if self.out_shape is not None:
            return self.out_shape
        dec = decimation if decimation is not None else self.decimation
        if dec is None:
            msg = "decimation required when out_shape is not attached"
            raise ValueError(msg)
        res: tuple[int, ...] = tuple(int(x) for x in decimated_shape(self.shape, dec))
        return res

    def fold_product(
        self,
        image_frequency: npt.NDArray[np.complexfloating],
        *,
        flip: bool = False,
        extra_at_indices: npt.NDArray | None = None,
        decimation: npt.ArrayLike | None = None,
        out: npt.NDArray[np.complexfloating] | None = None,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Fold ``values * image_frequency[indices]`` into the decimated frequency grid.

        Parameters
        ----------
        image_frequency : ndarray of complex
            Full-size frequency-domain image.
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : ndarray, optional
            Extra multiplicative weights already aligned with window indices
            (e.g. Riesz filter samples).
        decimation : array_like, optional
            Fallback when periodized maps are not attached.
        out : ndarray of complex, optional
            Reusable output buffer (cleared before use).

        Returns
        -------
        ndarray of complex
            Folded spectrum of shape ``out_shape``.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> image_freq = np.ones((4, 4), dtype=np.complex128)
        >>> folded = window.fold_product(image_freq)
        >>> folded.shape
        (2, 2)
        """
        indices, folded_indices = self.resolve_indices(flip=flip, decimation=decimation)
        out_shape = self._resolved_out_shape(decimation)
        if out is None:
            folded = np.zeros(out_shape, dtype=image_frequency.dtype)
        else:
            folded = out
            folded.fill(0)
        vals = image_frequency.flat[indices] * self.values.astype(image_frequency.dtype)
        if extra_at_indices is not None:
            vals = vals * extra_at_indices.astype(image_frequency.dtype)
        np.add.at(folded.ravel(), folded_indices, vals)
        return folded

    def scatter_tiled(
        self,
        small_fft: npt.NDArray[np.complexfloating],
        target: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        flip: bool = False,
        decimation: npt.ArrayLike | None = None,
    ) -> None:
        """
        Scatter ``scale * values * tile(small_fft)`` into ``target``.

        Parameters
        ----------
        small_fft : ndarray of complex
            ``fftn`` of the coefficient array.
        target : ndarray of complex
            Full-size frequency accumulator (modified in-place).
        scale : float
            Combined normalization / mode scale factor.
        flip : bool, optional
            If True, use flipped indices.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> target = np.zeros((4, 4), dtype=np.complex128)
        >>> small = np.ones((2, 2), dtype=np.complex128)
        >>> window.scatter_tiled(small, target, scale=1.0)
        >>> np.any(target != 0)
        True
        """
        indices, folded_indices = self.resolve_indices(flip=flip, decimation=decimation)
        target.flat[indices] += (
            small_fft.flat[folded_indices] * self.values.astype(target.dtype) * scale
        )

    def analyze(
        self,
        image_frequency: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        flip: bool = False,
        extra_at_indices: npt.NDArray | None = None,
        decimation: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Forward periodized wedge: fold → small IFFT → scale.

        Parameters
        ----------
        image_frequency : ndarray of complex
            Full-size frequency-domain image.
        scale : float
            Combined normalization factor applied after the small IFFT.
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : ndarray, optional
            Extra weights aligned with window indices (e.g. Riesz samples).
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Returns
        -------
        ndarray of complex
            Decimated coefficient array.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> image_freq = np.fft.fftn(np.random.randn(8, 8))
        >>> coeff = window.analyze(image_freq, scale=1.0)
        >>> coeff.shape
        (4, 4)
        """
        folded = self.fold_product(
            image_frequency,
            flip=flip,
            extra_at_indices=extra_at_indices,
            decimation=decimation,
        )
        return scale * np.fft.ifftn(folded)

    def synthesize(
        self,
        coefficient: npt.NDArray[np.complexfloating],
        target: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        flip: bool = False,
        decimation: npt.ArrayLike | None = None,
    ) -> None:
        """
        Backward periodized wedge: small FFT → tiled sparse scatter into ``target``.

        Parameters
        ----------
        coefficient : ndarray of complex
            Decimated coefficient array.
        target : ndarray of complex
            Full-size frequency accumulator (modified in-place).
        scale : float
            Combined normalization / mode scale factor.
        flip : bool, optional
            If True, use flipped indices.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> target = np.zeros((8, 8), dtype=np.complex128)
        >>> coeff = np.ones((4, 4), dtype=np.complex128)
        >>> window.synthesize(coeff, target, scale=1.0)
        >>> np.any(target != 0)
        True
        """
        self.scatter_tiled(
            np.fft.fftn(coefficient),
            target,
            scale,
            flip=flip,
            decimation=decimation,
        )

    def multiply_extract(
        self,
        arr: npt.NDArray,
        out: npt.NDArray | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray:
        """
        Extract values at indices, multiply by window, return dense result.

        This method replaces the pattern:
        ``freq_band.flat[idx] = image_frequency.flat[idx] * val``

        Parameters
        ----------
        arr : npt.NDArray
            Input array to extract values from.
        out : npt.NDArray | None, optional
            Output array to write results into. If None, a new array is created.
        dtype : npt.DTypeLike | None, optional
            Dtype for output array. If None, uses arr.dtype.

        Returns
        -------
        npt.NDArray
            Dense array with extracted and multiplied values.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.0, 0.5, 1.0]), threshold=0.3)
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> result = window.multiply_extract(arr)
        >>> result[1]  # Only index 1 and 2 are in window
        1.0
        """
        if out is None:
            out = np.zeros(self.shape, dtype=dtype or arr.dtype)
        else:
            out.fill(0)
        values = self.values.astype(out.dtype)
        out.flat[self.indices] = arr.flat[self.indices] * values
        return out

    def scatter_add(
        self,
        target: npt.NDArray,
        source: npt.NDArray,
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """
        Add source*window to target at window indices (in-place).

        This method replaces the pattern:
        ``target.flat[idx] += source.flat[idx] * val``

        Parameters
        ----------
        target : npt.NDArray
            Target array to accumulate into (modified in-place).
        source : npt.NDArray
            Source array to extract values from.
        dtype : npt.DTypeLike | None, optional
            Dtype for conversion. If None, uses target.dtype.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.0, 0.5, 1.0]), threshold=0.3)
        >>> target = np.zeros(3)
        >>> source = np.array([1.0, 2.0, 3.0])
        >>> window.scatter_add(target, source)
        >>> target[1]  # Only indices 1 and 2 are accumulated
        1.0
        """
        values = self.values.astype(dtype or target.dtype)
        target.flat[self.indices] += source.flat[self.indices] * values

    def multiply_at_indices(
        self,
        source: npt.NDArray,
        filter_arr: npt.NDArray,
        out: npt.NDArray | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray:
        """
        Multiply source values at window indices by filter values at same indices.

        This method replaces the pattern:
        ``out.flat[idx] = source.flat[idx] * filter_arr.flat[idx]``

        Parameters
        ----------
        source : npt.NDArray
            Source array (typically already windowed).
        filter_arr : npt.NDArray
            Filter array to multiply with (e.g., Riesz filter).
        out : npt.NDArray | None, optional
            Output array to write results into. If None, a new array is created.
        dtype : npt.DTypeLike | None, optional
            Dtype for output array. If None, uses source.dtype.

        Returns
        -------
        npt.NDArray
            Dense array with multiplied values at window indices.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.0, 0.5, 1.0]), threshold=0.3)
        >>> source = np.array([1.0, 2.0, 3.0])
        >>> filter_arr = np.array([0.5, 1.5, 2.5])
        >>> result = window.multiply_at_indices(source, filter_arr)
        >>> result[1]  # Only indices 1 and 2 are in window
        3.0
        """
        if out is None:
            out = np.zeros(self.shape, dtype=dtype or source.dtype)
        else:
            out.fill(0)
        idx_flat = self.indices.ravel()
        out.flat[idx_flat] = (source.flat[idx_flat] * filter_arr.flat[idx_flat]).astype(
            out.dtype
        )
        return out

    def to_dense(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray:
        """
        Convert to dense array.

        This method replaces the pattern:
        ``arr.flat[idx] = val``

        Parameters
        ----------
        dtype : npt.DTypeLike | None, optional
            Dtype for output array. If None, uses self.values.dtype.

        Returns
        -------
        npt.NDArray
            Dense array representation of the sparse window.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.1, 0.9, 0.2]), threshold=0.5)
        >>> dense = window.to_dense()
        >>> dense[1]  # Only index 1 is above threshold
        0.9
        """
        dtype = dtype or self.values.dtype
        arr = np.zeros(self.shape, dtype=dtype)
        arr.flat[self.indices] = self.values.astype(dtype)
        return arr
