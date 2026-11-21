"""Sparse window implementation for NumPy UDCT."""

from __future__ import annotations

# pylint: disable=duplicate-code
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ._fft import FFTBackend
from ._periodized_fft import (
    HalfSpectrumMaps,
    compute_folded_indices,
    compute_half_spectrum_maps,
    decimated_shape,
    flip_fft_indices,
)
from .typing import _F, _FloatingNDArray, _IntpNDArray, _to_real_dtype


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
    _half_maps: HalfSpectrumMaps | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _weight_cache: dict[tuple[str, np.dtype], _FloatingNDArray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _injective_fold: dict[bool, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

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
        indices = np.flatnonzero(arr_flat > threshold)
        values = arr_flat[indices]
        return cls(indices=indices, values=values, shape=arr.shape)

    def attach_periodized(
        self,
        decimation: npt.ArrayLike,
        *,
        with_flip: bool = False,
        half_spectrum: bool = False,
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
        half_spectrum : bool, optional
            If True, also attach the ``rfftn`` half-spectrum maps used by the
            real and monogenic transforms (otherwise built on first use).
            Default is False.

            .. versionadded:: 1.3

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
        self._injective_fold.clear()
        self._weight_cache.clear()
        self._half_maps = None
        if with_flip:
            self.flipped_indices = flip_fft_indices(self.indices, self.shape)
            self.flipped_folded_indices = compute_folded_indices(
                self.flipped_indices, self.shape, self.decimation
            )
        if half_spectrum:
            self.half_spectrum_maps()

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

    def _weights(
        self,
        dtype: npt.DTypeLike,
        key: str = "values",
        maps: HalfSpectrumMaps | None = None,
    ) -> npt.NDArray[np.floating]:
        """
        Window weights cast to the real dtype matching ``dtype``.

        ``key`` is ``"values"`` or a weight field of ``maps``; the cast is
        cached unless ``maps`` is a transient (unattached) map.
        """
        real_dtype = np.dtype(_to_real_dtype(dtype))
        cache_key = (key, real_dtype)
        weights = self._weight_cache.get(cache_key)
        if weights is None:
            source = self.values if maps is None else getattr(maps, key)
            weights = np.ascontiguousarray(source, dtype=real_dtype)
            if maps is None or maps is self._half_maps:
                self._weight_cache[cache_key] = weights
        return weights

    def _fold_is_injective(self, flip: bool, folded_indices: _IntpNDArray) -> bool:
        """Whether no two samples fold onto the same decimated bin (cached)."""
        injective = self._injective_fold.get(flip)
        if injective is None:
            injective = bool(np.unique(folded_indices).size == folded_indices.size)
            attached = self.flipped_folded_indices if flip else self.folded_indices
            if attached is not None:
                self._injective_fold[flip] = injective
        return injective

    def half_spectrum_maps(
        self, decimation: npt.ArrayLike | None = None
    ) -> HalfSpectrumMaps:
        """
        Index maps of this window onto the ``rfftn`` half spectrum.

        Cached once the periodized maps are attached.

        .. versionadded:: 1.3

        Parameters
        ----------
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Returns
        -------
        HalfSpectrumMaps
            Forward gather source and backward own / mirror contributions.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> window.half_spectrum_maps().source.shape
        (64,)
        """
        if self._half_maps is not None:
            return self._half_maps
        _, folded_indices = self.resolve_indices(decimation=decimation)
        maps = compute_half_spectrum_maps(
            self.indices, folded_indices, self.values, self.shape
        )
        if self.folded_indices is not None:
            self._half_maps = maps
        return maps

    def gather(
        self,
        spectrum: npt.NDArray[np.complexfloating],
        *,
        scale: float = 1.0,
        flip: bool = False,
        half: bool = False,
        decimation: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Weighted window samples ``scale * values * spectrum[indices]``.

        .. versionadded:: 1.3

        Parameters
        ----------
        spectrum : ndarray of complex
            Full-size frequency-domain image, or with ``half=True`` the
            :func:`~curvelets.numpy._periodized_fft.half_spectrum_source` of an
            ``rfftn`` half spectrum.
        scale : float, optional
            Factor folded into the weights. Default is 1.
        flip : bool, optional
            If True, use flipped indices (full spectrum only).
        half : bool, optional
            If True, read the samples from a half-spectrum source.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Returns
        -------
        ndarray of complex
            One sample per window entry, in window order.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> window.gather(np.ones((4, 4), dtype=np.complex128), scale=2.0)[:2]
        array([2.+0.j, 2.+0.j])
        """
        if half:
            if flip:
                msg = "flip is not supported with half-spectrum sources"
                raise ValueError(msg)
            source = self.half_spectrum_maps(decimation).source
        else:
            source, _ = self.resolve_indices(flip=flip, decimation=decimation)
        weights = self._weights(spectrum.dtype)
        if scale != 1.0:
            weights = weights * scale
        samples: npt.NDArray[np.complexfloating] = (
            spectrum.reshape(-1)[source] * weights
        )
        return samples

    def fold(
        self,
        samples: npt.NDArray[np.complexfloating],
        *,
        flip: bool = False,
        decimation: npt.ArrayLike | None = None,
        out: npt.NDArray[np.complexfloating] | None = None,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Fold window samples (from :meth:`gather`) into the decimated grid.

        .. versionadded:: 1.3

        Parameters
        ----------
        samples : ndarray of complex
            One value per window entry.
        flip : bool, optional
            If True, use the flipped folded indices.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.
        out : ndarray of complex, optional
            Reusable C-contiguous output buffer (cleared before use).

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
        >>> window.fold(np.ones(16, dtype=np.complex128)).real
        array([[4., 4.],
               [4., 4.]])
        """
        _, folded_indices = self.resolve_indices(flip=flip, decimation=decimation)
        out_shape = self._resolved_out_shape(decimation)
        if out is None:
            folded = np.zeros(out_shape, dtype=samples.dtype)
        else:
            folded = out
            folded.fill(0)
        flat = folded.reshape(-1)
        if self._fold_is_injective(flip, folded_indices):
            flat[folded_indices] = samples
        else:
            np.add.at(flat, folded_indices, samples)
        return folded

    def fold_product(
        self,
        image_frequency: npt.NDArray[np.complexfloating],
        *,
        flip: bool = False,
        extra_at_indices: npt.NDArray | None = None,
        decimation: npt.ArrayLike | None = None,
        out: npt.NDArray[np.complexfloating] | None = None,
        scale: float = 1.0,
        half: bool = False,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Fold ``scale * values * image_frequency[indices]`` into the decimated grid.

        Parameters
        ----------
        image_frequency : ndarray of complex
            Full-size frequency-domain image (or half-spectrum source with
            ``half=True``, see :meth:`gather`).
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : ndarray, optional
            Extra multiplicative weights already aligned with window indices
            (e.g. Riesz filter samples).
        decimation : array_like, optional
            Fallback when periodized maps are not attached.
        out : ndarray of complex, optional
            Reusable output buffer (cleared before use).
        scale : float, optional
            Factor folded into the weights. Default is 1.

            .. versionadded:: 1.3
        half : bool, optional
            If True, ``image_frequency`` is a half-spectrum source.

            .. versionadded:: 1.3

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
        samples = self.gather(
            image_frequency, scale=scale, flip=flip, half=half, decimation=decimation
        )
        if extra_at_indices is not None:
            samples *= extra_at_indices.astype(samples.dtype, copy=False)
        return self.fold(samples, flip=flip, decimation=decimation, out=out)

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
        contribution = small_fft.reshape(-1)[folded_indices] * (
            self._weights(target.dtype) * scale
        )
        if target.flags.c_contiguous:
            target.reshape(-1)[indices] += contribution
        else:
            target.flat[indices] += contribution

    def scatter_half(
        self,
        small_fft: npt.NDArray[np.complexfloating],
        own: npt.NDArray[np.complexfloating],
        mirror: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        decimation: npt.ArrayLike | None = None,
    ) -> None:
        """
        Half-spectrum version of :meth:`scatter_tiled` (real-valued outputs).

        Accumulates into two C-contiguous half-grid buffers; after all windows,
        :func:`~curvelets.numpy._periodized_fft.hermitian_from_half` combines
        them into the half spectrum whose ``irfftn`` equals the ``ifftn(...).real``
        of the full-spectrum scatter.

        .. versionadded:: 1.3

        Parameters
        ----------
        small_fft : ndarray of complex
            ``fftn`` of the coefficient array.
        own, mirror : ndarray of complex
            Half-spectrum accumulators (modified in place).
        scale : float
            Combined normalization / mode scale factor.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> own = np.zeros((4, 3), dtype=np.complex128)
        >>> mirror = np.zeros_like(own)
        >>> window.scatter_half(np.ones((2, 2), dtype=np.complex128), own, mirror, 1.0)
        >>> bool(np.any(own != 0) and np.any(mirror != 0))
        True
        """
        maps = self.half_spectrum_maps(decimation)
        small = small_fft.reshape(-1)
        dtype = own.dtype
        own.reshape(-1)[maps.own_indices] += small[maps.own_folded] * (
            self._weights(dtype, "own_values", maps) * scale
        )
        mirror.reshape(-1)[maps.mirror_indices] += small[maps.mirror_folded] * (
            self._weights(dtype, "mirror_values", maps) * scale
        )

    def analyze(
        self,
        image_frequency: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        flip: bool = False,
        extra_at_indices: npt.NDArray | None = None,
        decimation: npt.ArrayLike | None = None,
        half: bool = False,
        fft: FFTBackend | None = None,
    ) -> npt.NDArray[np.complexfloating]:
        """
        Forward periodized wedge: fold → small IFFT → scale.

        Parameters
        ----------
        image_frequency : ndarray of complex
            Full-size frequency-domain image (or half-spectrum source with
            ``half=True``, see :meth:`gather`).
        scale : float
            Combined normalization factor (applied to the window weights).
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : ndarray, optional
            Extra weights aligned with window indices (e.g. Riesz samples).
        decimation : array_like, optional
            Fallback when periodized maps are not attached.
        half : bool, optional
            If True, ``image_frequency`` is a half-spectrum source.

            .. versionadded:: 1.3
        fft : FFTBackend, optional
            FFT implementation; defaults to ``numpy.fft``.

            .. versionadded:: 1.3

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
            scale=scale,
            half=half,
        )
        ifftn = np.fft.ifftn if fft is None else fft.ifftn
        result: npt.NDArray[np.complexfloating] = ifftn(folded)
        return result

    def synthesize(
        self,
        coefficient: npt.NDArray[np.complexfloating],
        target: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        flip: bool = False,
        decimation: npt.ArrayLike | None = None,
        fft: FFTBackend | None = None,
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
        fft : FFTBackend, optional
            FFT implementation; defaults to ``numpy.fft``.

            .. versionadded:: 1.3

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
        fftn = np.fft.fftn if fft is None else fft.fftn
        self.scatter_tiled(
            fftn(coefficient),
            target,
            scale,
            flip=flip,
            decimation=decimation,
        )

    def synthesize_half(
        self,
        coefficient: npt.NDArray[np.complexfloating],
        own: npt.NDArray[np.complexfloating],
        mirror: npt.NDArray[np.complexfloating],
        scale: float,
        *,
        decimation: npt.ArrayLike | None = None,
        fft: FFTBackend | None = None,
    ) -> None:
        """
        Half-spectrum backward wedge: small FFT → :meth:`scatter_half`.

        .. versionadded:: 1.3

        Parameters
        ----------
        coefficient : ndarray of complex
            Decimated coefficient array.
        own, mirror : ndarray of complex
            Half-spectrum accumulators (modified in place).
        scale : float
            Combined normalization / mode scale factor.
        decimation : array_like, optional
            Fallback when periodized maps are not attached.
        fft : FFTBackend, optional
            FFT implementation; defaults to ``numpy.fft``.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import SparseWindow
        >>> window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> own = np.zeros((8, 5), dtype=np.complex128)
        >>> mirror = np.zeros_like(own)
        >>> window.synthesize_half(np.ones((4, 4), np.complex128), own, mirror, 1.0)
        >>> bool(np.any(own != 0))
        True
        """
        fftn = np.fft.fftn if fft is None else fft.fftn
        self.scatter_half(fftn(coefficient), own, mirror, scale, decimation=decimation)

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
