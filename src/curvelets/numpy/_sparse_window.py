"""Sparse window implementation for NumPy UDCT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._typing import F, FloatingNDArray, IntpNDArray


@dataclass
class SparseWindow:
    """
    Sparse frequency-domain window for efficient curvelet operations.

    This class represents a sparse window as a collection of indices and values,
    providing efficient methods for forward and backward curvelet transforms.

    Parameters
    ----------
    indices : npt.NDArray[np.intp]
        Flat indices into array where window has non-zero values.
    values : npt.NDArray[np.floating]
        Window values at those indices.
    shape : tuple[int, ...]
        Original array shape (for to_dense conversion).

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._sparse_window import SparseWindow
    >>> arr = np.array([[0.1, 0.9], [0.2, 0.8]])
    >>> window = SparseWindow.from_dense(arr, threshold=0.5)
    >>> window.size
    2
    >>> dense = window.to_dense()
    >>> dense.shape
    (2, 2)
    """

    indices: IntpNDArray  # Flat indices into array
    values: FloatingNDArray  # Window values at those indices
    shape: tuple[int, ...]  # Original array shape (for to_dense)

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
        >>> from curvelets.numpy._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.1, 0.9, 0.2]), threshold=0.5)
        >>> window.size
        1
        """
        return len(self.indices)

    @classmethod
    def from_dense(
        cls,
        arr: npt.NDArray[F],
        threshold: float,
    ) -> "SparseWindow":
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
        >>> from curvelets.numpy._sparse_window import SparseWindow
        >>> arr = np.array([[0.1, 0.9], [0.2, 0.8]])
        >>> window = SparseWindow.from_dense(arr, threshold=0.5)
        >>> window.size
        2
        """
        arr_flat = arr.ravel()
        indices = np.argwhere(arr_flat > threshold).ravel()
        values = arr_flat[indices]
        return cls(indices=indices, values=values, shape=arr.shape)

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
        >>> from curvelets.numpy._sparse_window import SparseWindow
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
        >>> from curvelets.numpy._sparse_window import SparseWindow
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
        >>> from curvelets.numpy._sparse_window import SparseWindow
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
        out.flat[idx_flat] = (
            source.flat[idx_flat] * filter_arr.flat[idx_flat]
        ).astype(out.dtype)
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
        >>> from curvelets.numpy._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(np.array([0.1, 0.9, 0.2]), threshold=0.5)
        >>> dense = window.to_dense()
        >>> dense[1]  # Only index 1 is above threshold
        0.9
        """
        dtype = dtype or self.values.dtype
        arr = np.zeros(self.shape, dtype=dtype)
        arr.flat[self.indices] = self.values.astype(dtype)
        return arr
