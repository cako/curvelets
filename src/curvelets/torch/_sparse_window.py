"""Sparse window implementation for PyTorch UDCT."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseWindow:
    """
    Sparse frequency-domain window for efficient curvelet operations (PyTorch).

    This class represents a sparse window as a collection of indices and values,
    providing efficient methods for forward and backward curvelet transforms.

    Parameters
    ----------
    indices : torch.Tensor
        Flat indices into array where window has non-zero values.
    values : torch.Tensor
        Window values at those indices.
    shape : tuple[int, ...]
        Original array shape (for to_dense conversion).

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch._sparse_window import SparseWindow
    >>> arr = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    >>> window = SparseWindow.from_dense(arr, threshold=0.5)
    >>> window.size
    2
    >>> dense = window.to_dense()
    >>> dense.shape
    torch.Size([2, 2])
    """

    indices: torch.Tensor  # Flat indices into array
    values: torch.Tensor  # Window values at those indices
    shape: tuple[int, ...]  # Original array shape (for to_dense)

    @property
    def device(self) -> torch.device:
        """
        Device of the window tensors.

        Returns
        -------
        torch.device
            Device where the window tensors are located.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> arr = torch.tensor([0.1, 0.9])
        >>> window = SparseWindow.from_dense(arr, threshold=0.5)
        >>> window.device
        device(type='cpu')
        """
        return self.values.device

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
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.1, 0.9, 0.2]), threshold=0.5)
        >>> window.size
        1
        """
        return len(self.indices)

    @classmethod
    def from_dense(
        cls,
        arr: torch.Tensor,
        threshold: float,
    ) -> "SparseWindow":
        """
        Create SparseWindow from dense tensor using threshold.

        Parameters
        ----------
        arr : torch.Tensor
            Input dense tensor.
        threshold : float
            Threshold for sparse storage (values above threshold are kept).

        Returns
        -------
        SparseWindow
            Sparse window containing only values above threshold.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> arr = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        >>> window = SparseWindow.from_dense(arr, threshold=0.5)
        >>> window.size
        2
        """
        arr_flat = arr.flatten()
        mask = arr_flat > threshold
        indices = torch.where(mask)[0]
        values = arr_flat[indices]
        return cls(indices=indices, values=values, shape=arr.shape)

    def multiply_extract(
        self,
        arr: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract values at indices, multiply by window, return dense result.

        This method replaces the pattern:
        ``freq_band.flatten()[idx] = image_frequency.flatten()[idx] * val``

        Parameters
        ----------
        arr : torch.Tensor
            Input tensor to extract values from.
        out : torch.Tensor | None, optional
            Output tensor to write results into. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            Dense tensor with extracted and multiplied values.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.0, 0.5, 1.0]), threshold=0.3)
        >>> arr = torch.tensor([1.0, 2.0, 3.0])
        >>> result = window.multiply_extract(arr)
        >>> result[1]  # Only index 1 and 2 are in window
        tensor(1.)
        """
        if out is None:
            out = torch.zeros(self.shape, dtype=arr.dtype, device=arr.device)
        else:
            out.zero_()
        idx_flat = self.indices.flatten()
        out.flatten()[idx_flat] = arr.flatten()[idx_flat] * self.values.flatten().to(
            out.dtype
        )
        return out

    def scatter_add(self, target: torch.Tensor, source: torch.Tensor) -> None:
        """
        Add source*window to target at window indices (in-place).

        This method replaces the pattern:
        ``target.flatten()[idx] += source.flatten()[idx] * val``

        Parameters
        ----------
        target : torch.Tensor
            Target tensor to accumulate into (modified in-place).
        source : torch.Tensor
            Source tensor to extract values from.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.0, 0.5, 1.0]), threshold=0.3)
        >>> target = torch.zeros(3)
        >>> source = torch.tensor([1.0, 2.0, 3.0])
        >>> window.scatter_add(target, source)
        >>> target[1]  # Only indices 1 and 2 are accumulated
        tensor(1.)
        """
        idx_flat = self.indices.flatten()
        target.flatten()[idx_flat] += (
            source.flatten()[idx_flat] * self.values.flatten().to(target.dtype)
        )

    def multiply_at_indices(
        self,
        source: torch.Tensor,
        filter_arr: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Multiply source values at window indices by filter values at same indices.

        This method replaces the pattern:
        ``out.flatten()[idx] = source.flatten()[idx] * filter_arr.flatten()[idx]``

        Parameters
        ----------
        source : torch.Tensor
            Source tensor (typically already windowed).
        filter_arr : torch.Tensor
            Filter tensor to multiply with (e.g., Riesz filter).
        out : torch.Tensor | None, optional
            Output tensor to write results into. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            Dense tensor with multiplied values at window indices.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.0, 0.5, 1.0]), threshold=0.3)
        >>> source = torch.tensor([1.0, 2.0, 3.0])
        >>> filter_arr = torch.tensor([0.5, 1.5, 2.5])
        >>> result = window.multiply_at_indices(source, filter_arr)
        >>> result[1]  # Only indices 1 and 2 are in window
        tensor(3.)
        """
        if out is None:
            out = torch.zeros(self.shape, dtype=source.dtype, device=source.device)
        else:
            out.zero_()
        idx_flat = self.indices.flatten()
        out.flatten()[idx_flat] = (
            source.flatten()[idx_flat] * filter_arr.flatten()[idx_flat]
        ).to(out.dtype)
        return out

    def to_dense(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Convert to dense tensor.

        This method replaces the pattern:
        ``arr.flatten()[idx] = val``

        Parameters
        ----------
        dtype : torch.dtype | None, optional
            Dtype for output tensor. If None, uses self.values.dtype.

        Returns
        -------
        torch.Tensor
            Dense tensor representation of the sparse window.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.1, 0.9, 0.2]), threshold=0.5)
        >>> dense = window.to_dense()
        >>> dense[1]  # Only index 1 is above threshold
        tensor(0.9000)
        """
        dtype = dtype or self.values.dtype
        arr = torch.zeros(self.shape, dtype=dtype, device=self.device)
        arr.flatten()[self.indices.flatten()] = self.values.flatten().to(dtype)
        return arr

    def to(self, device: torch.device) -> "SparseWindow":
        """
        Move window to specified device.

        Parameters
        ----------
        device : torch.device
            Target device to move window to.

        Returns
        -------
        SparseWindow
            New SparseWindow instance on the specified device.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.tensor([0.1, 0.9]), threshold=0.5)
        >>> window_gpu = window.to(torch.device("cuda"))
        >>> window_gpu.device
        device(type='cuda', index=0)
        """
        return SparseWindow(
            self.indices.to(device),
            self.values.to(device),
            self.shape,
        )
