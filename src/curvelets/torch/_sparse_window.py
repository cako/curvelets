"""Sparse window implementation for PyTorch UDCT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ._periodized_fft import compute_folded_indices, decimated_shape, flip_fft_indices


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
    folded_indices : torch.Tensor | None, optional
        Decimated-grid flat indices attached by ``attach_periodized``.
    flipped_indices : torch.Tensor | None, optional
        FFT-axis flipped flat indices attached when ``with_flip=True``.
    flipped_folded_indices : torch.Tensor | None, optional
        Folded indices corresponding to ``flipped_indices``.
    decimation : torch.Tensor | None, optional
        Per-axis decimation ratios attached with periodized maps.
    out_shape : tuple[int, ...] | None, optional
        Decimated coefficient shape ``s // d`` per axis.

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
    folded_indices: torch.Tensor | None = field(default=None, repr=False)
    flipped_indices: torch.Tensor | None = field(default=None, repr=False)
    flipped_folded_indices: torch.Tensor | None = field(default=None, repr=False)
    decimation: torch.Tensor | None = field(default=None, repr=False)
    out_shape: tuple[int, ...] | None = field(default=None, repr=False)

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
    ) -> SparseWindow:
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

    def attach_periodized(self, decimation: Any, *, with_flip: bool = False) -> None:
        """
        Attach decimation and precomputed folded (and optional flipped) index maps.

        Parameters
        ----------
        decimation : array_like or torch.Tensor
            Per-axis decimation ratios for this window.
        with_flip : bool, optional
            If True, also attach flipped index maps for complex mode.
            Default is False.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2], with_flip=True)
        >>> window.folded_indices is not None
        True
        >>> window.flipped_indices is not None
        True
        >>> window.out_shape
        (4, 4)
        """
        if isinstance(decimation, torch.Tensor):
            self.decimation = decimation.clone().detach().to(
                dtype=torch.long, device=self.device
            ).flatten()
        else:
            self.decimation = torch.tensor(
                [int(x) for x in decimation], dtype=torch.long, device=self.device
            )
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
        decimation: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(indices, folded_indices)``, computing on the fly if needed.

        Parameters
        ----------
        flip : bool, optional
            If True, use flipped indices (complex negative-frequency wedges).
        decimation : array_like or torch.Tensor, optional
            Used only when folded maps are not yet attached.

        Returns
        -------
        indices : torch.Tensor
            Flat indices into the full-size frequency grid.
        folded_indices : torch.Tensor
            Flat indices into the decimated grid.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((8, 8)), threshold=0.5)
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

    def _resolved_out_shape(self, decimation: Any | None = None) -> tuple[int, ...]:
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
        image_frequency: torch.Tensor,
        *,
        flip: bool = False,
        extra_at_indices: torch.Tensor | None = None,
        decimation: Any | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fold ``values * image_frequency[indices]`` into the decimated frequency grid.

        Parameters
        ----------
        image_frequency : torch.Tensor
            Full-size frequency-domain tensor.
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : torch.Tensor, optional
            Extra multiplicative weights already aligned with window indices
            (e.g. Riesz filter samples).
        decimation : array_like or torch.Tensor, optional
            Fallback when periodized maps are not attached.
        out : torch.Tensor, optional
            Reusable output buffer (cleared before use).

        Returns
        -------
        torch.Tensor
            Folded spectrum of shape ``out_shape``.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> image_freq = torch.ones((4, 4), dtype=torch.complex128)
        >>> folded = window.fold_product(image_freq)
        >>> folded.shape
        torch.Size([2, 2])
        """
        indices, folded_indices = self.resolve_indices(flip=flip, decimation=decimation)
        out_shape = self._resolved_out_shape(decimation)
        vals = image_frequency.view(-1)[indices] * self.values.to(
            image_frequency.dtype
        )
        if extra_at_indices is not None:
            vals = vals * extra_at_indices.to(image_frequency.dtype)
        if out is None:
            folded = torch.zeros(
                out_shape, dtype=image_frequency.dtype, device=image_frequency.device
            )
            folded = folded.view(-1).index_add(0, folded_indices, vals).view(out_shape)
        else:
            folded = out
            folded.zero_()
            try:
                folded.view(-1).index_add_(0, folded_indices, vals)
            except RuntimeError as e:
                if "vmap" in str(e) or "out-of-place" in str(e):
                    res = (
                        folded.view(-1)
                        .index_add(0, folded_indices, vals)
                        .view(out_shape)
                    )
                    try:
                        folded.copy_(res)
                    except Exception:
                        pass
                else:
                    raise
        return folded

    def scatter_tiled(
        self,
        small_fft: torch.Tensor,
        target: torch.Tensor,
        scale: float,
        *,
        flip: bool = False,
        decimation: Any | None = None,
    ) -> None:
        """
        Scatter ``scale * values * tile(small_fft)`` into ``target``.

        Parameters
        ----------
        small_fft : torch.Tensor
            ``fftn`` of the coefficient tensor.
        target : torch.Tensor
            Full-size frequency accumulator (modified in-place).
        scale : float
            Combined normalization / mode scale factor.
        flip : bool, optional
            If True, use flipped indices.
        decimation : array_like or torch.Tensor, optional
            Fallback when periodized maps are not attached.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((4, 4)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> target = torch.zeros((4, 4), dtype=torch.complex128)
        >>> small = torch.ones((2, 2), dtype=torch.complex128)
        >>> window.scatter_tiled(small, target, scale=1.0)
        >>> bool(torch.any(target != 0))
        True
        """
        indices, folded_indices = self.resolve_indices(flip=flip, decimation=decimation)
        vals = (
            small_fft.view(-1)[folded_indices]
            * self.values.to(target.dtype)
            * scale
        )
        try:
            target.view(-1).index_add_(0, indices, vals)
        except RuntimeError as e:
            if "vmap" in str(e) or "out-of-place" in str(e):
                res = target.view(-1).index_add(0, indices, vals).view_as(target)
                try:
                    target.copy_(res)
                except Exception:
                    pass
            else:
                raise

    def analyze(
        self,
        image_frequency: torch.Tensor,
        scale: float,
        *,
        flip: bool = False,
        extra_at_indices: torch.Tensor | None = None,
        decimation: Any | None = None,
    ) -> torch.Tensor:
        """
        Forward periodized wedge: fold → small IFFT → scale.

        Parameters
        ----------
        image_frequency : torch.Tensor
            Full-size frequency-domain tensor.
        scale : float
            Combined normalization factor applied after the small IFFT.
        flip : bool, optional
            If True, use flipped indices.
        extra_at_indices : torch.Tensor, optional
            Extra weights aligned with window indices (e.g. Riesz samples).
        decimation : array_like or torch.Tensor, optional
            Fallback when periodized maps are not attached.

        Returns
        -------
        torch.Tensor
            Decimated coefficient tensor.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> image_freq = torch.fft.fftn(torch.randn(8, 8))
        >>> coeff = window.analyze(image_freq, scale=1.0)
        >>> coeff.shape
        torch.Size([4, 4])
        """
        folded = self.fold_product(
            image_frequency,
            flip=flip,
            extra_at_indices=extra_at_indices,
            decimation=decimation,
        )
        res: torch.Tensor = scale * torch.fft.ifftn(folded)
        return res

    def synthesize(
        self,
        coefficient: torch.Tensor,
        target: torch.Tensor,
        scale: float,
        *,
        flip: bool = False,
        decimation: Any | None = None,
    ) -> None:
        """
        Backward periodized wedge: small FFT → tiled sparse scatter into ``target``.

        Parameters
        ----------
        coefficient : torch.Tensor
            Decimated coefficient tensor.
        target : torch.Tensor
            Full-size frequency accumulator (modified in-place).
        scale : float
            Combined normalization / mode scale factor.
        flip : bool, optional
            If True, use flipped indices.
        decimation : array_like or torch.Tensor, optional
            Fallback when periodized maps are not attached.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch._sparse_window import SparseWindow
        >>> window = SparseWindow.from_dense(torch.ones((8, 8)), threshold=0.5)
        >>> window.attach_periodized([2, 2])
        >>> target = torch.zeros((8, 8), dtype=torch.complex128)
        >>> coeff = torch.ones((4, 4), dtype=torch.complex128)
        >>> window.synthesize(coeff, target, scale=1.0)
        >>> bool(torch.any(target != 0))
        True
        """
        self.scatter_tiled(
            torch.fft.fftn(coefficient),
            target,
            scale,
            flip=flip,
            decimation=decimation,
        )

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
        idx_flat = self.indices.view(-1)
        out.view(-1)[idx_flat] = arr.view(-1)[idx_flat] * self.values.view(-1).to(
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
        idx_flat = self.indices.view(-1)
        target.view(-1)[idx_flat] += source.view(-1)[
            idx_flat
        ] * self.values.view(-1).to(target.dtype)

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
        idx_flat = self.indices.view(-1)
        out.view(-1)[idx_flat] = (
            source.view(-1)[idx_flat] * filter_arr.view(-1)[idx_flat]
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
        arr.view(-1)[self.indices.view(-1)] = self.values.view(-1).to(dtype)
        return arr

    def to(self, device: torch.device) -> SparseWindow:
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
            folded_indices=self.folded_indices.to(device)
            if self.folded_indices is not None
            else None,
            flipped_indices=self.flipped_indices.to(device)
            if self.flipped_indices is not None
            else None,
            flipped_folded_indices=self.flipped_folded_indices.to(device)
            if self.flipped_folded_indices is not None
            else None,
            decimation=self.decimation.to(device)
            if self.decimation is not None
            else None,
            out_shape=self.out_shape,
        )
