"""Riesz transform filters for PyTorch UDCT implementation."""

from __future__ import annotations

import torch


def riesz_filters(
    shape: tuple[int, ...], device: torch.device | None = None
) -> list[torch.Tensor]:
    r"""
    Create Riesz transform filters in frequency domain.

    The Riesz transform is an N-D generalization of the Hilbert transform,
    defined componentwise in the frequency domain as:
    :math:`R_k(f)(\xi) = i (\xi_k / |\xi|) \hat{f}(\xi)`

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data. Determines the size of frequency grids.
    device : torch.device, optional
        Device on which to create the filters. Default is None (CPU).

    Returns
    -------
    list[torch.Tensor]
        List of Riesz filters :math:`R_1, R_2, \ldots, R_{\text{ndim}}` where:
        - :math:`R_k(\xi) = i \xi_k / |\xi|`
        - Each filter has the same shape as the input
        - DC component (zero frequency) is set to 0

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch._riesz import riesz_filters
    >>> filters = riesz_filters((64, 64))
    >>> len(filters)  # Number of dimensions
    2
    >>> filters[0].shape  # Same shape as input
    torch.Size([64, 64])
    >>> filters[0].dtype  # Complex dtype
    torch.complex128
    >>> # DC component is zero
    >>> filters[0][0, 0]
    tensor(0.+0.j)
    """
    # Create frequency grids for each dimension
    # Using fftfreq to get FFT frequency coordinates (in cycles per sample)
    # Convert to radians by multiplying by 2*pi
    grids = [2 * torch.pi * torch.fft.fftfreq(s) for s in shape]  # pylint: disable=not-callable

    # Move grids to device if specified
    if device is not None:
        grids = [g.to(device) for g in grids]

    # Create meshgrids for all dimensions
    meshgrids = torch.meshgrid(*grids, indexing="ij")

    # Compute |xi| = sqrt(sum of squares of all frequency components)
    xi_norm_squared = sum(g**2 for g in meshgrids)
    xi_norm = torch.sqrt(xi_norm_squared)

    # Avoid division by zero at DC component
    # Set to 1 where |xi| == 0, then we'll set those components to 0 later
    xi_norm = torch.where(xi_norm == 0, torch.ones_like(xi_norm), xi_norm)

    # Compute Riesz filters: R_k = i * xi_k / |xi|
    riesz_filters_list: list[torch.Tensor] = [1j * g / xi_norm for g in meshgrids]

    # Set DC component (zero frequency) to 0 for all filters
    # The zero frequency point is at index (0, 0, ...) in all dimensions
    dc_index = tuple(0 for _ in shape)
    for r_filter in riesz_filters_list:
        r_filter[dc_index] = 0

    return riesz_filters_list
