from __future__ import annotations

import numpy as np
import numpy.typing as npt


def riesz_filters(shape: tuple[int, ...]) -> list[npt.NDArray[np.complexfloating]]:
    """
    Create Riesz transform filters in frequency domain.

    The Riesz transform is an N-D generalization of the Hilbert transform,
    defined componentwise in the frequency domain as:
    :math:`R_k(f)(\\xi) = i (\\xi_k / |\\xi|) \\hat{f}(\\xi)`

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data. Determines the size of frequency grids.

    Returns
    -------
    list[npt.NDArray[np.complexfloating]]
        List of Riesz filters :math:`R_1, R_2, \\ldots, R_{\\text{ndim}}` where:
        - :math:`R_k(\\xi) = i \\xi_k / |\\xi|`
        - Each filter has the same shape as the input
        - DC component (zero frequency) is set to 0

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._riesz import riesz_filters
    >>> filters = riesz_filters((64, 64))
    >>> len(filters)  # Number of dimensions
    2
    >>> filters[0].shape  # Same shape as input
    (64, 64)
    >>> filters[0].dtype  # Complex dtype
    dtype('complex128')
    >>> # DC component is zero
    >>> filters[0][0, 0]
    0j
    """
    # Create frequency grids for each dimension
    # Using fftfreq to get FFT frequency coordinates (in cycles per sample)
    # Convert to radians by multiplying by 2*pi to match continuous Riesz transform definition
    # fftfreq gives frequencies in range [-0.5, 0.5), multiply by 2*pi to get [-pi, pi)
    grids = [2 * np.pi * np.fft.fftfreq(s) for s in shape]

    # Create meshgrids for all dimensions
    meshgrids = np.meshgrid(*grids, indexing="ij")

    # Compute |xi| = sqrt(sum of squares of all frequency components
    xi_norm_squared = sum(g**2 for g in meshgrids)
    xi_norm = np.sqrt(xi_norm_squared)

    # Avoid division by zero at DC component
    # Set to 1 where |xi| == 0, then we'll set those components to 0 later
    xi_norm[xi_norm == 0] = 1

    # Compute Riesz filters: R_k = i * xi_k / |xi|
    riesz_filters_list: list[npt.NDArray[np.complexfloating]] = [
        1j * g / xi_norm for g in meshgrids
    ]

    # Set DC component (zero frequency) to 0 for all filters
    # The zero frequency point is at index (0, 0, ...) in all dimensions
    dc_index = tuple(0 for _ in shape)
    for r_filter in riesz_filters_list:
        r_filter[dc_index] = 0

    return riesz_filters_list
