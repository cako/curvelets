"""Utility functions for examples.

This module contains utility functions used by multiple example scripts.
These functions are only used for examples and are not part of the main
curvelets package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["make_r", "make_zone_plate"]


def make_r(
    shape: tuple[int, ...], exponent: float = 1, origin: tuple[int, ...] | None = None
) -> NDArray[np.floating]:
    """Compute radial distance array from origin.

    Creates an array where each element contains the radial distance from
    a specified origin point, raised to the given exponent.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the output array.
    exponent : float, optional
        Exponent to apply to the radial distance. Default is 1.
    origin : tuple[int, ...] | None, optional
        Origin point for distance calculation. If None, uses the center of
        the array. Default is None.

    Returns
    -------
    NDArray[np.floating]
        Array of radial distances from origin, raised to the exponent.

    Examples
    --------
    >>> import numpy as np
    >>> from examples._utils import make_r
    >>> r = make_r((5, 5), exponent=1)
    >>> r.shape
    (5, 5)
    >>> r[2, 2]  # Center point should be 0
    0.0
    >>> r = make_r((3, 3), exponent=2, origin=(0, 0))
    >>> r[0, 0]  # Origin point should be 0
    0.0
    """
    orig = (
        tuple((np.asarray(shape).astype(float) - 1) / 2) if origin is None else origin
    )

    ramps = np.meshgrid(
        *[np.arange(s, dtype=float) - o for s, o in zip(shape, orig)], indexing="ij"
    )
    return sum(x**2 for x in ramps) ** (exponent / 2)


def make_zone_plate(
    shape: tuple[int, ...], amplitude: float = 1.0, phase: float = 0.0
) -> NDArray[np.floating]:
    """Generate a zone plate test pattern.

    Creates a zone plate pattern, which is a circular pattern of concentric
    rings with alternating intensity. This is commonly used as a test image
    for frequency domain analysis.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the output array.
    amplitude : float, optional
        Amplitude of the cosine pattern. Default is 1.0.
    phase : float, optional
        Phase offset for the cosine pattern. Default is 0.0.

    Returns
    -------
    NDArray[np.floating]
        Zone plate pattern array with values in the range [-amplitude, amplitude].

    Examples
    --------
    >>> import numpy as np
    >>> from examples._utils import make_zone_plate
    >>> zone_plate = make_zone_plate((64, 64))
    >>> zone_plate.shape
    (64, 64)
    >>> np.min(zone_plate) >= -1.0
    True
    >>> np.max(zone_plate) <= 1.0
    True
    >>> zone_plate = make_zone_plate((128, 128), amplitude=2.0, phase=np.pi/2)
    >>> zone_plate.shape
    (128, 128)
    """
    mxsz = max(*shape)

    return amplitude * np.cos((np.pi / mxsz) * make_r(shape, 2) + phase)

