"""Integration tests for plot module functions.

These tests simply run the code to ensure no exceptions are raised.
No assertions or verifications are performed.
"""

from __future__ import annotations

import warnings

# Suppress pyparsing deprecation warnings from matplotlib imports
# Must be set before importing matplotlib to catch warnings during import
# Use a catch-all filter for pyparsing deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Re-enable deprecation warnings for our code
warnings.filterwarnings("error", category=DeprecationWarning, module="curvelets")

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")  # Non-interactive backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from curvelets.numpy import UDCT  # noqa: E402
from curvelets.plot import (  # noqa: E402
    create_colorbar,
    create_inset_axes_grid,
    despine,
    overlay_arrows,
    overlay_disk,
)
from curvelets.utils import normal_vector_field  # noqa: E402


def test_overlay_disk(rng):
    """Test overlay_disk with real UDCT coefficients structure."""
    # Create a 2D UDCT transform
    transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)

    # Generate test data and get coefficients
    data = rng.normal(size=(64, 64))
    coeffs = transform.forward(data)

    # Convert coefficients to scalar values for overlay_disk
    # overlay_disk expects list[list[list[float]]]
    c_struct = []
    for scale in coeffs:
        scale_list = []
        for direction in scale:
            direction_list = []
            for wedge in direction:
                # Use magnitude of complex coefficients
                direction_list.append(float(np.abs(wedge).max()))
            scale_list.append(direction_list)
        c_struct.append(scale_list)

    # Add lowpass scale (first scale is lowpass, has different structure)
    # For 2D, first scale should be [[[value]]]
    lowpass_value = float(np.abs(coeffs[0][0][0]).max())
    c_struct[0] = [[lowpass_value]]

    # Run overlay_disk with default parameters
    fig, ax = plt.subplots()
    overlay_disk(c_struct, ax=ax)
    plt.close(fig)


def test_create_colorbar(rng):
    """Test create_colorbar with imshow output."""
    fig, ax = plt.subplots()
    im = ax.imshow(rng.normal(size=(10, 10)), vmin=-1, vmax=1, cmap="gray")
    create_colorbar(im, ax=ax)
    plt.close(fig)


def test_create_inset_axes_grid(rng):
    """Test create_inset_axes_grid with simple grid."""
    fig, ax = plt.subplots()
    ax.imshow(rng.normal(size=(10, 10)), vmin=-1, vmax=1, cmap="gray")
    create_inset_axes_grid(ax, rows=2, cols=2)
    plt.close(fig)


def test_despine():
    """Test despine function."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    despine(ax)
    plt.close(fig)


def test_overlay_arrows(rng):
    """Test overlay_arrows with normal vector field data."""
    # Create test data
    data = rng.normal(size=(32, 32))

    # Generate normal vector field
    rows, cols = 3, 4
    vectors = normal_vector_field(data, rows=rows, cols=cols)

    # Create plot and overlay arrows
    fig, ax = plt.subplots()
    ax.imshow(data.T, vmin=-1, vmax=1, cmap="gray", extent=(0, 1, 1, 0))
    overlay_arrows(vectors, ax)
    plt.close(fig)
