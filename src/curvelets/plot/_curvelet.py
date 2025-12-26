from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from curvelets.utils import deepflatten


def overlay_disk(
    c_struct: list[list[list[float]]],
    ax: Axes | None = None,
    linewidth: float = 5,
    linecolor: str = "r",
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    direction: str = "normal",
) -> Axes:
    r"""Overlay a curvelet structure onto its locations on a multiscale, multidirectional disk.

    Its intended usage is to display various scalars derived from curvelet
    wedges of a certain image with a disk display.


    Parameters
    ----------
    c_struct : list[list[list[float]]]
        A scale for every curvelet wedge.
    ax : :obj:`Axes <matplotlib.axes.Axes>`, optional
        Axis on which to overlay the disk. Uses :obj:`plt.gca <matplotlib.pyplot.gca>` if None.
    linewidth : float, optional
        Width of line separating scales as a percentage of a single scale height, by default 5.
        Set to zero to disable.
    linecolor : str, optional
        Color of line separating scales, by default "r".
    cmap : str, optional
        Colormap, by default ``"turbo"``.
    vmin, vmax : float, optional
        Data range that the colormap covers. By default, the colormap covers the
        complete value range of the supplied data.
    direction : str, optional
        "tangent" or, by default "normal"

    Returns
    -------
    :obj:`Axes <matplotlib.axes.Axes>`
        Axis used.
    """
    ax = plt.gca() if ax is None else ax
    ax.axis("off")

    if vmin is None:
        vmin = min(v for v in deepflatten(c_struct))
    if vmax is None:
        vmax = max(v for v in deepflatten(c_struct))
    cmapper = ScalarMappable(norm=Normalize(vmin, vmax, clip=True), cmap=cmap)

    nscales = len(c_struct)
    ndir = 2  # Only available for 2D!

    deg_360 = 2 * np.pi
    deg_135 = np.pi * 3 / 4
    deg_n45 = -np.pi / 4
    deg_90 = np.pi / 2

    linewidth *= 0.01 / (nscales - 1)
    wedge_height = 1 / (nscales - 1)
    magic_shift = 0  # 0 or -np.pi/8 or -np.pi/16? something else?
    color = cmapper.to_rgba(c_struct[0][0][0])
    ax.bar(x=0, height=wedge_height, width=deg_360, bottom=0, color=color)
    for iscale, s in enumerate(c_struct[1:], start=1):
        assert len(s) == ndir, ValueError(
            f"{len(s)=} != {ndir=} c_struct must be from 2D input"
        )
        for idir, d in enumerate(s):
            nwedges = len(d)
            angles_per_wedge = deg_90 / nwedges
            pm = (-1) ** idir  # CCW for idir == 0, CC otherwise
            for iwedge, w in enumerate(d):
                color = cmapper.to_rgba(w)
                for offset in [deg_135, deg_n45]:  # top-left, bottom-right
                    # Center the wedge at its midpoint
                    # Note: iwedge indexing should match the actual frequency space wedge ordering
                    wedge_x = (
                        offset + pm * angles_per_wedge * (iwedge + 0.5) + magic_shift
                    )
                    if direction == "tangent":
                        wedge_x += deg_90
                    wedge_width = angles_per_wedge
                    wedge_bottom = iscale * wedge_height
                    ax.bar(
                        x=wedge_x,
                        height=wedge_height,
                        width=wedge_width,
                        bottom=wedge_bottom,
                        color=color,
                    )
    if linewidth == 0:
        return ax
    # Plot after so they are on top
    for iscale, s in enumerate(c_struct):
        # Scale separators
        ax.bar(
            x=0,
            height=linewidth,
            width=deg_360,
            bottom=(iscale + 1 - linewidth / 2) / (nscales - 1),
            color=linecolor,
        )
        if iscale == 0:
            continue
        # Wedge separators
        for idir, d in enumerate(s):
            nwedges = len(d)
            angles_per_wedge = deg_90 / nwedges
            pm = (-1) ** idir
            for iwedge in range(nwedges):
                for offset in [deg_135, deg_n45]:  # top-left, bottom-right
                    # Center the wedge: use iwedge + 0.5 to get the center angle
                    # This ensures the wedge angle matches the FFT kmax interpretation
                    wedge_x = (
                        offset + pm * angles_per_wedge * (iwedge + 0.5) + magic_shift
                    )
                    if direction == "tangent":
                        wedge_x += deg_90
                    wedge_width = angles_per_wedge
                    wedge_bottom = iscale * wedge_height
                    ax.bar(
                        x=wedge_x - wedge_width / 2,
                        height=wedge_height,
                        width=linewidth,
                        bottom=wedge_bottom,
                        color=linecolor,
                    )

    return ax
