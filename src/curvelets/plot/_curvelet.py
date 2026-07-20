from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.projections.polar import PolarAxes

from curvelets.utils import deepflatten


def overlay_disk(
    c_struct: list[list[list[float]]],
    ax: Axes | None = None,
    linewidth: float = 5,
    linecolor: str = "r",
    cmap: str | Colormap = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    direction: str = "normal",
    origin: str = "upper",
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
    cmap : str or :obj:`Colormap <matplotlib.colors.Colormap>`, optional
        Colormap or name of colormap, by default ``"plasma"``.
    vmin, vmax : float, optional
        Data range that the colormap covers. By default, the colormap covers the
        complete value range of the supplied data.
    direction : str, optional
        "tangent" or, by default "normal"
    origin : str, optional
        "upper" or "lower", by default "upper". Matches :obj:`plt.imshow <matplotlib.pyplot.imshow>`
        default where the vertical axis increases downwards.

    Returns
    -------
    :obj:`Axes <matplotlib.axes.Axes>`
        Axis used.

    Notes
    -----
    Wedge angles are frequency angles of the underlying *array* (cycles per
    sample along each axis) — the UDCT has no notion of physical sample
    spacing. When overlaying the disk on an image, display the image with
    square pixels (index extents with the default ``imshow`` aspect, or
    physical extents with ``aspect=dy/dx``) so that on-screen directions
    match the disk.
    """
    ax = plt.gca() if ax is None else ax
    ax.axis("off")
    if origin not in ("upper", "lower"):
        msg = f"origin must be 'lower' or 'upper', got {origin!r}"
        raise ValueError(msg)

    if isinstance(ax, PolarAxes):
        if origin == "upper":
            ax.set_theta_direction(-1)
        else:
            ax.set_theta_direction(1)

    if vmin is None:
        vmin = min(v for v in deepflatten(c_struct))
    if vmax is None:
        vmax = max(v for v in deepflatten(c_struct))
    cmapper = ScalarMappable(norm=Normalize(vmin, vmax, clip=True), cmap=cmap)

    nscales = len(c_struct)
    ndir = 2  # Only available for 2D!

    deg_360 = 2 * np.pi
    deg_180 = np.pi
    deg_45 = np.pi / 4
    deg_90 = np.pi / 2

    linewidth *= 0.01 / (nscales - 1)
    wedge_height = 1 / (nscales - 1)
    magic_shift = 0  # 0 or -np.pi/8 or -np.pi/16? something else?
    # matplotlib's to_rgba accepts scalars despite type stub saying ndarray
    color = cmapper.to_rgba(c_struct[0][0][0])  # ty: ignore[invalid-argument-type]
    ax.bar(x=0, height=wedge_height, width=deg_360, bottom=0, color=color)
    for iscale, s in enumerate(c_struct[1:], start=1):
        assert len(s) == ndir, ValueError(
            f"{len(s)=} != {ndir=} c_struct must be from 2D input"
        )
        for idir, d in enumerate(s):
            nwedges = len(d)
            angles_per_wedge = deg_90 / nwedges
            # Both directional cones are anchored at their shared 45° boundary:
            # dir-0 wedge angles decrease from 45° towards -45°, dir-1 wedge
            # angles increase from 45° towards 135°, matching the UDCT's
            # frequency-space wedge ordering.
            pm = -1 if idir == 0 else 1
            base_offset = deg_45
            for iwedge, w in enumerate(d):
                # matplotlib's to_rgba accepts scalars despite type stub saying ndarray
                color = cmapper.to_rgba(w)  # ty: ignore[invalid-argument-type]
                for offset in [base_offset, base_offset + deg_180]:
                    # Center the wedge at its midpoint
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
            # Same parity convention as the wedge loop above
            pm = -1 if idir == 0 else 1
            base_offset = deg_45
            for iwedge in range(nwedges):
                for offset in [base_offset, base_offset + deg_180]:
                    # Center the wedge: use iwedge + 0.5 to get the center angle
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
