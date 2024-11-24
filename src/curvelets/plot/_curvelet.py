from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from curvelets.utils import deepflatten


def overlay_disk(
    c_struct: list[list[list[float]]],
    ax: Axes,
    linewidth: float = 5,
    linecolor: str = "r",
    cmap="turbo",
    vmin=None,
    vmax=None,
    direction="normal",
):
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
    for iscale in range(nscales):
        ndir = len(c_struct[iscale])
        if iscale > 0:
            assert ndir == 2
        for idir in range(ndir):
            nwedges = len(c_struct[iscale][idir])
            angles_per_wedge = deg_90 / nwedges
            pm = (-1) ** (idir + 1)  # CC for idir == 0, CCW otherwise
            for iwedge in range(nwedges):
                color = cmapper.to_rgba(c_struct[iscale][idir][iwedge])
                if iscale == 0:
                    ax.bar(
                        x=0, height=wedge_height, width=deg_360, bottom=0, color=color
                    )
                    continue
                for offset in [deg_135, deg_n45]:  # top-left, bottom-right
                    wedge_x = (
                        offset + pm * angles_per_wedge * (0.5 + iwedge) + magic_shift
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

    # Plot after so they are on top
    if linewidth > 0:
        for iscale in range(nscales):
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
            for idir in range(len(c_struct[iscale])):
                nwedges = len(c_struct[iscale][idir])
                angles_per_wedge = deg_90 / nwedges
                pm = (-1) ** (idir + 1)
                for iwedge in range(nwedges):
                    for offset in [deg_135, deg_n45]:  # top-left, bottom-right
                        wedge_x = (
                            offset
                            + pm * angles_per_wedge * (0.5 + iwedge)
                            + magic_shift
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
    ax.axis("off")
    return ax
