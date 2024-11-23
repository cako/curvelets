from __future__ import annotations

__all__ = [
    # _curvelet.py
    "curveshow",
    "overlay_disk",
    "overlay_disks",
    # _matplotlib.py
    "create_axes_grid",
    "create_colorbar",
    "create_inset_axes_grid",
    "despine",
    "overlay_arrows",
]
import logging

from .._internal import MATPLOTLIB_ENABLED
from ._curvelet import curveshow, overlay_disk, overlay_disks

logger = logging.getLogger()

if MATPLOTLIB_ENABLED:
    from ._matplotlib import (
        create_axes_grid,
        create_colorbar,
        create_inset_axes_grid,
        despine,
        overlay_arrows,
    )
else:
    logger.warning("matplotlib is not installed, not all functions will be available")
