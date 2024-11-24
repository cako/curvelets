from __future__ import annotations

__all__ = [
    "create_colorbar",
    "create_inset_axes_grid",
    "despine",
    "overlay_arrows",
    "overlay_disk",
]
import logging

from .._internal import MATPLOTLIB_ENABLED
from ._curvelet import overlay_disk

logger = logging.getLogger()

if MATPLOTLIB_ENABLED:
    from ._matplotlib import (
        create_colorbar,
        create_inset_axes_grid,
        despine,
        overlay_arrows,
    )
else:
    logger.warning("matplotlib is not installed, not all functions will be available")
