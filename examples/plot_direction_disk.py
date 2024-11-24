r"""
Multiscale Direction Disks
==========================
This example shows how to use the UDCT curvelets transform to visualize
multiscale preferrential directions in an image. Inspired by
`Kymatio's Scattering disks <https://www.kymat.io/gallery_2d/plot_scattering_disk.html>`__.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from curvelets.numpy import UDCT, SimpleUDCT

# from curvelets.plot import create_inset_axes_grid, overlay_arrows, overlay_disk
# from curvelets.utils import apply_along_wedges, normal_vector_field

inputfile = "../testdata/sigmoid.npz"

# %%
# Input Data
# ##########
data = np.load(inputfile)
data = data["sigmoid"][:100, :64]
# data = data["sigmoid"][:128, :64].T
# data = data["sigmoid"][128:, :64]
nx, nz = data.shape
dx, dz = 0.005, 0.004
x, z = np.arange(nx) * dx, np.arange(nz) * dz
