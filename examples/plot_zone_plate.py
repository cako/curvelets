"""
Zone Plate
==========
"""
from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from curvelets.numpy.udct import UDCT
from curvelets.plot import create_colorbar, despine


def make_r(shape: tuple[int, int], exponent: float = 1, origin: tuple[int, int] = None):
    origin = (np.asarray(shape).astype(float) - 1) / 2
    xramp, yramp = np.meshgrid(
        np.arange(shape[0], dtype=float) - origin[0],
        np.arange(shape[1], dtype=float) - origin[1],
    )
    return (xramp**2 + yramp**2) ** (exponent / 2)


def make_zone_plate(shape: tuple[int, int], amplitude=1, phase=0):
    mxsz = max(*shape)

    return amplitude * np.cos((np.pi / mxsz) * make_r(shape, 2) + phase)


# %%
# Setup
# #####

shape = (256, 256)
zone_plate = make_zone_plate(shape)
C = UDCT(size=shape)

# %%
# Uniform Discrete Curvelet Transform Round Trip
# ##############################################

coeffs = C.forward(zone_plate)
zone_plate_inv = C.backward(coeffs)

# %%
vmax = np.abs(zone_plate).max()
opts = {"aspect": "equal", "cmap": "gray", "vmin": -vmax, "vmax": vmax}

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(zone_plate.T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
fmt = ticker.FuncFormatter(lambda x, pos: f"{x:.0e}")
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title="Input")

# %%
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(zone_plate_inv.T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title="UDCT Round Trip")

# %%
opts["vmax"] = np.abs(zone_plate - zone_plate_inv).max()
opts["vmin"] = -opts["vmax"]
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow((zone_plate - zone_plate_inv).T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title=f"Error (max = {opts['vmax']:.2g})")

print(f"Max Error: {opts['vmax']:.2g}")  # noqa: T201
