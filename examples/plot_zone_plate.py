"""
Zone Plate
==========
"""
from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from curvelets.numpy.udct import UDCT
from curvelets.plot import create_colorbar, despine


def make_r(
    shape: tuple[int, ...], exponent: float = 1, origin: tuple[int, ...] | None = None
):
    orig = (
        tuple((np.asarray(shape).astype(float) - 1) / 2) if origin is None else origin
    )

    ramps = np.meshgrid(*[np.arange(s, dtype=float) - o for s, o in zip(shape, orig)])
    return sum(x**2 for x in ramps) ** (exponent / 2)


def make_zone_plate(shape: tuple[int, ...], amplitude: float = 1.0, phase: float = 0.0):
    mxsz = max(*shape)

    return amplitude * np.cos((np.pi / mxsz) * make_r(shape, 2) + phase)


# %%
# Setup
# #####

shape = (256, 256)
zone_plate = make_zone_plate(shape)
cfg = np.array([[3, 3], [6, 6], [12, 6]])
C = UDCT(size=shape, cfg=cfg)

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
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
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

# %%
# Curvelet Coefficients: Amplitude and Phase
# ##########################################
z = coeffs[1][1]
opts["vmax"] = np.abs(z).max()
opts["vmin"] = 0
opts["cmap"] = "gray"
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
im = axs[0].imshow(np.abs(z).T, **opts)
_, cb = create_colorbar(im=im, ax=axs[0])
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
cb.ax.yaxis.set_major_formatter(fmt)
opts["vmax"] = 180
opts["vmin"] = -opts["vmax"]
opts["cmap"] = "hsv"
im = axs[1].imshow(np.angle(z, deg=True).T, **opts)
_, cb = create_colorbar(im=im, ax=axs[1])
cb.ax.yaxis.set_major_locator(ticker.MultipleLocator(45))
cb.ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}°")
axs[0].set(title="Amplitude")
axs[1].set(title="Phase")
fig.suptitle("Scale 1")
fig.tight_layout()

for i in range(2, max(coeffs.keys())):
    for j in coeffs[i]:
        for a in coeffs[i][j]:
            z = coeffs[i][j][a]
            opts["vmax"] = np.abs(z).max()
            opts["vmin"] = 0
            opts["cmap"] = "gray"
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            im = axs[0].imshow(np.abs(z).T, **opts)
            _, cb = create_colorbar(im=im, ax=axs[0])
            fmt = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
            cb.ax.yaxis.set_major_formatter(fmt)
            opts["vmax"] = 180
            opts["vmin"] = -opts["vmax"]
            opts["cmap"] = "hsv"
            im = axs[1].imshow(np.angle(z, deg=True).T, **opts)
            _, cb = create_colorbar(im=im, ax=axs[1])
            cb.ax.yaxis.set_major_locator(ticker.MultipleLocator(45))
            cb.ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}°")
            axs[0].set(title="Amplitude")
            axs[1].set(title="Phase")
            fig.suptitle(f"Scale {i} | Direction {j} | Angle {a}")
            fig.tight_layout()


# %%
# Curvelet Coefficients: Real and Imaginary
# #########################################
z = coeffs[1][1]
opts["vmax"] = np.abs(z).max()
opts["vmin"] = -opts["vmax"]
opts["cmap"] = "gray"
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for ax, img in zip(axs.ravel(), [z.real, z.imag]):
    im = ax.imshow(img.T, **opts)
    _, cb = create_colorbar(im=im, ax=ax)
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
    cb.ax.yaxis.set_major_formatter(fmt)
axs[0].set(title="Real")
axs[1].set(title="Imaginary")
fig.suptitle("Scale 1")
fig.tight_layout()

for i in range(2, max(coeffs.keys())):
    for j in coeffs[i]:
        for a in coeffs[i][j]:
            z = coeffs[i][j][a]
            opts["vmax"] = np.abs(z).max()
            opts["vmin"] = -opts["vmax"]
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            for ax, img in zip(axs.ravel(), [z.real, z.imag]):
                im = ax.imshow(img.T, **opts)
                _, cb = create_colorbar(im=im, ax=ax)
                fmt = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
                cb.ax.yaxis.set_major_formatter(fmt)
            axs[0].set(title="Real")
            axs[1].set(title="Imaginary")
            fig.suptitle(f"Scale {i} | Direction {j} | Angle {a}")
            fig.tight_layout()
