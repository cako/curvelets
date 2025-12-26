"""
Meyer Wavelet Transform
========================
This example demonstrates the Meyer wavelet transform using a zone plate
test image. The Meyer wavelet decomposes a 2D signal into 4 subbands:
1 lowpass subband and 3 highpass subbands (horizontal, vertical, and
diagonal). The transform is perfectly invertible, allowing exact
reconstruction of the original signal.
"""

from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from curvelets.numpy import MeyerWavelet
from curvelets.plot import create_colorbar, despine


def make_r(
    shape: tuple[int, ...], exponent: float = 1, origin: tuple[int, ...] | None = None
):
    """Compute radial distance from origin."""
    orig = (
        tuple((np.asarray(shape).astype(float) - 1) / 2) if origin is None else origin
    )

    ramps = np.meshgrid(
        *[np.arange(s, dtype=float) - o for s, o in zip(shape, orig)], indexing="ij"
    )
    return sum(x**2 for x in ramps) ** (exponent / 2)


def make_zone_plate(shape: tuple[int, ...], amplitude: float = 1.0, phase: float = 0.0):
    """Generate a zone plate test pattern."""
    mxsz = max(*shape)

    return amplitude * np.cos((np.pi / mxsz) * make_r(shape, 2) + phase)


# %%
# Setup
# #####

shape = (256, 256)
zone_plate = make_zone_plate(shape)
wavelet = MeyerWavelet(shape=shape)

# %%
# Meyer Wavelet Forward Transform
# ###############################

lowpass = wavelet.forward(zone_plate)
highpass_bands = wavelet._highpass_bands

print(f"Input shape: {zone_plate.shape}")  # noqa: T201
print(f"Lowpass shape: {lowpass.shape}")  # noqa: T201
print(f"Number of highpass bands: {len(highpass_bands)}")  # noqa: T201
for i, band in enumerate(highpass_bands):
    print(f"Highpass band {i} shape: {band.shape}")  # noqa: T201

# %%
# Input Image
# ###########

vmax = np.abs(zone_plate).max()
opts = {"aspect": "equal", "cmap": "gray", "vmin": -vmax, "vmax": vmax}

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(zone_plate.T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title="Input Zone Plate")

# %%
# Lowpass Subband
# ###############

lowpass_vmax = np.abs(lowpass).max()
lowpass_opts = {"aspect": "equal", "cmap": "gray", "vmin": -lowpass_vmax, "vmax": lowpass_vmax}

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(lowpass.T, **lowpass_opts)
_, cb = create_colorbar(im=im, ax=ax)
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title="Lowpass Subband")

# %%
# Highpass Subbands
# #################

# For 2D, we have 3 highpass bands:
# - Band 0: Low-Low (after first dimension) -> High-Low (after second dimension)
# - Band 1: Low-High (after first dimension) -> Low-High (after second dimension)
# - Band 2: High-Low (after first dimension) -> High-High (after second dimension)
# Note: The exact interpretation depends on the order of dimension processing

band_names = ["Highpass Band 0", "Highpass Band 1", "Highpass Band 2"]
highpass_vmax = max(np.abs(band).max() for band in highpass_bands)
highpass_opts = {
    "aspect": "equal",
    "cmap": "RdBu_r",
    "vmin": -highpass_vmax,
    "vmax": highpass_vmax,
}

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, band, name in zip(axs, highpass_bands, band_names):
    im = ax.imshow(band.T, **highpass_opts)
    _, cb = create_colorbar(im=im, ax=ax)
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    cb.ax.yaxis.set_major_formatter(fmt)
    despine(ax)
    ax.set(title=name)
fig.tight_layout()

# %%
# Energy Distribution
# ###################

lowpass_energy = np.sum(np.abs(lowpass) ** 2)
highpass_energies = [np.sum(np.abs(band) ** 2) for band in highpass_bands]
total_energy = lowpass_energy + sum(highpass_energies)

print(f"\nEnergy Distribution:")  # noqa: T201
print(f"Lowpass: {lowpass_energy:.2e} ({100*lowpass_energy/total_energy:.1f}%)")  # noqa: T201
for i, energy in enumerate(highpass_energies):
    print(f"Highpass {i}: {energy:.2e} ({100*energy/total_energy:.1f}%)")  # noqa: T201
print(f"Total: {total_energy:.2e}")  # noqa: T201

# %%
# Reconstruction
# ##############

reconstructed = wavelet.backward(lowpass)

# %%
# Reconstructed Image
# ###################

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(reconstructed.T, **opts)
_, cb = create_colorbar(im=im, ax=ax)
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title="Reconstructed Zone Plate")

# %%
# Reconstruction Error
# ####################

error = zone_plate - reconstructed
error_max = np.abs(error).max()
error_opts = {
    "aspect": "equal",
    "cmap": "RdBu_r",
    "vmin": -error_max,
    "vmax": error_max,
}

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(error.T, **error_opts)
_, cb = create_colorbar(im=im, ax=ax)
fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
cb.ax.yaxis.set_major_formatter(fmt)
despine(ax)
ax.set(title=f"Reconstruction Error (max = {error_max:.2e})")

print(f"\nReconstruction Quality:")  # noqa: T201
print(f"Max absolute error: {error_max:.2e}")  # noqa: T201
print(f"Relative error: {error_max / np.abs(zone_plate).max():.2e}")  # noqa: T201
print(f"RMSE: {np.sqrt(np.mean(error**2)):.2e}")  # noqa: T201

