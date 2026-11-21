"""
Local Orientation at Isotropic Scales
=====================================

The curvelet tiling is directional only at the fine scales. The coarse, isotropic
low-frequency band is a single scalar array: in the ``"real"`` and ``"complex"``
transforms it carries no information about local orientation.

The Riesz components of the monogenic transform do. For a locally oriented
pattern with wave vector direction :math:`\\theta`, the two Riesz components are
proportional to :math:`(\\cos\\theta, \\sin\\theta)` times a common factor, so

.. math::
   \\hat\\theta = \\operatorname{atan2}(\\mathcal{R}_2\\beta,\\ \\mathcal{R}_1\\beta)
   \\pmod \\pi

estimates the local orientation at every location of the isotropic band
:cite:`Storath2011`, after local averaging of the Riesz products to stabilise it
at the zero crossings of the carrier. This example checks that estimate against three wavepackets
with known orientations.
"""

from __future__ import annotations

from typing import Literal

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

# %%
# Test image with known orientations
# ##################################
#
# Three wavepackets whose carriers have different wave vector directions
# :math:`\theta` (in radians, measured in the array index plane).

shape = (256, 256)
x: npt.NDArray[np.int64]
y: npt.NDArray[np.int64]
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

packets = [  # (cx, cy, wavenumber, theta, sigma)
    (70, 70, 0.22, 0.4, 20),
    (180, 80, 0.22, 1.6, 20),
    (125, 185, 0.22, 2.6, 22),
]
image = np.zeros(shape)
theta_true = np.full(shape, np.nan)
best = np.zeros(shape)
for cx, cy, k, theta, sigma in packets:
    u = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    env = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    image += env * np.cos(k * u)
    closer = env > best
    theta_true[closer] = theta % np.pi
    best = np.maximum(best, env)

# %%
# Transforms
# ##########

kinds: tuple[Literal["real", "complex", "monogenic"], ...] = (
    "real",
    "complex",
    "monogenic",
)
bands = {}
for kind in kinds:
    coeffs = UDCT(
        shape=shape,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
    ).forward(image)
    bands[kind] = coeffs[0][0][0]

print("Low band shapes (real, complex, monogenic):")
print(f"  {bands['real'].shape}, {bands['complex'].shape}, {bands['monogenic'].shape}")

# %%
# Orientation from the Riesz components
# #####################################

mono = bands["monogenic"]
r1, r2 = mono[..., 2], mono[..., 3]

# The Riesz vector of an oscillating pattern vanishes at the zero crossings of the
# carrier, where its direction is noise. Averaging the products over about one
# wavelength (a structure tensor of the Riesz vector) gives a robust estimate.
sigma_avg = 4  # in low-band samples
t11: npt.NDArray[np.float64] = gaussian_filter(r1 * r1, sigma_avg)
t22: npt.NDArray[np.float64] = gaussian_filter(r2 * r2, sigma_avg)
t12: npt.NDArray[np.float64] = gaussian_filter(r1 * r2, sigma_avg)
theta_est = (0.5 * np.arctan2(2 * t12, t11 - t22)) % np.pi
amp = np.sqrt((mono**2).sum(axis=-1))

step = shape[0] // mono.shape[0]
theta_ref = theta_true[::step, ::step]
weight = best[::step, ::step]
signal = (weight > 0.5) & (amp > 0.1 * amp.max())


def angular_error(
    est: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Smallest difference between orientations (period :math:`\\pi`)."""
    d = np.abs(est - ref) % np.pi
    return np.minimum(d, np.pi - d)


err = angular_error(theta_est, theta_ref)
print(
    f"Orientation error where the envelope > 0.5 "
    f"(median / 95th percentile): {np.degrees(np.median(err[signal])):.2f} / "
    f"{np.degrees(np.percentile(err[signal], 95)):.2f} degrees"
)

# %%
# Results
# #######

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
vmax = np.abs(bands["real"]).max()
im = axs[0].imshow(
    bands["real"].real.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax
)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title="Real / complex low band\n(no orientation information)")

opts = {"aspect": "equal", "cmap": "hsv", "vmin": 0, "vmax": np.pi}
im = axs[1].imshow(np.ma.masked_where(~signal, theta_ref).T, **opts)
create_colorbar(im=im, ax=axs[1])
axs[1].set(title=r"True orientation $\theta$")

im = axs[2].imshow(np.ma.masked_where(~signal, theta_est).T, **opts)
create_colorbar(im=im, ax=axs[2])
axs[2].set(title=r"Monogenic estimate $\hat\theta$")

im = axs[3].imshow(
    np.degrees(np.ma.masked_where(~signal, err)).T,
    aspect="equal",
    cmap="magma",
    vmin=0,
    vmax=10,
)
create_colorbar(im=im, ax=axs[3])
axs[3].set(title="Error (degrees)")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Takeaways
# #########
#
# - The isotropic band of the real and complex transforms is one scalar array;
#   it cannot say which way a feature is oriented.
# - The Riesz components of the monogenic band give a local orientation estimate
#   at the coarse scales, where the directional curvelet tiling has no resolution.
