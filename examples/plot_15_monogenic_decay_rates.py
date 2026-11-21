"""
Regularity Estimation with Monogenic Amplitudes
===============================================

This example follows the application in Section 7 of :cite:`Storath2011`:
estimating local regularity from the decay of curvelet coefficient amplitudes
across scales, :math:`|W f(a, b)| \\sim C_b\\, a^{d_b}`. In practice :math:`d_b` is
obtained by a least-squares fit of :math:`\\log |W f|` against the scale index.

A stable fit needs an amplitude that behaves well at *all* scales. Using the
isotropic low-frequency band as an extra data point is attractive (it adds a scale
for free), but there the real and complex coefficients are real-valued and their
modulus oscillates. This produces thin curved artifacts in the estimated decay
rate and a large fit residual. The monogenic amplitude is smooth, so the fit stays
well behaved.

.. note::
   The UDCT has a single isotropic band and decimated, anisotropic bands, so this
   is a simplified version of Storath's experiment. Amplitudes are brought to a
   common grid by interpolation and, for each scale, the maximum over all
   directions is taken. The fitted exponent therefore approximates the minimum
   decay rate over orientations (the bottom row of Figure 7 in
   :cite:`Storath2011`). The scale index is used in place of :math:`\\log_2 a`.
"""

from __future__ import annotations

from typing import Literal

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import zoom

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

# %%
# Test image
# ##########
#
# A smooth Gaussian blob (regular), a disc and a rectangle (jump discontinuities).

shape = (256, 256)
x: npt.NDArray[np.int64]
y: npt.NDArray[np.int64]
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
image = np.exp(-((x - 70) ** 2 + (y - 70) ** 2) / (2 * 18**2))
image += ((x - 170) ** 2 + (y - 90) ** 2 < 35**2).astype(float)
image += 0.7 * ((np.abs(x - 128) < 40) & (np.abs(y - 190) < 25)).astype(float)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image.T, cmap="gray", aspect="equal")
despine(ax)
ax.set(title="Test image")

# %%
# Amplitude per scale
# ###################
#
# For each ``transform_kind`` we compute, per scale, the maximum amplitude over
# all directions and wedges on the full-resolution grid.

num_scales = 5


Kind = Literal["real", "complex", "monogenic"]


def amplitude(arr: npt.NDArray[np.generic], kind: Kind) -> npt.NDArray[np.float64]:
    """Modulus of a coefficient array (all four channels for monogenic)."""
    if kind == "monogenic":
        return np.sqrt((arr**2).sum(axis=-1))
    return np.abs(arr)


def scale_amplitudes(kind: Kind) -> npt.NDArray[np.float64]:
    transform = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=3,
        transform_kind=kind,
    )
    coeffs = transform.forward(image)
    out = np.zeros((num_scales, *shape))
    for s in range(num_scales):
        for direction in coeffs[s]:
            for wedge in direction:
                a = amplitude(wedge, kind)
                factors = (shape[0] / a.shape[0], shape[1] / a.shape[1])
                out[s] = np.maximum(out[s], zoom(a, factors, order=1))
    return out


kinds: tuple[Kind, ...] = ("real", "complex", "monogenic")
amps = {kind: scale_amplitudes(kind) for kind in kinds}

# %%
# Decay-rate fit
# ##############
#
# Least-squares line of :math:`\log_2 A` against the scale index. We also return
# the RMS residual of the fit: a smooth power law gives a small residual.


def fit_decay(
    amp: npt.NDArray[np.float64], scales: list[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    s: npt.NDArray[np.float64] = np.asarray(scales, dtype=float)
    log_a = np.log2(amp[scales] + 1e-9)
    sc = s - s.mean()
    centered = log_a - log_a.mean(axis=0)
    slope = np.tensordot(sc, centered, axes=(0, 0)) / (sc**2).sum()
    resid = centered - sc[:, None, None] * slope
    return -slope, np.sqrt((resid**2).mean(axis=0))


all_scales = list(range(num_scales))  # includes the isotropic low band
decay = {}
resid = {}
for kind in kinds:
    decay[kind], resid[kind] = fit_decay(amps[kind], all_scales)

# %%
# Decay rate including the isotropic band
# #######################################
#
# The real and complex estimates show thin curved artifacts around the edges; the
# monogenic estimate does not.

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, kind in zip(axs, kinds, strict=True):
    im = ax.imshow(decay[kind].T, aspect="equal", vmin=-1, vmax=4, cmap="viridis")
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=f"{kind}: decay rate $d_b$")
plt.tight_layout()

# %%
# Fit residual
# ############
#
# A direct measure of how well the amplitudes follow a power law across scales.

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, kind in zip(axs, kinds, strict=True):
    im = ax.imshow(resid[kind].T, aspect="equal", vmin=0, vmax=1.5, cmap="magma")
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=f"{kind}: RMS residual")
plt.tight_layout()

# Compare only where there is detail to fit: where the finest-but-one scale
# carries a non-negligible amplitude in the monogenic transform.
mask = amps["monogenic"][1] > 0.1 * amps["monogenic"][1].max()
print(f"Median RMS residual of the log-amplitude fit ({mask.mean():.0%} of pixels):")
for kind in kinds:
    print(f"  {kind:>9}: {np.median(resid[kind][mask]):.3f}")

# %%
# Takeaways
# #########
#
# - Restricting the fit to the anisotropic scales avoids the problem, but leaves
#   few data points when only a few directions are used.
# - Adding the isotropic band gives more points, yet only a smooth amplitude can
#   exploit them. The monogenic amplitude is smooth at every scale, so the fit
#   remains stable :cite:`Storath2011`.
