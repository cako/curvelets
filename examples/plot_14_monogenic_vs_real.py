"""
Monogenic vs. Real Transform
============================

Three cases where the monogenic curvelet transform :cite:`Storath2010,Storath2011`
gives meaningful results at the coarse, **isotropic** scales and the real transform
does not.

The curvelet tiling is directional only at the fine scales. The isotropic
low-frequency band is symmetric in frequency, so its coefficients are real in every
mode: there is no analytic signal, and the transform itself outputs a single real
scalar array per band, with no orientation. (The ``"complex"`` transform, meant for
complex input, does not change this: its isotropic band is the same real array.)
The monogenic transform adds the Riesz transforms of each band, giving

.. math::
   A = \\sqrt{\\beta^2 + (\\mathcal{R}_1\\beta)^2 + (\\mathcal{R}_2\\beta)^2},
   \\qquad
   \\varphi = \\operatorname{atan2}\\left(\\sqrt{(\\mathcal{R}_1\\beta)^2 +
   (\\mathcal{R}_2\\beta)^2},\\ \\beta\\right)

and a local orientation at every scale. The parts below are:

1. **Amplitude and phase** of oscillatory events: smooth envelope and continuous phase.
2. **Regularity estimation** from the decay of amplitudes across scales.
3. **Local orientation** from the Riesz components, compared with a gradient-based
   estimate computed from the real band.
"""

from __future__ import annotations

# %%
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter, zoom

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

Kind = Literal["real", "monogenic"]
kinds: tuple[Kind, ...] = ("real", "monogenic")

# %%
# Part 1: Amplitude and phase
# ###########################
#
# Real coefficients of the isotropic band give an oscillating amplitude
# and a binary phase; the monogenic ones give an envelope and a continuous phase.

# %%
# Test image with a known envelope
# --------------------------------
#
# Two wavepackets with different dips and amplitudes. The carrier wavelengths
# (about 25 samples) place their energy in the lowest isotropic band.

shape = (256, 256)
x: npt.NDArray[np.int64]
y: npt.NDArray[np.int64]
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")


def wavepacket(
    cx: float, cy: float, k: float, theta: float, sigma: float, amp: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a Gabor wavepacket and its envelope."""
    u = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    envelope = amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return envelope * np.cos(k * u), envelope


p1, env1 = wavepacket(90, 90, 0.25, 0.3, 22, 1.0)
p2, env2 = wavepacket(170, 160, 0.20, 1.2, 26, 0.8)
image_gabor = p1 + p2
true_envelope = np.maximum(env1, env2)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
vmax = np.abs(image_gabor).max()
im = axs[0].imshow(image_gabor.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title="Test image")
im = axs[1].imshow(true_envelope.T, aspect="equal", cmap="magma", vmin=0)
create_colorbar(im=im, ax=axs[1])
axs[1].set(title="True envelope")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Transforms of the wavepackets
# -----------------------------
#
# Same parameters for both; only ``transform_kind`` changes.

bands_gabor = {}
for kind in kinds:
    coeffs_gabor = UDCT(
        shape=shape, num_scales=3, wedges_per_direction=3, transform_kind=kind
    ).forward(image_gabor)
    bands_gabor[kind] = coeffs_gabor[0][0][0]  # isotropic low-frequency band

print(
    f"Imaginary part of the real low band (max |Im|): {np.abs(bands_gabor['real'].imag).max():.2e}"
)

# %%
# The low band is decimated, so sample the true envelope on the same grid.

step_gabor = shape[0] // bands_gabor["real"].shape[0]
env_ref = true_envelope[::step_gabor, ::step_gabor]

real_band = bands_gabor["real"].real
mono_gabor = bands_gabor["monogenic"]
beta = mono_gabor[..., 0]  # scalar.real; channel 1 is scalar.imag (zero here)
r1_g, r2_g = mono_gabor[..., 2], mono_gabor[..., 3]

amp_real = np.abs(real_band)
amp_mono = np.sqrt(beta**2 + r1_g**2 + r2_g**2)

# Phase is only meaningful where there is signal, so mask the background.
signal_gabor = amp_mono > 0.05 * amp_mono.max()
phase_real = np.ma.masked_where(~signal_gabor, np.where(real_band >= 0, 0.0, 180.0))
phase_mono = np.ma.masked_where(
    ~signal_gabor, np.degrees(np.arctan2(np.hypot(r1_g, r2_g), beta))
)

# %%
# Amplitude
# ---------
#
# The real amplitude oscillates; the monogenic
# amplitude follows the envelope. Each panel is normalised by its maximum, since
# the band coefficients carry an overall scaling.

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
amp_opts = {"aspect": "equal", "cmap": "magma", "vmin": 0, "vmax": 1}
panels = [
    (env_ref, "True envelope"),
    (amp_real, r"Real: $|\beta|$"),
    (amp_mono, "Monogenic: $A$"),
]
for ax, (data, title) in zip(axs, panels, strict=True):
    im = ax.imshow((data / data.max()).T, **amp_opts)
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=title)
plt.tight_layout()

# %%
# Quantitative comparison
# -----------------------
#
# Correlation with the true envelope, and the relative oscillation left in the
# amplitude (energy of the amplitude's high-pass part, relative to its total).


def correlation(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def roughness(a: npt.NDArray[np.float64]) -> float:
    """Relative energy of the second difference along both axes."""
    d = np.diff(a, n=2, axis=0)[:, 1:-1] ** 2 + np.diff(a, n=2, axis=1)[1:-1, :] ** 2
    return float(d.sum() / (a**2).sum())


print(f"{'':>10} {'corr. w/ envelope':>18} {'roughness':>10}")
for name, a in (
    ("real", amp_real),
    ("monogenic", amp_mono),
):
    print(f"{name:>10} {correlation(a, env_ref):>18.3f} {roughness(a):>10.2e}")

# %%
# Phase
# -----
#
# The real phase is a 0°/180° step function that flips at every zero crossing.
# The monogenic phase varies continuously across the oscillations of each wavepacket.
# The phase lies in [0°, 180°], so a diverging colormap is used: orange is the
# positive lobe (0°), white the zero crossing (90°), purple the negative lobe (180°).

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
vmax = np.abs(real_band).max()
im = axs[0].imshow(real_band.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title=r"Low band $\beta$")
im = axs[1].imshow(phase_real.T, aspect="equal", cmap="PuOr", vmin=0, vmax=180)
_, cb = create_colorbar(im=im, ax=axs[1])
cb.set_ticks([0, 45, 90, 135, 180])
axs[1].set(title="Real phase (°)\n(0° / 180° only)")
im = axs[2].imshow(phase_mono.T, aspect="equal", cmap="PuOr", vmin=0, vmax=180)
_, cb = create_colorbar(im=im, ax=axs[2])
cb.set_ticks([0, 45, 90, 135, 180])
axs[2].set(title="Monogenic phase (°)")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Part 2: Regularity estimation
# #############################
#
# Following Section 7 of :cite:t:`Storath2011`: decay rates of amplitudes across
# scales, with the isotropic band as an extra data point.

# %%
# Test image with edges
# ---------------------
#
# A smooth Gaussian blob (regular), a disc and a rectangle (jump discontinuities).

image_shapes = np.exp(-((x - 70) ** 2 + (y - 70) ** 2) / (2 * 18**2))
image_shapes += ((x - 170) ** 2 + (y - 90) ** 2 < 35**2).astype(float)
image_shapes += 0.7 * ((np.abs(x - 128) < 40) & (np.abs(y - 190) < 25)).astype(float)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image_shapes.T, cmap="gray", aspect="equal")
despine(ax)
ax.set(title="Test image")

# %%
# Amplitude per scale
# -------------------
#
# For each ``transform_kind`` we compute, per scale, the maximum amplitude over
# all directions and wedges on the full-resolution grid.

num_scales = 5


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
    coeffs = transform.forward(image_shapes)
    out = np.zeros((num_scales, *shape))
    for s in range(num_scales):
        for direction in coeffs[s]:
            for wedge in direction:
                a = amplitude(wedge, kind)
                factors = (shape[0] / a.shape[0], shape[1] / a.shape[1])
                out[s] = np.maximum(out[s], zoom(a, factors, order=1))
    return out


amps = {kind: scale_amplitudes(kind) for kind in kinds}

# %%
# Decay-rate fit
# --------------
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
# ---------------------------------------
#
# The real estimates show thin curved artifacts around the edges; the
# monogenic estimate does not.

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, kind in zip(axs, kinds, strict=True):
    im = ax.imshow(decay[kind].T, aspect="equal", vmin=-1, vmax=4, cmap="viridis")
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=f"{kind}: decay rate $d_b$")
plt.tight_layout()

# %%
# Fit residual
# ------------
#
# A direct measure of how well the amplitudes follow a power law across scales.

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
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
# Part 3: Local orientation
# #########################
#
# The Riesz components give an orientation estimate in the isotropic band. The real
# transform has no such output; the closest real-only alternative is to differentiate
# the real band and take the structure tensor of its gradient. That is not part of
# the transform, but it is the natural baseline, and we compare against it.

# %%
# Test image with known orientations
# ----------------------------------
#
# Three wavepackets whose carriers have different wave vector directions
# :math:`\theta` (in degrees, measured in the array index plane). Noise is added
# to make the comparison informative.


packet_params = [  # (cx, cy, wavenumber, theta, sigma)
    (70, 70, 0.22, 0.4, 20),
    (180, 80, 0.22, 1.6, 20),
    (125, 185, 0.22, 2.6, 22),
]
image_packets = np.zeros(shape)
theta_true = np.full(shape, np.nan)
best = np.zeros(shape)
for cx, cy, k, theta, sigma in packet_params:
    u = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    env = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    image_packets += env * np.cos(k * u)
    closer = env > best
    theta_true[closer] = np.degrees(theta) % 180
    best = np.maximum(best, env)

noise_std = 1.0  # relative to unit-amplitude wavepackets
image_packets += noise_std * np.random.default_rng(0).standard_normal(shape)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
vmax = np.abs(image_packets).max()
im = axs[0].imshow(
    image_packets.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax
)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title="Test image (noisy)")
im = axs[1].imshow(
    np.ma.masked_where(best < 0.5, theta_true).T,
    aspect="equal",
    cmap="twilight_shifted",
    vmin=0,
    vmax=180,
)
_, cb = create_colorbar(im=im, ax=axs[1])
cb.set_ticks([0, 45, 90, 135, 180])
axs[1].set(title=r"True orientation $\theta$ (°)")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Low-band coefficients
# ---------------------

bands_packets = {}
for kind in kinds:
    coeffs = UDCT(
        shape=shape,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
    ).forward(image_packets)
    bands_packets[kind] = coeffs[0][0][0]

print("Low band shapes (real, monogenic):")
print(f"  {bands_packets['real'].shape}, {bands_packets['monogenic'].shape}")

# %%
# Orientation estimates
# ---------------------
#
# Both estimators use the same structure tensor of a two-component field,
# averaged over about one wavelength: the Riesz vector :math:`(\mathcal{R}_1\beta,
# \mathcal{R}_2\beta)` for the monogenic band, and the finite-difference gradient
# :math:`\nabla\beta` for the real band. Without averaging, the Riesz vector's
# direction is noise at the zero crossings of the carrier.

sigma_avg = 4  # in low-band samples


def structure_orientation(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Dominant orientation (degrees, mod 180) of the two-component field (a, b)."""
    t11: npt.NDArray[np.float64] = gaussian_filter(a * a, sigma_avg)
    t22: npt.NDArray[np.float64] = gaussian_filter(b * b, sigma_avg)
    t12: npt.NDArray[np.float64] = gaussian_filter(a * b, sigma_avg)
    return np.degrees(0.5 * np.arctan2(2 * t12, t11 - t22)) % 180


mono_packets = bands_packets["monogenic"]
theta_mono = structure_orientation(mono_packets[..., 2], mono_packets[..., 3])

real_packets = bands_packets["real"].real
theta_real = structure_orientation(*np.gradient(real_packets))

amp_packets = np.sqrt((mono_packets**2).sum(axis=-1))
step_packets = shape[0] // mono_packets.shape[0]
theta_ref = theta_true[::step_packets, ::step_packets]
weight = best[::step_packets, ::step_packets]
signal_packets = (weight > 0.5) & (amp_packets > 0.1 * amp_packets.max())


def angular_error(
    est: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Smallest difference between orientations in degrees (period 180)."""
    d = np.abs(est - ref) % 180
    return np.minimum(d, 180 - d)


err_mono = angular_error(theta_mono, theta_ref)
err_real = angular_error(theta_real, theta_ref)
print("Orientation error in degrees where the envelope > 0.5:")
print(f"{'':>20} {'median':>8} {'95th pct':>9}")
for name, e in (("real (gradient)", err_real), ("monogenic (Riesz)", err_mono)):
    print(
        f"{name:>20} {np.median(e[signal_packets]):>8.2f} "
        f"{np.percentile(e[signal_packets], 95):>9.2f}"
    )

# %%
# Results
# -------
#
# The real band itself shows no orientation. The gradient of the real band does,
# but is noisier than the Riesz vector, whose multiplier has unit modulus and so does
# not amplify high-frequency noise the way a derivative does.

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
vmax = np.abs(real_packets).max()
im = axs[0].imshow(real_packets.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title="Real low band")

opts = {"aspect": "equal", "cmap": "twilight_shifted", "vmin": 0, "vmax": 180}
panels = [
    (theta_ref, r"True orientation $\theta$ (°)"),
    (theta_real, "Real band: gradient estimate (°)"),
    (theta_mono, "Monogenic: Riesz estimate (°)"),
]
for ax, (data, title) in zip(axs[1:], panels, strict=True):
    im = ax.imshow(np.ma.masked_where(~signal_packets, data).T, **opts)
    _, cb = create_colorbar(im=im, ax=ax)
    cb.set_ticks([0, 45, 90, 135, 180])
    ax.set(title=title)
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Error
# -----
#
# Angular error of each estimate, on the same scale.

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, (err, title) in zip(
    axs,
    ((err_real, "Real gradient: error (°)"), (err_mono, "Monogenic: error (°)")),
    strict=True,
):
    im = ax.imshow(
        np.ma.masked_where(~signal_packets, err).T,
        aspect="equal",
        cmap="magma",
        vmin=0,
        vmax=20,
    )
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=title)
plt.tight_layout()

# %%
# Takeaways
# #########
#
# - At the fine, directional scales the real and monogenic amplitudes agree
#   :cite:p:`Storath2011`.
# - At the isotropic scales the real transform yields real coefficients
#   with an oscillating amplitude and a binary phase, and the transform gives no
#   orientation. A gradient of the real band can estimate it, with larger errors
#   under noise.
# - The monogenic transform gives a smooth envelope, a continuous phase and a local
#   orientation at every scale, which stabilises multiscale analyses such as
#   regularity estimation from coefficient decay.

# %%
# References
# ##########
#
# .. bibliography::
#    :filter: docname in docnames
