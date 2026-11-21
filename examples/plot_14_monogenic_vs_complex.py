"""
Monogenic vs. Real and Complex Amplitude/Phase
==============================================

This example shows a case where the monogenic curvelet transform
:cite:`Storath2010,Storath2011` gives a meaningful amplitude and phase and the
real and complex transforms do not.

A real-valued coefficient :math:`\\beta` cannot be split into an envelope and an
oscillation: :math:`|\\beta|` still oscillates and the phase is only a 0/:math:`\\pi`
jump. The usual remedy is an *analytic* signal, which is what the ``"complex"``
transform provides at the directional scales. At the coarse, **isotropic** scales,
however, the low-frequency band is symmetric in frequency and its coefficients are
real in every mode. There is nothing analytic to form, so ``"complex"`` offers no
improvement over ``"real"`` there.

The monogenic transform adds the Riesz transforms of the band, so that

.. math::
   A = \\sqrt{\\beta^2 + (\\mathcal{R}_1\\beta)^2 + (\\mathcal{R}_2\\beta)^2},
   \\qquad
   \\varphi = \\operatorname{atan2}\\left(\\sqrt{(\\mathcal{R}_1\\beta)^2 +
   (\\mathcal{R}_2\\beta)^2},\\ \\beta\\right)

are a smooth envelope and a continuous phase at every scale.

We use two Gabor wavepackets with known envelopes (an oscillatory event with a
slowly varying amplitude, as in seismic data) and compare the three transforms on
the isotropic low-frequency band.
"""

from __future__ import annotations

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from curvelets.numpy import UDCT
from curvelets.plot import create_colorbar, despine

# %%
# Test image with a known envelope
# ################################
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
image = p1 + p2
true_envelope = np.maximum(env1, env2)

# %%
# Transforms
# ##########
#
# Same parameters for all three; only ``transform_kind`` changes.

opts = {"shape": shape, "num_scales": 3, "wedges_per_direction": 3}
bands = {}
for kind in ("real", "complex", "monogenic"):
    coeffs = UDCT(transform_kind=kind, **opts).forward(image)
    bands[kind] = coeffs[0][0][0]  # isotropic low-frequency band

print("Imaginary part of the low band (max |Im|):")
for kind in ("real", "complex"):
    print(f"  {kind:>9}: {np.abs(bands[kind].imag).max():.2e}")

# %%
# The low band is decimated, so sample the true envelope on the same grid.

step = shape[0] // bands["real"].shape[0]
env_ref = true_envelope[::step, ::step]

real_band = bands["real"].real
complex_band = bands["complex"]  # real in practice, see the printout above
mono = bands["monogenic"]
beta = mono[..., 0]  # scalar.real; channel 1 is scalar.imag (zero here)
r1, r2 = mono[..., 2], mono[..., 3]

amp_real = np.abs(real_band)
amp_complex = np.abs(complex_band)
amp_mono = np.sqrt(beta**2 + r1**2 + r2**2)

# Phase is only meaningful where there is signal, so mask the background.
signal = amp_mono > 0.05 * amp_mono.max()
phase_real = np.ma.masked_where(~signal, np.where(real_band >= 0, 0.0, np.pi))
phase_mono = np.ma.masked_where(~signal, np.arctan2(np.hypot(r1, r2), beta))

# %%
# Amplitude
# #########
#
# The real and complex amplitudes are identical and oscillate; the monogenic
# amplitude follows the envelope. Each panel is normalised by its maximum, since
# the band coefficients carry an overall scaling.

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
amp_opts = {"aspect": "equal", "cmap": "magma", "vmin": 0, "vmax": 1}
panels = [
    (env_ref, "True envelope"),
    (amp_real, r"Real: $|\beta|$"),
    (amp_complex, r"Complex: $|\beta|$"),
    (amp_mono, "Monogenic: $A$"),
]
for ax, (data, title) in zip(axs, panels, strict=False):
    im = ax.imshow((data / data.max()).T, **amp_opts)
    create_colorbar(im=im, ax=ax)
    despine(ax)
    ax.set(title=title)
plt.tight_layout()

# %%
# Quantitative comparison
# #######################
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
    ("complex", amp_complex),
    ("monogenic", amp_mono),
):
    print(f"{name:>10} {correlation(a, env_ref):>18.3f} {roughness(a):>10.2e}")

# %%
# Phase
# #####
#
# The real phase is a 0/:math:`\pi` step function that flips at every zero crossing.
# The monogenic phase varies continuously across the oscillations of each wavepacket.

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
vmax = np.abs(real_band).max()
im = axs[0].imshow(real_band.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
create_colorbar(im=im, ax=axs[0])
axs[0].set(title=r"Low band $\beta$")
im = axs[1].imshow(phase_real.T, aspect="equal", cmap="viridis", vmin=0, vmax=np.pi)
create_colorbar(im=im, ax=axs[1])
axs[1].set(title="Real phase\n(0 / $\\pi$ only)")
im = axs[2].imshow(phase_mono.T, aspect="equal", cmap="viridis", vmin=0, vmax=np.pi)
create_colorbar(im=im, ax=axs[2])
axs[2].set(title="Monogenic phase")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Takeaways
# #########
#
# - At fine, directional scales the curvelets are already (nearly) analytic and
#   the real, complex and monogenic amplitudes agree :cite:`Storath2011`.
# - At the isotropic scales the real and complex transforms yield real coefficients
#   with oscillating amplitude and a binary phase.
# - The monogenic transform gives a smooth envelope and a continuous phase at all
#   scales, which is what makes multiscale amplitude/phase analysis (for example
#   regularity estimation from coefficient decay) well defined.
