"""
Do Monogenic Curvelets Detect Faults Better?
============================================

A hypothesis: in seismic images with faults, monogenic curvelets should outperform
standard curvelets, because a reflector that is cut or offset by a fault changes
abruptly along its length, and the extra Riesz components (the "inverted phase")
describe that change better than a single coefficient can.

This example tests the hypothesis on a synthetic folded model with a fault of
known position, and reports where it holds and where it does not.

What the experiment measures
----------------------------

A fault attribute is built from the curvelet bands. Along a reflector the wavelet
is nearly constant, so the *relative change of the band field along the local
event direction* is small on intact reflectors and large across a fault. For a
band with channels :math:`f_c` (a list of real or complex fields) and event
direction :math:`t`,

.. math::
   \\mathrm{attribute} = \\sqrt{\\frac{\\sum_c |\\partial_t f_c|^2}
   {\\overline{\\sum_c |f_c|^2}}}

where the overline is a Gaussian smoothing. Two versions are compared:

- **standard**: the channel is the standard UDCT coefficient (the real transform).
- **monogenic**: the channels are the scalar coefficient and its two Riesz
  components, :math:`(\\beta, \\mathcal{R}_1\\beta, \\mathcal{R}_2\\beta)`.

Each attribute is scored against the known fault with the area under the ROC curve
(AUC): 1 is perfect, 0.5 is chance.

Two kinds of bands are distinguished. The **directional** bands of the standard
UDCT are already complex-valued (analytic), so they already have an amplitude and
phase. Only the **isotropic** low-frequency band is purely real, and has no
orientation of its own.
"""

from __future__ import annotations

# %%
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from curvelets.numpy import UDCT
from curvelets.numpy._riesz import riesz_filters
from curvelets.plot import create_colorbar, despine

# %%
# Synthetic folded model with a fault
# ###################################
#
# Reflectors follow a sigmoid-shaped fold, and each carries a Ricker wavelet. A
# fault plane dips through the model. Either the right-hand side is displaced
# vertically by a *throw* (the reflectors are offset), or the reflectors are cut
# abruptly by the fault with no displacement (a *truncation*).

shape = (200, 256)
PAD = 48  # reflect-padding around the image, to avoid FFT wrap-around artifacts
FAULT_DIP = 0.35  # the fault plane is x = 128 + FAULT_DIP * (z - 100)

z: npt.NDArray[np.float64]
x: npt.NDArray[np.float64]
z, x = np.meshgrid(
    np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float), indexing="ij"
)
x_fault = 128 + FAULT_DIP * (z - shape[0] / 2)
fault_trace = x_fault[:, 0]  # x position of the fault at each depth
distance = np.abs(x - x_fault) * np.cos(np.arctan(FAULT_DIP))  # to the fault, samples

rng_model = np.random.default_rng(1)
nominal_depths = np.arange(-20, shape[0] + 20, 9)
depths = nominal_depths + rng_model.uniform(-2, 2, nominal_depths.size)
amplitudes = rng_model.uniform(0.5, 1.0, depths.size) * rng_model.choice(
    [-1, 1], depths.size
)
fold = 12 * np.tanh((x - 128) / 40) + 0.1 * (x - 128)  # sigmoid-shaped fold


def ricker(t: npt.NDArray[np.float64], width: float = 7.0) -> npt.NDArray[np.float64]:
    a = (np.pi * t / width) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_model(throw: float = 0.0, truncate: bool = False) -> npt.NDArray[np.float64]:
    """Folded reflectors, offset by ``throw`` or cut by the fault (``truncate``)."""
    right = x > x_fault
    model = np.zeros(shape)
    for depth, amplitude in zip(depths, amplitudes, strict=True):
        model += amplitude * ricker(z - depth - fold - throw * right)
    if truncate:
        model *= ~right
    return model


# One normalisation for all models, so that noise levels mean the same thing.
scale = make_model().std()


def noisy(
    model: npt.NDArray[np.float64], noise: float, seed: int
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return model / scale + noise * rng.standard_normal(shape)


fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, (title, data) in zip(
    axs,
    (
        ("Offset by a throw of 4 samples", make_model(4.0) / scale),
        ("Same, with noise ($\\sigma = 0.5$)", noisy(make_model(4.0), 0.5, 0)),
        ("Truncated by the fault", make_model(truncate=True) / scale),
    ),
    strict=True,
):
    ax.imshow(data, cmap="gray", aspect="auto", vmin=-3, vmax=3)
    ax.plot(fault_trace, np.arange(shape[0]), "r--", lw=1)
    ax.set(title=title, xlim=(0, shape[1] - 1), ylim=(shape[0] - 1, 0))
    despine(ax)
plt.tight_layout()

# %%
# Band fields without decimation
# ##############################
#
# The attribute is evaluated at every pixel, so the bands are used at full
# resolution. The frequency windows of the transform are ``transform.windows``;
# multiplying the image spectrum by a window and inverting the FFT gives the band
# before decimation (the decimated coefficients of the transform are samples of
# these fields). The event direction of a directional band is perpendicular to the
# centroid frequency of its window. The Riesz components are
# :math:`\widehat{\mathcal{R}_k \beta} = i\,\xi_k/|\xi|\,\widehat{\beta}`.

padded_shape = (shape[0] + 2 * PAD, shape[1] + 2 * PAD)
transform = UDCT(shape=padded_shape, num_scales=3, wedges_per_direction=3)
freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in padded_shape], indexing="ij")
riesz = riesz_filters(padded_shape)

iso_window = transform.windows[0][0][0].to_dense()
directional: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
for scale_windows in transform.windows[1:]:
    for direction in scale_windows:
        for wedge in direction:
            window = wedge.to_dense()
            power = np.abs(window) ** 2
            centroid = np.array([(power * f).sum() / power.sum() for f in freqs])
            tangent = np.array([-centroid[1], centroid[0]])
            directional.append((window, tangent / np.linalg.norm(tangent)))

# The fields are real or complex, so the attribute works with either.
Spectrum: TypeAlias = npt.NDArray[np.complexfloating]

# Smoothing of the denominator and of the structure tensor, in samples.
SIGMA = 6


def spectrum(image: npt.NDArray[np.float64]) -> Spectrum:
    return np.fft.fftn(np.pad(image, PAD, mode="reflect"))


def crop(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return a[PAD:-PAD, PAD:-PAD]


def energy(channels: list[Spectrum]) -> npt.NDArray[np.float64]:
    total = np.zeros(padded_shape)
    for c in channels:
        total += np.abs(np.fft.ifftn(c)) ** 2
    return total


def relative_change(
    channels: list[Spectrum],
    tangent: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Numerator and denominator of the attribute, before summing over bands."""
    num = np.zeros(padded_shape)
    for c in channels:
        d0 = np.fft.ifftn(2j * np.pi * freqs[0] * c)
        d1 = np.fft.ifftn(2j * np.pi * freqs[1] * c)
        num += np.abs(tangent[0] * d0 + tangent[1] * d1) ** 2
    return num, energy(channels)


def finish(
    num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    den = gaussian_filter(den, SIGMA)
    return crop(np.sqrt(num / (den + 1e-3 * den.max())))


def structure_tangent(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Local event direction from the structure tensor of a two-component field."""
    t11: npt.NDArray[np.float64] = gaussian_filter(a * a, SIGMA)
    t22: npt.NDArray[np.float64] = gaussian_filter(b * b, SIGMA)
    t12: npt.NDArray[np.float64] = gaussian_filter(a * b, SIGMA)
    theta = 0.5 * np.arctan2(2 * t12, t11 - t22)  # direction of the wave vector
    return -np.sin(theta), np.cos(theta)


def directional_attribute(
    image_freq: Spectrum, channels: Literal[1, 3]
) -> npt.NDArray[np.float64]:
    """Attribute from the directional (analytic) bands, summed over all of them."""
    num = np.zeros(padded_shape)
    den = np.zeros(padded_shape)
    for window, tangent in directional:
        band = window * image_freq
        fields = (
            [band]
            if channels == 1
            else [band, 1j * riesz[0] * band, 1j * riesz[1] * band]
        )
        n, d = relative_change(fields, tangent)
        num += n
        den += d
    return finish(num, den)


def isotropic_attribute(
    image_freq: Spectrum,
    channels: Literal[1, 3],
    dip: Literal["gradient", "riesz"],
) -> npt.NDArray[np.float64]:
    """Attribute from the isotropic band, along a locally estimated event direction.

    The dip comes from the structure tensor of the gradient of the real band
    (``"gradient"``) or of the Riesz vector (``"riesz"``).
    """
    band = iso_window * image_freq
    riesz_fields = [1j * riesz[0] * band, 1j * riesz[1] * band]
    if dip == "gradient":
        pair = [np.fft.ifftn(2j * np.pi * f * band).real for f in freqs]
    else:
        pair = [np.fft.ifftn(r).real for r in riesz_fields]
    tangent = structure_tangent(pair[0], pair[1])
    fields = [band] if channels == 1 else [band, *riesz_fields]
    return finish(*relative_change(fields, tangent))


def auc(score: npt.NDArray[np.float64], near: float = 4, far: float = 14) -> float:
    """Area under the ROC curve: pixels within ``near`` of the fault against the
    pixels farther than ``far`` (the borders of the image are excluded)."""
    m = 24
    s = score[m:-m, m:-m]
    d = distance[m:-m, m:-m]
    pos, neg = s[d <= near], s[d >= far]
    ranks = rankdata(np.concatenate([pos, neg]))
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
        / (pos.size * neg.size)
    )


# %%
# Attribute maps
# ##############
#
# One example: a throw of 4 samples with noise. The isotropic band is shown for the
# real band alone (standard) and for all three channels (monogenic).

example = spectrum(noisy(make_model(4.0), 0.5, 0))
maps = {
    "Directional, standard": directional_attribute(example, 1),
    "Directional, monogenic": directional_attribute(example, 3),
    "Isotropic, standard": isotropic_attribute(example, 1, "gradient"),
    "Isotropic, monogenic": isotropic_attribute(example, 3, "riesz"),
}

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, attribute) in zip(axs, maps.items(), strict=True):
    im = ax.imshow(attribute, aspect="auto", vmax=np.percentile(attribute, 99))
    create_colorbar(im=im, ax=ax)
    ax.plot(fault_trace, np.arange(shape[0]), "r--", lw=1)
    ax.set(
        title=f"{title}\nAUC = {auc(attribute):.2f}",
        xlim=(0, shape[1] - 1),
        ylim=(shape[0] - 1, 0),
    )
    despine(ax)
plt.tight_layout()

# %%
# Sweep over throw and noise
# ##########################
#
# For each case (three throws and a truncation), noise level and noise seed, the
# four attributes are scored. Lines are the mean over the seeds.

noise_levels = [0.25, 0.5, 1.0, 2.0]
n_seeds = 3
cases: list[tuple[str, dict[str, float | bool]]] = [
    ("Throw 2", {"throw": 2.0}),
    ("Throw 4", {"throw": 4.0}),
    ("Throw 8", {"throw": 8.0}),
    ("Truncation", {"truncate": True}),
]
names = (
    "directional standard",
    "directional monogenic",
    "isotropic standard",
    "isotropic monogenic",
)
scores = np.zeros((len(cases), len(noise_levels), len(names), n_seeds))
for i, (_, kwargs) in enumerate(cases):
    model = make_model(**kwargs)  # type: ignore[arg-type]
    for j, noise in enumerate(noise_levels):
        for seed in range(n_seeds):
            f = spectrum(noisy(model, noise, seed))
            scores[i, j, :, seed] = [
                auc(directional_attribute(f, 1)),
                auc(directional_attribute(f, 3)),
                auc(isotropic_attribute(f, 1, "gradient")),
                auc(isotropic_attribute(f, 3, "riesz")),
            ]
mean_scores = scores.mean(axis=-1)

# %%
# Mean AUC against noise, per case. Solid lines are monogenic, dashed are standard;
# the two directional lines (grey) overlap.

fig, axs = plt.subplots(1, len(cases), figsize=(16, 3.8), sharey=True)
styles = (("0.55", "--"), ("0.25", "-"), ("C0", "--"), ("C3", "-"))
for ax, (case_name, _kwargs), per_case in zip(axs, cases, mean_scores, strict=True):
    for k, (name, (color, ls)) in enumerate(zip(names, styles, strict=True)):
        ax.plot(
            noise_levels, per_case[:, k], color=color, ls=ls, marker="o", label=name
        )
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set(title=case_name, xlabel="noise $\\sigma$", xscale="log", ylim=(0.35, 1.02))
    ax.set_xticks(noise_levels, [f"{n:g}" for n in noise_levels])
    ax.minorticks_off()
axs[0].set_ylabel("AUC")
axs[0].legend(fontsize=8)
plt.tight_layout()

print("Mean AUC (over seeds). Rows: case / noise; columns: attributes")
print(f"{'':>20}" + "".join(f"{n:>24}" for n in names))
for (case_name, _kwargs), per_case in zip(cases, mean_scores, strict=True):
    for noise, row in zip(noise_levels, per_case, strict=True):
        print(f"{case_name:>12} {noise:>6.2f}" + "".join(f"{v:>24.2f}" for v in row))

# %%
# What makes the isotropic monogenic attribute better?
# ####################################################
#
# The monogenic attribute differs from the standard one in two ways: it uses three
# channels instead of one, and it estimates the event direction from the Riesz
# vector instead of from the gradient. A 2x2 ablation separates them (throw of 4).

Channels = Literal[1, 3]
Dip = Literal["gradient", "riesz"]
combos: list[tuple[Channels, Dip]] = [
    (1, "gradient"),
    (3, "gradient"),
    (1, "riesz"),
    (3, "riesz"),
]
ablation = np.zeros((2, len(combos)))
for a, noise in enumerate((0.5, 1.0)):
    for b, (channels, dip) in enumerate(combos):
        ablation[a, b] = np.mean([
            auc(
                isotropic_attribute(
                    spectrum(noisy(make_model(4.0), noise, s)), channels, dip
                )
            )
            for s in range(n_seeds)
        ])

fig, ax = plt.subplots(figsize=(7, 3.8))
width = 0.35
labels = [f"{c} channel{'s' if c > 1 else ''}\n{d} dip" for c, d in combos]
for a, noise in enumerate((0.5, 1.0)):
    ax.bar(
        np.arange(len(combos)) + (a - 0.5) * width,
        ablation[a],
        width,
        label=f"noise $\\sigma$ = {noise}",
    )
ax.set(xticks=np.arange(len(combos)), xticklabels=labels, ylabel="AUC", ylim=(0.4, 1.0))
ax.axhline(0.5, color="k", lw=0.5)
ax.legend()
plt.tight_layout()

# %%
# Conclusions
# ###########
#
# - **Directional bands: no difference.** The standard UDCT directional bands are
#   already analytic, and the Riesz components of a narrow directional band mostly
#   repeat the same information, so the two attributes score within about 0.02 AUC.
#   Both are weak in noise: these bands carry the finest detail and the most noise,
#   so this is a comparison of two equally weak detectors.
# - **Isotropic band, offset faults: monogenic is better.** For example, with a
#   throw of 4 samples and noise :math:`\sigma = 0.5` the AUC is about 0.92
#   (monogenic) against 0.78 (standard), and the gap persists across throws and
#   noise levels. The ablation shows that the gain comes from the three channels,
#   not from the way the event direction is estimated. The monogenic energy
#   :math:`\beta^2 + (\mathcal{R}_1\beta)^2 + (\mathcal{R}_2\beta)^2` does not
#   depend on where in the oscillation a pixel sits, while a lone real coefficient
#   loses sensitivity near its zero crossings. A phase jump across the fault is
#   therefore seen everywhere with the former and only partly with the latter.
# - **Abrupt truncation: no advantage.** Standard and monogenic attributes score
#   alike, and poorly. The idea that monogenic curvelets "stop" a reflector better
#   is not supported here. The benefit is in handling the phase shift of an offset,
#   not in the termination itself.
#
# Caveats: one synthetic model and an attribute chosen for this test. The advantage
# also depends on how much of the signal the isotropic band holds, and it reverses
# when that band is too coarse. With ``num_scales=3`` the isotropic band holds the
# wavelet's main energy. With ``num_scales=4`` (an isotropic band below the
# wavelet's peak frequency) a separate check over five noise seeds, with the same
# throw of 4 samples, gave the opposite ordering: AUC about 0.52 (monogenic) against
# 0.71 (standard) at :math:`\sigma = 0.5`. A plausible but untested reason is that
# the Riesz transform of a band containing the zero frequency is non-local, which
# blurs the fault. So the result is not that monogenic curvelets are better in
# general, but that phase invariance helps detect offsets when the isotropic band
# contains the reflectors.
