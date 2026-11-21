"""
Do Monogenic Curvelets Detect Faults Better?
============================================

A hypothesis: in seismic images with faults, monogenic curvelets
:cite:`Storath2010,Storath2011`, which extend curvelets with the monogenic signal
:cite:`Felsberg2001`, should outperform standard curvelets, because a reflector that is cut or offset by a fault changes
abruptly along its length, and the extra Riesz components (the "inverted phase")
describe that change better than a single coefficient can.

In real sections faults are rarely sharp. Steeply dipping reflectors and fault
planes are poorly imaged without wide offsets, and what is recovered there is
already lower in frequency, so a fault often appears as a *gap*: a zone of dim,
low-frequency, incoherent energy between the two sides.

This example tests the hypothesis on a synthetic folded model with a fault of
known position, first as a sharp fault and then as a fault zone with a gap, and
reports where it holds and where it does not.

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

where the overline is a Gaussian smoothing. The attribute can also be smoothed
itself, by a Gaussian of width ``post``; comparing the two versions at matched
smoothing turns out to decide the result. Two versions are compared:

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
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import rankdata

from curvelets.numpy import UDCT
from curvelets.numpy._riesz import riesz_filters
from curvelets.plot import create_colorbar, despine

# %%
# Synthetic folded model with a fault
# ###################################
#
# Reflectors follow a sigmoid-shaped fold, and each carries a Ricker wavelet. A
# fault plane dips through the model, and the right-hand side is displaced
# vertically by a *throw*. A fault zone replaces the reflectors within a distance
# ``gap / 2`` of the fault plane by weak, low-frequency, incoherent energy, with the
# reflectors tapering off on both sides.

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


def make_model(throw: float = 0.0) -> npt.NDArray[np.float64]:
    """Folded reflectors, offset by ``throw`` samples across the fault."""
    right = x > x_fault
    model = np.zeros(shape)
    for depth, amplitude in zip(depths, amplitudes, strict=True):
        model += amplitude * ricker(z - depth - fold - throw * right)
    return model


def make_fault_zone(
    throw: float, gap: float, fill: float = 0.3
) -> npt.NDArray[np.float64]:
    """Fault zone of width ``gap``: reflectors taper off on both sides, and the zone
    holds low-frequency incoherent energy (``fill`` times the reflector RMS)."""
    model = make_model(throw)
    if gap == 0:
        return model
    keep = 0.5 * (1 + np.tanh((distance - gap / 2) / 2))  # 0 in the zone, 1 outside
    incoherent: npt.NDArray[np.float64] = gaussian_filter(
        np.random.default_rng(123).standard_normal(shape), 3
    )
    incoherent *= fill * model.std() / incoherent.std()
    return keep * model + (1 - keep) * incoherent


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
        ("Fault zone with a gap of 16 samples", make_fault_zone(4.0, 16) / scale),
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
    num: npt.NDArray[np.float64], den: npt.NDArray[np.float64], post: float = 0.0
) -> npt.NDArray[np.float64]:
    """Square root of the normalised ratio, optionally smoothed by ``post`` samples."""
    den = gaussian_filter(den, SIGMA)
    ratio = num / (den + 1e-3 * den.max())
    if post > 0:
        ratio = gaussian_filter(ratio, post)
    return crop(np.sqrt(ratio))


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
    image_freq: Spectrum, channels: Literal[1, 3], post: float = 0.0
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
    return finish(num, den, post)


def isotropic_attribute(
    image_freq: Spectrum,
    channels: Literal[1, 3],
    dip: Literal["gradient", "riesz"],
    post: float = 0.0,
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
    return finish(*relative_change(fields, tangent), post)


def auc(
    score: npt.NDArray[np.float64],
    near: float = 4,
    far: float = 14,
    dist: npt.NDArray[np.float64] | None = None,
) -> float:
    """Area under the ROC curve: pixels within ``near`` of the fault against the
    pixels farther than ``far`` (the borders of the image are excluded). ``dist`` is
    the distance to the fault, by default that of the synthetic model."""
    m = 24
    s = score[m:-m, m:-m]
    d = (distance if dist is None else dist)[m:-m, m:-m]
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
# For each throw, noise level and noise seed, the four attributes are scored,
# without smoothing of the attributes. Lines are the mean over the seeds.

noise_levels = [0.25, 0.5, 1.0, 2.0]
n_seeds = 3
cases: list[tuple[str, float]] = [("Throw 2", 2.0), ("Throw 4", 4.0), ("Throw 8", 8.0)]
names = (
    "directional standard",
    "directional monogenic",
    "isotropic standard",
    "isotropic monogenic",
)
scores = np.zeros((len(cases), len(noise_levels), len(names), n_seeds))
for i, (_, throw) in enumerate(cases):
    model = make_model(throw)
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

fig, axs = plt.subplots(1, len(cases), figsize=(13, 3.8), sharey=True)
styles = (("0.55", "--"), ("0.25", "-"), ("C0", "--"), ("C3", "-"))
for ax, (case_name, _throw), per_case in zip(axs, cases, mean_scores, strict=True):
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
for (case_name, _throw), per_case in zip(cases, mean_scores, strict=True):
    for noise, row in zip(noise_levels, per_case, strict=True):
        print(f"{case_name:>12} {noise:>6.2f}" + "".join(f"{v:>24.2f}" for v in row))

# %%
# What makes the isotropic monogenic attribute better?
# ####################################################
#
# This section and the next compare unsmoothed attributes; the section on matched
# smoothing below shows how much of the difference survives smoothing.
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
# Faults as a gap
# ###############
#
# Now the fault is a zone of width ``gap`` (see the third panel of the first
# figure). Two effects could change the picture. The attribute at the isotropic
# scale sees about one dominant wavelength, so once the gap is wider than that, the
# two sides no longer meet and there is no phase jump to see; what is left is two
# sets of terminations and a zone of incoherent energy. And a dim zone suggests a
# different attribute: *dimming*, the local envelope energy relative to its
# surroundings, where the smooth monogenic envelope might beat the oscillating real
# band.
#
# The attributes are scored against the whole zone: positives within
# ``gap / 2 + 3`` samples of the fault plane, negatives beyond ``gap / 2 + 14``.


def dimming(
    image_freq: Spectrum, channels: Literal[1, 3], local: float = 1.0
) -> npt.NDArray[np.float64]:
    """Log ratio of the surrounding to the local energy of the isotropic band."""
    band = iso_window * image_freq
    fields = (
        [band] if channels == 1 else [band, 1j * riesz[0] * band, 1j * riesz[1] * band]
    )
    e = energy(fields)
    near_e: npt.NDArray[np.float64] = gaussian_filter(e, local)
    around: npt.NDArray[np.float64] = gaussian_filter(e, 15)
    floor = 1e-6 * around.max()
    return crop(np.log((around + floor) / (near_e + floor)))


def zone_auc(score: npt.NDArray[np.float64], gap: float) -> float:
    return auc(score, near=gap / 2 + 3, far=gap / 2 + 14)


zone = spectrum(noisy(make_fault_zone(4.0, 16), 0.5, 0))
zone_maps = {
    "Change, standard": isotropic_attribute(zone, 1, "gradient"),
    "Change, monogenic": isotropic_attribute(zone, 3, "riesz"),
    "Dimming, standard": dimming(zone, 1),
    "Dimming, monogenic": dimming(zone, 3),
}

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, attribute) in zip(axs, zone_maps.items(), strict=True):
    im = ax.imshow(attribute, aspect="auto", vmax=np.percentile(attribute, 99))
    create_colorbar(im=im, ax=ax)
    ax.set(
        title=f"{title} (isotropic)\nAUC = {zone_auc(attribute, 16):.2f}",
        xlim=(0, shape[1] - 1),
        ylim=(shape[0] - 1, 0),
    )
    despine(ax)
plt.tight_layout()

# %%
# Sweep over the gap width, with and without a throw (noise :math:`\sigma = 0.5`).

gaps = [4, 8, 16, 32]
zone_names = (
    "change standard",
    "change monogenic",
    "dimming standard",
    "dimming monogenic",
)
zone_throws = (0.0, 4.0)
zone_scores = np.zeros((len(zone_throws), len(gaps), len(zone_names)))
for i, throw in enumerate(zone_throws):
    for j, gap in enumerate(gaps):
        per_seed = []
        for seed in range(n_seeds):
            f = spectrum(noisy(make_fault_zone(throw, gap), 0.5, seed))
            per_seed.append([
                zone_auc(isotropic_attribute(f, 1, "gradient"), gap),
                zone_auc(isotropic_attribute(f, 3, "riesz"), gap),
                zone_auc(dimming(f, 1), gap),
                zone_auc(dimming(f, 3), gap),
            ])
        zone_scores[i, j] = np.mean(per_seed, axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
zone_styles = (("C0", "--"), ("C3", "-"), ("C2", "--"), ("C1", "-"))
for ax, throw, per_throw in zip(axs, zone_throws, zone_scores, strict=True):
    for k, (name, (color, ls)) in enumerate(zip(zone_names, zone_styles, strict=True)):
        ax.plot(gaps, per_throw[:, k], color=color, ls=ls, marker="o", label=name)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set(
        title=f"Fault zone, throw {throw:g}",
        xlabel="gap width (samples)",
        xscale="log",
        ylim=(0.45, 1.0),
    )
    ax.set_xticks(gaps, [str(g) for g in gaps])
    ax.minorticks_off()
axs[0].set_ylabel("AUC")
axs[0].legend(fontsize=8)
plt.tight_layout()

print("Mean AUC for fault zones (isotropic band, noise 0.5)")
print(f"{'':>16}" + "".join(f"{n:>20}" for n in zone_names))
for throw, per_throw in zip(zone_throws, zone_scores, strict=True):
    for gap, row in zip(gaps, per_throw, strict=True):
        print(f"throw {throw:g} gap {gap:>2}" + "".join(f"{v:>20.2f}" for v in row))

# %%
# The dimming attribute depends on how much the local energy is smoothed. Without
# smoothing, the real band's energy :math:`\beta^2` oscillates and the monogenic
# envelope helps; with a few samples of smoothing, :math:`\beta^2` becomes an
# envelope itself and the two agree (gap of 16 samples, throw of 4).

print("Dimming AUC against the local smoothing (samples)")
for local in (0.5, 1.0, 2.0, 5.0):
    pair = np.mean(
        [
            [
                zone_auc(dimming(f, c, local), 16)
                for c in (1, 3)
                for f in [spectrum(noisy(make_fault_zone(4.0, 16), 0.5, seed))]
            ]
            for seed in range(n_seeds)
        ],
        axis=0,
    )
    print(f"  {local:>4}: standard {pair[0]:.2f}, monogenic {pair[1]:.2f}")

# %%
# Matched smoothing
# #################
#
# The unsmoothed standard attribute is speckled: the real band oscillates, so its
# along-event change fluctuates with the phase. The monogenic attribute is smooth by
# construction. A fair comparison smooths both by the same amount. The isotropic
# attributes are recomputed with a post-smoothing of width ``post`` (samples), for a
# sharp throw of 4 samples and for a 16-sample fault zone with the same throw.

posts = [0, 1, 2, 3, 4, 6]
smoothing_cases: list[tuple[str, npt.NDArray[np.float64], float | None]] = [
    ("Sharp fault, throw 4", make_model(4.0), None),
    ("Fault zone, gap 16, throw 4", make_fault_zone(4.0, 16), 16.0),
]
smoothing_noise = (0.5, 1.0)
smoothed = np.zeros((len(smoothing_cases), len(smoothing_noise), len(posts), 2))
for i, (_, model, zone_width) in enumerate(smoothing_cases):
    for j, noise in enumerate(smoothing_noise):
        for k, post in enumerate(posts):
            per_seed = []
            for seed in range(n_seeds):
                f = spectrum(noisy(model, noise, seed))
                pair = (
                    isotropic_attribute(f, 1, "gradient", post),
                    isotropic_attribute(f, 3, "riesz", post),
                )
                per_seed.append([
                    auc(a) if zone_width is None else zone_auc(a, zone_width)
                    for a in pair
                ])
            smoothed[i, j, k] = np.mean(per_seed, axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (title, _model, _gap), per_case in zip(
    axs, smoothing_cases, smoothed, strict=True
):
    for j, noise in enumerate(smoothing_noise):
        ax.plot(
            posts,
            per_case[j, :, 0],
            "C0--",
            marker="o",
            alpha=1 - 0.4 * j,
            label=f"standard, noise {noise}",
        )
        ax.plot(
            posts,
            per_case[j, :, 1],
            "C3-",
            marker="o",
            alpha=1 - 0.4 * j,
            label=f"monogenic, noise {noise}",
        )
    ax.set(title=title, xlabel="post-smoothing (samples)", ylim=(0.6, 1.01))
axs[0].set_ylabel("AUC")
axs[0].legend(fontsize=8)
plt.tight_layout()

print("Mean AUC (standard / monogenic) against post-smoothing, isotropic band")
for (title, _model, _gap), per_case in zip(smoothing_cases, smoothed, strict=True):
    for j, noise in enumerate(smoothing_noise):
        cells = "  ".join(
            f"{p}: {a:.2f}/{b:.2f}"
            for p, (a, b) in zip(posts, per_case[j], strict=True)
        )
        print(f"{title}, noise {noise}: {cells}")

# %%
# Real faults: the Thebe gas field
# ################################
#
# Four tiles of 200 time samples by 256 traces from the Thebe gas field volume
# (Exmouth Plateau, NW Australia), with expert fault labels, are included in the
# test data. They come from the fault-recognition dataset of :cite:t:`An2021`,
# released under CC0 at https://doi.org/10.7910/DVN/YBYGBK;
# ``testdata/make_thebe_tiles.py`` cuts them from the original volume. The labels
# are fault zones a few samples wide; positives are pixels within 3 samples of a
# labelled pixel and negatives those farther than 12.

tiles_path = Path("testdata/thebe_tiles.npz")
if not tiles_path.exists():
    tiles_path = Path("../testdata/thebe_tiles.npz")
thebe = np.load(tiles_path)
thebe_seismic = thebe["seismic"].astype(np.float64)
thebe_faults = np.unpackbits(thebe["faults"], axis=-1, count=shape[1]).astype(bool)

real_posts = (0, 3)
real_scores = np.zeros((len(thebe_seismic), len(real_posts), 2))
real_maps = []
for t, (tile, labels) in enumerate(zip(thebe_seismic, thebe_faults, strict=True)):
    dist_to_fault = distance_transform_edt(~labels)
    f = spectrum(tile)
    for k, post in enumerate(real_posts):
        pair = (
            isotropic_attribute(f, 1, "gradient", post),
            isotropic_attribute(f, 3, "riesz", post),
        )
        real_scores[t, k] = [auc(a, near=3, far=12, dist=dist_to_fault) for a in pair]
        if t == 0 and post == 0:
            real_maps = list(pair)

print("Thebe tiles: AUC (standard / monogenic), isotropic band")
for t, row in enumerate(real_scores):
    cells = "  ".join(
        f"post {p}: {a:.2f}/{b:.2f}" for p, (a, b) in zip(real_posts, row, strict=True)
    )
    print(f"  tile {t}: {cells}")
for k, post in enumerate(real_posts):
    a, b = real_scores[:, k].mean(axis=0)
    print(f"  mean, post {post}: {a:.3f}/{b:.3f}")

# %%
# The first tile, its fault labels (red), and the unsmoothed attributes.

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(thebe_seismic[0], cmap="gray", aspect="auto", vmin=-3, vmax=3)
axs[0].contour(thebe_faults[0], levels=[0.5], colors="r", linewidths=0.6)
axs[0].set(title="Thebe tile with fault labels")
for ax, title, attribute, score in zip(
    axs[1:],
    ("standard", "monogenic"),
    real_maps,
    real_scores[0, 0],
    strict=True,
):
    im = ax.imshow(attribute, aspect="auto", vmax=np.percentile(attribute, 99))
    create_colorbar(im=im, ax=ax)
    ax.set(title=f"Isotropic, {title}\nAUC = {score:.2f}")
for ax in axs:
    despine(ax)
plt.tight_layout()

# %%
# Conclusions
# ###########
#
# - **Directional bands: no difference.** The standard UDCT directional bands are
#   already analytic, and the Riesz components of a narrow directional band mostly
#   repeat the same information, so the two attributes score within about 0.02 AUC.
# - **Isotropic band, unsmoothed: monogenic is much better.** For a throw of 4
#   samples and noise :math:`\sigma = 0.5` the AUC is about 0.92 against 0.78, for
#   offsets and for fault zones alike. The ablation attributes this to the three
#   channels: the monogenic energy
#   :math:`\beta^2 + (\mathcal{R}_1\beta)^2 + (\mathcal{R}_2\beta)^2` does not depend
#   on where in the oscillation a pixel sits, so the attribute is smooth, while the
#   real band's attribute is speckled by its oscillation.
# - **At matched smoothing most of that lead disappears.** Smoothing both
#   attributes by 2 samples or more makes them equal at moderate noise, and leaves
#   monogenic ahead by about 0.02 to 0.03 AUC at high noise
#   (:math:`\sigma = 1`). The monogenic attribute is smooth for free and keeps full
#   resolution; once the standard attribute is smoothed, it is nearly as good.
# - **Dimming: no practical advantage**, for the same reason: with a few samples of
#   smoothing the real band's energy is an envelope too.
#
# Real data
# ---------
#
# The Thebe tiles above follow the same pattern. A larger check outside this
# example, on 113 tiles of 256 by 256 samples from ten sections of the same volume,
# gave the same picture: unsmoothed, monogenic wins clearly (mean AUC 0.65 against
# 0.60 with ``num_scales=3``, better in 97% of the tiles); with 3 samples of
# smoothing the lead is about 0.01 (0.709 against 0.701, better in 64% of the
# tiles), and with ``num_scales=4`` it vanishes by 4 to 6 samples of smoothing.
#
# A one-sided truncation, with no reflectors at all on one side of the fault, is
# not tested: a local attribute cannot tell a reflector-free half-plane from a very
# wide gap, so scoring it against a line would be ill-posed.
#
# Caveats: synthetic models and attributes chosen for this test. The comparison
# also depends on how much of the signal the isotropic band holds. On the
# synthetic model with ``num_scales=4`` (an isotropic band below the wavelet's peak
# frequency), a separate unsmoothed check over five seeds gave the opposite
# ordering (AUC about 0.52 for monogenic against 0.71 for standard at
# :math:`\sigma = 0.5`), while on the Thebe data ``num_scales=4`` favoured
# monogenic. The overall result: monogenic curvelets give a phase-invariant,
# speckle-free fault attribute at the isotropic scale without any smoothing, but
# not a substantially better detector once the standard attribute is smoothed.

# %%
# References
# ----------
#
# .. bibliography::
#    :filter: docname in docnames
