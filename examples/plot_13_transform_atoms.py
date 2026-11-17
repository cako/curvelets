r"""
What Does a Point in Transformed Space Look Like?
=================================================

Each curvelet coefficient is the inner product of the image with one
*curvelet*: a small, oriented, band-limited wave packet. Conversely, the
inverse transform builds an image by adding up curvelets weighted by the
coefficients. So a single non-zero coefficient, a "point" in transformed space,
is the image of one curvelet: a localized, oriented oscillation.

Because the UDCT is a tight frame, the inverse transform
:meth:`~curvelets.numpy.UDCT.backward` is the adjoint of the forward
transform, and applying it to a coefficient array that is zero except for one
entry returns exactly that curvelet. This example looks at how the position,
scale and direction of the point decide what the curvelet looks like.

The second half compares those atoms with the ones of other multiscale
transforms, and compares how many coefficients each transform produces (its
*redundancy*).
"""

# sphinx_gallery_thumbnail_number = 1

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from curvelets.numpy import UDCT

# %%
# Setup
# #####
#
# We use a 256 x 256 image with four scales and three wedges per direction at
# the coarsest directional scale. The number of wedges doubles with every
# scale (3, 6, 12), so the finest scale has the sharpest directions.

shape = (256, 256)
C = UDCT(shape=shape, num_scales=4, wedges_per_direction=3)
n_wedges = [[len(direction) for direction in scale] for scale in C.coefficient_shapes()]
print("wedges per direction at each scale:", n_wedges)

# %%
# A function that returns the curvelet of one coefficient
# -------------------------------------------------------
#
# We start from an all-zero coefficient structure, set one entry, and invert.
# ``position`` is the index of the coefficient inside its wedge; ``None`` selects the
# center of the wedge's array.


def curvelet(
    scale: int,
    direction: int,
    wedge: int,
    position: tuple[int, int] | None = None,
    value: complex = 1.0,
) -> npt.NDArray[np.floating]:
    """Image of the single coefficient ``value`` at ``position``."""
    coeffs = C.forward(np.zeros(shape))  # right structure, all zeros
    array = coeffs[scale][direction][wedge]
    if position is None:
        position = (array.shape[0] // 2, array.shape[1] // 2)
    array[position] = value
    return C.backward(coeffs)


def crop(img: npt.NDArray[np.floating], half: int = 48) -> npt.NDArray[np.floating]:
    """Central ``2 * half`` square, where a centered coefficient puts its curvelet."""
    cy, cx = (s // 2 for s in img.shape)
    return img[cy - half : cy + half, cx - half : cx + half]


# %%
# Scale and direction
# ###################
#
# Each column is a different wedge, which is a different direction; each row is
# a different scale. All coefficients sit at the center of their array, so every
# curvelet is centered on the image center. Going down the rows, the curvelets
# get finer, with more oscillations in a smaller support, and the wedge count
# grows, so neighboring directions get closer together. The second row of
# panels shows the Fourier magnitude of the curvelet. Its support is the wedge
# of Fourier space that the coefficient belongs to, together with the opposite
# wedge: the real transform keeps both, which is why the curvelet is real. In
# the image, the curvelet oscillates across the direction of its wedge and is
# elongated along the perpendicular, like the ridge of a wave packet.

picks = [
    (1, 0, [0, 1, 2]),  # scale 1: three wedges of the first direction
    (2, 0, [0, 2, 5]),
    (3, 0, [0, 4, 11]),
]
fig, axs = plt.subplots(2 * len(picks), 3, figsize=(7, 14), layout="constrained")
for r, (scale, direction, wedges) in enumerate(picks):
    for c, wedge in enumerate(wedges):
        img = curvelet(scale, direction, wedge)
        vmax = np.abs(img).max()
        ax = axs[2 * r, c]
        ax.imshow(crop(img), cmap="gray", vmin=-vmax, vmax=vmax)
        ax.set_title(f"scale {scale}, wedge {wedge}", fontsize=9)
        spec = np.fft.fftshift(np.abs(np.fft.fft2(img)))
        axs[2 * r + 1, c].imshow(spec, cmap="magma")
for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axs[0, 0].set_ylabel("image")
axs[1, 0].set_ylabel("|FFT|")
plt.show()

# %%
# Position
# ########
#
# Moving the point *inside* a wedge's coefficient array moves the curvelet
# across the image, without changing its shape. Left, a coefficient array with
# three points; right, the sum of their curvelets. Each curvelet is centered at the
# image position that corresponds to its coefficient. The coefficient arrays are
# subsampled (coarser scales and thinner wedges use fewer samples), so one step in the
# coefficient array moves the curvelet by several pixels.

scale, direction, wedge = 2, 0, 1
shape_w = C.coefficient_shapes()[scale][direction][wedge]
points = [
    (shape_w[0] // 4, shape_w[1] // 4),
    (shape_w[0] // 2, shape_w[1] // 2),
    (3 * shape_w[0] // 4, shape_w[1] // 2),
]
coeff_map = np.zeros(shape_w)
total = np.zeros(shape)
for p in points:
    coeff_map[p] = 1
    total += curvelet(scale, direction, wedge, position=p)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
ax0.imshow(coeff_map, cmap="gray_r", interpolation="nearest")
ax0.set_title(f"coefficient array {shape_w}")
vmax = np.abs(total).max()
ax1.imshow(total, cmap="gray", vmin=-vmax, vmax=vmax)
ax1.set_title("sum of the three curvelets")
for ax in (ax0, ax1):
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %%
# Atoms of other transforms
# #########################
#
# The same experiment works for any invertible transform: set one
# coefficient, invert, look at the image. Besides the UDCT we use the two
# wavelet transforms of `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_:
# the separable discrete wavelet transform (DWT, Daubechies 3) and the
# dual-tree complex wavelet transform (DTCWT).
#
# Those atoms are not computed while the documentation builds. They are
# registered, together with the redundancy numbers further down, by
# ``benchmarks/library_comparison.py --stats``, which also covers the
# libraries that cannot be installed alongside a recent NumPy or SciPy.
#
# One row per transform, one column per orientation of a single scale. The DWT
# only has three orientations (horizontal, vertical and diagonal, so it cannot
# tell the two diagonals apart); the DTCWT has six, spaced by 30 degrees;
# the UDCT has as many wedges as it is configured to: this scale has twelve (six
# directions are shown, covering a quarter of all orientations), and the finest
# scale has more.

STATS = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "library_stats.npz"
    if "__file__" in globals()
    else Path("benchmarks/data/library_stats.npz")
)
registered = np.load(STATS) if STATS.exists() else None
if registered is None:
    print(
        f"No registered atoms at {STATS}; run benchmarks/library_comparison.py --stats."
    )
else:
    stats = json.loads(str(registered["stats"]))

if registered is not None:
    rows = [
        ("UDCT (this package)", [curvelet(2, 0, w) for w in range(6)]),
        (
            "DTCWT (Pytorch Wavelets)",
            [registered[f"DTCWT/L3/o{o}"] for o in range(6)],
        ),
        ("DWT db3 (Pytorch Wavelets)", [registered[f"DWT/L3/o{o}"] for o in range(3)]),
    ]
    fig, axs = plt.subplots(
        len(rows), 6, figsize=(9, 4.8), layout="constrained", squeeze=False
    )
    for ax in axs.ravel():
        ax.axis("off")
    for r, (name, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            patch = crop(img, 32)
            vmax = np.abs(patch).max()
            axs[r, c].imshow(patch, cmap="gray", vmin=-vmax, vmax=vmax)
        axs[r, 0].set_title(name, loc="left", fontsize=9)
    plt.show()

# %%
# Redundancy
# ##########
#
# How many (real-valued) numbers does each transform return for a 256 x 256 image?
# The redundancy is that count divided by the 65536 pixels. A redundancy of one is
# an orthogonal basis, as the DWT nearly is; the price of the redundancy of a frame is
# memory and time, its reward is a representation that is stable to shifts and
# separates more orientations. Two transforms have an extra caveat: Kymatio's
# scattering transform is not invertible, and its number is the size of its second-order
# output, computed from the formula for ``J=4``, ``L=8``.

if registered is not None:
    names = {
        "curvelets (torch, CUDA graphs)": "UDCT",
        "pytorch_wavelets DTCWT": "DTCWT",
        "pytorch_wavelets DWT": "DWT db3",
        "Kymatio Scattering2D": "Kymatio\nscattering",
    }
    red = {names[k]: v for k, v in stats["redundancy"].items() if k in names}
    # The live UDCT count must agree with the registered one.
    live = sum(
        w.size * (2 if np.iscomplexobj(w) else 1)
        for scale in C.forward(np.zeros(shape))
        for direction in scale
        for w in direction
    )
    print(
        f"UDCT redundancy, live {live / np.prod(shape):.2f}x, registered {red['UDCT']:.2f}x"
    )
    for name, value in sorted(red.items(), key=lambda kv: kv[1]):
        print(f"{name.replace(chr(10), ' '):18s} {value:5.2f}x")
    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
    ax.bar(list(red), list(red.values()), color="0.35")
    ax.set_ylabel("coefficients per pixel")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    for i, v in enumerate(red.values()):
        ax.text(i, v, f"{v:.1f}x", ha="center", va="bottom")
    plt.show()
