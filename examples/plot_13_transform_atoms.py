r"""
Comparison of Transform Atoms
=============================

An *atom* of an invertible transform is the image that a single coefficient
stands for: set one coefficient to one, all the others to zero, and invert the
transform. This example compares the atoms of the UDCT :cite:`Nguyen2010` (this
package) with the ones of the discrete wavelet transform (DWT) and the dual-tree
complex wavelet transform (DTCWT) :cite:`Kingsbury2001`, and with the images that
excite one coefficient of Kymatio's scattering transform :cite:`Bruna2013,Andreux2019`
the most (the scattering transform is not invertible, so it
has no atoms). It closes with the redundancy of each transform.
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
# A UDCT for a 256 x 256 image with four scales and three wedges per direction
# at the coarsest directional scale. The number of wedges doubles with every
# scale (3, 6, 12), so the finest scale has the sharpest directions.

shape = (256, 256)
C = UDCT(shape=shape, num_scales=4, wedges_per_direction=3)


def curvelet(scale: int, direction: int, wedge: int) -> npt.NDArray[np.floating]:
    """Atom of the UDCT: the image of the coefficient at the center of a wedge."""
    coeffs = C.forward(np.zeros(shape))  # right structure, all zeros
    array = coeffs[scale][direction][wedge]
    array[array.shape[0] // 2, array.shape[1] // 2] = 1.0
    return C.backward(coeffs)


def crop(img: npt.NDArray[np.floating], half: int = 48) -> npt.NDArray[np.floating]:
    """Central ``2 * half`` square, where a centered coefficient puts its atom."""
    cy, cx = (s // 2 for s in img.shape)
    return img[cy - half : cy + half, cx - half : cx + half]


# %%
# Atoms
# #####
#
# Besides the UDCT we use the two wavelet transforms of
# `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_: the separable
# discrete wavelet transform (DWT, Daubechies 3) and the dual-tree complex
# wavelet transform (DTCWT).
#
# Because the requirements of these libraries are incompatible with this
# library's (recent NumPy and SciPy), their atoms are stored instead of being
# computed while the documentation builds. The same goes for the redundancy
# numbers further down. Both are registered by
# ``benchmarks/library_comparison.py --stats``.
#
# One row per transform, one column per orientation of a single scale. The DWT
# only has three orientations (horizontal, vertical and diagonal, so it cannot
# tell the two diagonals apart); the DTCWT has six, spaced by 30 degrees;
# the UDCT has as many wedges as it is configured to: this scale has twelve (six
# directions are shown, covering about half of all orientations), and the finest
# scale has more.
#
# Kymatio's scattering transform cannot be inverted (it takes a modulus), so a
# coefficient has no synthesis atom. We show instead the image that excites one
# coefficient the most: the unit-norm image :math:`x` that maximizes the
# coefficient at the center of its output. The scattering transform is positively homogeneous,
# :math:`S(cx) = cS(x)`, so linearizing it around the current image and
# maximizing over the sphere gives the update :math:`x \leftarrow \nabla S(x) / \|\nabla S(x)\|`
# with no step size (for first-order coefficients; see below for second order). The gradient, which is the adjoint of the linearized transform,
# comes from PyTorch autograd. Each step costs one forward and one backward pass of the transform.
# The rows "Kymatio, order 1" are first-order coefficients (a Morlet wavelet
# modulus, smoothed) of one scale and six orientations; the row "Kymatio, order 2"
# are second-order coefficients that follow a fixed first-order path.


def find_registered(name: str) -> Path:
    """Locate ``benchmarks/data/<name>``, wherever the script is run from.

    ``__file__`` is not defined in every environment that runs gallery
    scripts (Read the Docs, notebooks), so the search also walks up from the
    current directory.
    """
    here = [Path(__file__).resolve().parent] if "__file__" in globals() else []
    for start in [*here, Path.cwd()]:
        for folder in [start, *start.parents]:
            candidate = folder / "benchmarks" / "data" / name
            if candidate.exists():
                return candidate
    return Path("benchmarks") / "data" / name


STATS = find_registered("library_stats.npz")
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
    kymatio_rows = [
        ("Kymatio, order 1, scale 2 (maximizing image)", "Kymatio/order1/j2/o{}"),
        (
            "Kymatio, order 2, paths (2, 2) -> (3, o) (maximizing image)",
            "Kymatio/order2/j23/o2{}",
        ),
    ]
    rows += [
        (name, [registered[tmpl.format(o)] for o in range(6)])
        for name, tmpl in kymatio_rows
        if tmpl.format(0) in registered
    ]
    fig, axs = plt.subplots(
        len(rows),
        6,
        figsize=(9, 2.4 * len(rows) / 2),
        squeeze=False,
    )
    # fixed margins: the layout engine widens a column for a long row name
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.45
    )
    for ax in axs.ravel():
        ax.axis("off")
    for r, (name, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            patch = crop(img, 32)
            vmax = np.abs(patch).max()
            axs[r, c].imshow(patch, cmap="gray", vmin=-vmax, vmax=vmax)
        # in_layout=False: a long name must not widen the first column
        label = axs[r, 0].text(0, 1.04, name, transform=axs[r, 0].transAxes, fontsize=9)
        label.set_in_layout(False)
    plt.show()

# %%
# Convergence of the Kymatio atoms
# --------------------------------
#
# The value of the coefficient along the iteration, starting from a random
# image of unit norm. The first-order coefficients are convex in the image, so
# the iteration cannot decrease them: they reach a plateau in a few steps.
# The second-order coefficients are not convex: the full jump overshoots and
# the coefficient bounces, so their update is damped,
# :math:`x \leftarrow (1-a)\,x + a\,\nabla S(x)/\|\nabla S(x)\|` with :math:`a = 0.2`
# (followed by renormalization). It converges smoothly in 10 to 15 steps to a
# larger coefficient than the full jump reaches (about 50% larger on the paths
# we tried). The image of the best iterate is kept.

if registered is not None:
    curves = {
        k[len("curve/") :]: registered[k]
        for k in registered.files
        if k.startswith("curve/")
    }
    if curves:
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        for key, values in curves.items():
            order = int(key.split("/")[1][-1])
            ax.semilogy(
                values,
                color="C0" if order == 1 else "C1",
                alpha=0.6,
                label=f"order {order}" if key.endswith(("j1/o0", "j23/o00")) else None,
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("coefficient value")
        ax.legend()
        ax.grid(alpha=0.3)
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
    ranked = sorted(red, key=lambda name: red[name])  # smallest at the bottom
    invertible = {"Kymatio\nscattering": False}
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for i, name in enumerate(ranked):
        bar = ax.barh(
            i,
            red[name],
            height=0.62,
            color="C3" if name == "UDCT" else "0.62",
            hatch=None if invertible.get(name, True) else "///",
            edgecolor="white",
        )
        ax.bar_label(bar, labels=[f"{red[name]:.2f}x"], padding=4, fontsize=10)
    ax.axvline(1, color="0.25", lw=1, ls="--")
    ax.set_ylim(-0.95, len(ranked) - 0.4)
    ax.text(1.04, -0.82, "orthogonal basis (1x)", fontsize=8, color="0.25")
    ax.set_yticks(range(len(ranked)), [n.replace("\n", " ") for n in ranked])
    ax.set_xlim(0, max(red.values()) * 1.18)
    ax.set_xlabel("coefficients per pixel")
    ax.set_title("Redundancy of a 256 x 256 transform", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.99,
        0.02,
        "hatched: not invertible",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="0.35",
    )
    plt.show()

# %%
# References
# ==========
#
# .. bibliography::
#    :filter: docname in docnames
