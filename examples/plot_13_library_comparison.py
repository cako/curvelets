"""
Speed Compared to Other Transforms
==================================

How long does it take to compute a curvelet decomposition on a GPU, compared
with other multiscale transforms that have Python implementations?

This example **does not run any benchmark**: GPU timings cannot be produced
while the documentation builds. It plots results recorded by
``benchmarks/library_comparison.py`` on a single GPU and stored in
``benchmarks/data/library_comparison.json``. To reproduce or refresh them:

.. code-block:: bash

   pip install kymatio pytorch-wavelets dtcwt
   python benchmarks/library_comparison.py \\
       --export benchmarks/data/library_comparison.json

The transforms compared are:

- **curvelets** (this package, ``curvelets.torch.UDCT``): with and without
  ``cuda_graphs=True``, and the NumPy implementation on the CPU.
- `Kymatio <https://www.kymat.io/>`_ ``Scattering2D``: second-order wavelet
  scattering. It is not invertible.
- `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_: the
  separable DWT and the dual-tree complex wavelet transform (DTCWT), on the GPU.
- `dtcwt <https://dtcwt.readthedocs.io/>`_: the NumPy DTCWT, CPU only
  (2D and 3D).

.. note::

   The transforms are **not equivalent**. They differ in redundancy
   (coefficients per input sample), directional selectivity and in what the
   coefficients are meant for. The numbers below measure the cost of computing
   each decomposition, not the cost of producing the same result.
   The sizes were chosen to show curvelets at their best (small and medium
   inputs, where the GPU path is launch-bound and CUDA graphs help most) but
   all sizes measured are shown.
"""

# sphinx_gallery_thumbnail_number = 1

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# %%
# Load the registered results
# ---------------------------
RESULTS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "data"
    / "library_comparison.json"
    if "__file__" in globals()
    else Path("benchmarks/data/library_comparison.json")
)
REFERENCE = "curvelets (torch, CUDA graphs)"

if RESULTS.exists():
    data = json.loads(RESULTS.read_text())
    meta = data["meta"]
    rows = [
        r
        for r in data["results"]
        if r["median_ms"] is not None and r["direction"] == "forward"
    ]
    print(f"Device: {meta['device']} | date: {meta['date']} | float32, batch 1")
    print("Versions:", {k: v for k, v in meta["versions"].items() if v})
else:
    data, rows = None, []
    print(f"No results found at {RESULTS}; run benchmarks/library_comparison.py.")

# %%
# Forward transform time
# ----------------------
# Median time of one forward transform of a single 2D image, in milliseconds
# (log scale). Libraries that were not run for a size are left out.

if rows:
    sizes = sorted({tuple(r["shape"]) for r in rows if len(r["shape"]) == 2})
    libs = list(dict.fromkeys(r["library"] for r in rows if len(r["shape"]) == 2))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(libs), 1)
    for i, lib in enumerate(libs):
        t = {tuple(r["shape"]): r for r in rows if r["library"] == lib}
        vals = [t[s]["median_ms"] if s in t else np.nan for s in sizes]
        label = f"{lib} ({next(iter(t.values()))['device']})"
        ax.bar(np.arange(len(sizes)) + i * width, vals, width, label=label)
    ax.set_xticks(np.arange(len(sizes)) + width * (len(libs) - 1) / 2)
    ax.set_xticklabels([f"{s[0]}²" for s in sizes])
    ax.set_yscale("log")
    ax.set_ylabel("forward time (ms, lower is better)")
    ax.set_title(f"2D forward transform, {meta['device']}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

# %%
# Time relative to curvelets
# --------------------------
# Time of each library divided by the time of ``curvelets.torch.UDCT`` with
# ``cuda_graphs=True`` on the same input. Values above one mean that library
# is slower than curvelets; values below one mean it is faster. Both
# directions are listed for the invertible transforms, together with the
# redundancy (real-valued coefficients per input sample).


def ratio_table() -> str:
    allrows = [r for r in data["results"] if r["median_ms"] is not None]
    keys = sorted(
        {(tuple(r["shape"]), r["direction"]) for r in allrows},
        key=lambda k: (len(k[0]), k[0], k[1]),
    )
    ref = {
        (tuple(r["shape"]), r["direction"]): r
        for r in allrows
        if r["library"] == REFERENCE
    }
    header = ["library", "device"] + [
        f"{'x'.join(map(str, s))} {d[:3]}" for s, d in keys
    ]
    out = [" | ".join(header), " | ".join("---" for _ in header)]
    for lib in dict.fromkeys(r["library"] for r in allrows):
        mine = {
            (tuple(r["shape"]), r["direction"]): r
            for r in allrows
            if r["library"] == lib
        }
        cells = []
        for k in keys:
            if k in mine and k in ref:
                cells.append(f"{mine[k]['median_ms'] / ref[k]['median_ms']:.1f}")
            else:
                cells.append("-")
        device = next(iter(mine.values()))["device"]
        out.append(" | ".join([lib, device, *cells]))
    return "\n".join(out)


if rows:
    print(ratio_table())

# %%
# Where curvelets is fastest
# --------------------------
# The largest advantage of ``curvelets.torch`` over each library, and the size
# at which it occurs. For honest context, the smallest advantage (or largest
# disadvantage) is reported too.

if rows:
    ref = {
        (tuple(r["shape"]), r["direction"]): r
        for r in rows
        if r["library"] == REFERENCE
    }
    for lib in dict.fromkeys(r["library"] for r in rows):
        if lib == REFERENCE:
            continue
        ratios = {
            tuple(r["shape"]): r["median_ms"]
            / ref[(tuple(r["shape"]), "forward")]["median_ms"]
            for r in rows
            if r["library"] == lib and (tuple(r["shape"]), "forward") in ref
        }
        if not ratios:
            continue
        best = max(ratios, key=ratios.get)
        worst = min(ratios, key=ratios.get)
        print(
            f"{lib}: {ratios[best]:.1f}x slower at {'x'.join(map(str, best))}, "
            f"{ratios[worst]:.1f}x at {'x'.join(map(str, worst))}"
        )
