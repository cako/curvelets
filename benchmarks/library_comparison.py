"""Speed of curvelets.torch against other multiscale transforms.

Times the forward (and, where the transform has one, inverse) transform of

- ``curvelets`` (torch, ``cuda_graphs=True`` and eager) and ``curvelets.numpy``,
- ``Kymatio`` (``Scattering2D``, not invertible),
- ``pytorch_wavelets`` (``DWTForward``/``DWTInverse`` and
  ``DTCWTForward``/``DTCWTInverse``),
- ``dtcwt`` (NumPy, CPU only; 2D and 3D),

on one device, in float32, with a batch of one image or volume. Every
library is optional: a missing one is recorded as skipped.

The libraries are not equivalent. They differ in redundancy (coefficients per
input sample), directional selectivity and what the output is meant for, so
``ncoef`` and ``redundancy`` are stored beside every time. Read the numbers as
"cost of computing this decomposition", not as "same result, faster".

Usage (the extra libraries are only needed for their rows)::

    pip install kymatio pytorch-wavelets dtcwt
    python benchmarks/library_comparison.py --export benchmarks/data/library_comparison.json
    python benchmarks/library_comparison.py --markdown benchmarks/data/library_comparison.json
    python benchmarks/library_comparison.py --table speed --format rst
    python benchmarks/library_comparison.py --table redundancy --format md

``--stats`` needs no GPU. It registers what ``examples/plot_13_transform_atoms.py``
shows: the redundancy of each transform and the image-space atom (the
synthesis of a single coefficient) of every wavelet subband::

    python benchmarks/library_comparison.py --stats benchmarks/data/library_stats.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

NUM_SCALES = 4
WEDGES_PER_DIRECTION = 3
DEFAULT_SHAPES_2D = [256, 512, 1024, 2048]
DEFAULT_SHAPES_3D = [64]


@dataclass
class Row:
    """One timing: a library, a size and a direction."""

    library: str
    device: str
    shape: list[int]
    direction: str  # "forward" or "inverse"
    median_ms: float | None
    min_ms: float | None
    ncoef: int | None  # real-valued coefficients produced by the forward
    redundancy: float | None  # ncoef / input size
    invertible: bool
    config: str
    status: str = "ok"


@dataclass
class Transform:
    """A transform ready to time: callables take no argument."""

    forward: Callable[[], Any]
    inverse: Callable[[], Any] | None
    ncoef: int
    device: str
    config: str


# --------------------------------------------------------------------- timing


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(
    fn: Callable[[], Any],
    dev: str,
    *,
    warmup: int = 5,
    min_time: float = 1.0,
    max_iter: int = 500,
) -> tuple[float, float]:
    """Median and minimum wall time in ms, synchronizing around every call."""
    for _ in range(warmup):
        fn()
    _sync(dev)
    times: list[float] = []
    t_start = time.perf_counter()
    while len(times) < 5 or (
        time.perf_counter() - t_start < min_time and len(times) < max_iter
    ):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), min(times)


def _count(obj: Any) -> int:
    """Real-valued coefficient count of tensors/arrays (complex counts twice)."""
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * (2 if obj.is_complex() else 1)
    if isinstance(obj, np.ndarray):
        return int(obj.size) * (2 if np.iscomplexobj(obj) else 1)
    if isinstance(obj, list | tuple):
        return sum(_count(o) for o in obj)
    return 0


# ------------------------------------------------------------------ builders


def build_curvelets(
    shape: tuple[int, ...], dev: str, *, cuda_graphs: bool, triton: bool = True
) -> Transform:
    """``curvelets.torch.UDCT`` on the flat ``forward(structured=False)``/``backward`` path.

    On CUDA the folds run as fused Triton kernels when Triton is installed;
    ``triton=False`` sets ``CURVELETS_TRITON=0`` while the plan is built.
    """
    import os

    from curvelets.torch import UDCT

    previous = os.environ.get("CURVELETS_TRITON")
    os.environ["CURVELETS_TRITON"] = "1" if triton else "0"
    try:
        return _build_curvelets(UDCT, shape, dev, cuda_graphs, triton)
    finally:
        if previous is None:
            del os.environ["CURVELETS_TRITON"]
        else:
            os.environ["CURVELETS_TRITON"] = previous


def _build_curvelets(
    udct_cls: Any,
    shape: tuple[int, ...],
    dev: str,
    cuda_graphs: bool,
    triton: bool,
) -> Transform:
    UDCT = udct_cls

    udct = UDCT(
        shape=shape,
        num_scales=NUM_SCALES,
        wedges_per_direction=WEDGES_PER_DIRECTION,
        dtype=torch.float32,
        cuda_graphs=cuda_graphs,
    )
    udct.apply_to_tensors(lambda t: t.to(dev))
    x = torch.randn(shape, device=dev, dtype=torch.float32)

    def forward(t: torch.Tensor) -> torch.Tensor:
        return udct.forward(t, structured=False)

    backward = udct.backward
    v = forward(x)
    return Transform(
        forward=lambda: forward(x),
        inverse=lambda: backward(v),
        ncoef=_count(v),
        device=dev,
        config=f"num_scales={NUM_SCALES}, wedges_per_direction={WEDGES_PER_DIRECTION}"
        f", cuda_graphs={cuda_graphs}, triton={triton}",
    )


def build_curvelets_numpy(shape: tuple[int, ...], dev: str) -> Transform:
    """``curvelets.numpy.UDCT`` (CPU)."""
    from curvelets.numpy import UDCT

    del dev
    udct = UDCT(
        shape=shape,
        num_scales=NUM_SCALES,
        wedges_per_direction=WEDGES_PER_DIRECTION,
    )
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    c = udct.forward(x)
    ncoef = sum(
        w.size * (2 if np.iscomplexobj(w) else 1) for s in c for d in s for w in d
    )
    return Transform(
        forward=lambda: udct.forward(x),
        inverse=lambda: udct.backward(c),
        ncoef=ncoef,
        device="cpu",
        config=f"num_scales={NUM_SCALES}, wedges_per_direction={WEDGES_PER_DIRECTION}",
    )


def _scattering2d() -> Any:
    """``kymatio.torch.Scattering2D``, importable with recent SciPy.

    Kymatio 0.3.0 imports ``scipy.special.sph_harm`` (removed in SciPy 1.17)
    for its 3D transforms. The 2D transform does not use it, so a placeholder
    is enough to import the package.
    """
    import scipy.special

    if not hasattr(scipy.special, "sph_harm"):

        def _unavailable(*_args: Any, **_kwargs: Any) -> None:
            msg = "sph_harm is not available in this SciPy (3D scattering only)"
            raise NotImplementedError(msg)

        scipy.special.sph_harm = _unavailable  # type: ignore[attr-defined]
    from kymatio.torch import Scattering2D

    return Scattering2D


def build_kymatio(shape: tuple[int, ...], dev: str) -> Transform:
    """Kymatio's second-order 2D wavelet scattering (not invertible)."""
    Scattering2D = _scattering2d()

    scat = Scattering2D(J=NUM_SCALES, shape=shape, L=8).to(dev)
    x = torch.randn(1, 1, *shape, device=dev, dtype=torch.float32)
    out = scat(x)
    return Transform(
        forward=lambda: scat(x),
        inverse=None,
        ncoef=_count(out),
        device=dev,
        config=f"J={NUM_SCALES}, L=8, max_order=2",
    )


def build_pw_dwt(shape: tuple[int, ...], dev: str) -> Transform:
    """pytorch_wavelets' separable DWT (Daubechies 3)."""
    from pytorch_wavelets import DWTForward, DWTInverse

    xfm = DWTForward(J=NUM_SCALES, wave="db3", mode="zero").to(dev)
    ifm = DWTInverse(wave="db3", mode="zero").to(dev)
    x = torch.randn(1, 1, *shape, device=dev, dtype=torch.float32)
    coeffs = xfm(x)
    return Transform(
        forward=lambda: xfm(x),
        inverse=lambda: ifm(coeffs),
        ncoef=_count(coeffs[0]) + _count(coeffs[1]),
        device=dev,
        config=f"J={NUM_SCALES}, wave=db3",
    )


def build_pw_dtcwt(shape: tuple[int, ...], dev: str) -> Transform:
    """pytorch_wavelets' dual-tree complex wavelet transform."""
    from pytorch_wavelets import DTCWTForward, DTCWTInverse

    xfm = DTCWTForward(J=NUM_SCALES, biort="near_sym_a", qshift="qshift_a").to(dev)
    ifm = DTCWTInverse(biort="near_sym_a", qshift="qshift_a").to(dev)
    x = torch.randn(1, 1, *shape, device=dev, dtype=torch.float32)
    coeffs = xfm(x)
    return Transform(
        forward=lambda: xfm(x),
        inverse=lambda: ifm(coeffs),
        ncoef=_count(coeffs[0]) + _count(coeffs[1]),
        device=dev,
        config=f"J={NUM_SCALES}, near_sym_a, qshift_a",
    )


def build_dtcwt(shape: tuple[int, ...], dev: str) -> Transform:
    """The ``dtcwt`` package (NumPy, CPU) in 2D or 3D."""
    import dtcwt

    del dev
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    transform = dtcwt.Transform2d() if len(shape) == 2 else dtcwt.Transform3d()
    pyramid = transform.forward(x, nlevels=NUM_SCALES)
    ncoef = _count(pyramid.lowpass) + _count(list(pyramid.highpasses))
    return Transform(
        forward=lambda: transform.forward(x, nlevels=NUM_SCALES),
        inverse=lambda: transform.inverse(pyramid),
        ncoef=ncoef,
        device="cpu",
        config=f"nlevels={NUM_SCALES}, NumPy",
    )


def _check_dtcwt(shape: tuple[int, ...], lowpass: Any, highpasses: Any) -> None:
    """Raise unless a ``dtcwt`` backend agrees with the NumPy backend.

    A timing of a transform that returns different coefficients means nothing.
    """
    import dtcwt

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    cls = dtcwt.Transform2d if len(shape) == 2 else dtcwt.Transform3d
    ref = cls().forward(x, nlevels=NUM_SCALES)
    pairs = [(ref.lowpass, lowpass), *zip(ref.highpasses, highpasses, strict=True)]
    for i, (expected, got) in enumerate(pairs):
        scale = float(np.abs(expected).max())
        diff = float(np.abs(np.asarray(got) - expected).max())
        if diff > 1e-3 * scale:
            name = "lowpass" if i == 0 else f"highpass {i - 1}"
            msg = f"{name} differs from the NumPy backend by {diff:.3g} (scale {scale:.3g})"
            raise RuntimeError(msg)


def build_dtcwt_opencl(shape: tuple[int, ...], dev: str) -> Transform:
    """The ``dtcwt`` OpenCL backend (2D and 3D).

    NumPy arrays go in and come out, so the timing includes the host/device
    copies, which is how the backend is normally called.
    """
    import dtcwt.opencl
    from dtcwt.opencl.transform3d import Transform3d

    del dev
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    transform = dtcwt.opencl.Transform2d() if len(shape) == 2 else Transform3d()
    pyramid = transform.forward(x, nlevels=NUM_SCALES)
    _check_dtcwt(shape, pyramid.lowpass, pyramid.highpasses)
    ncoef = _count(pyramid.lowpass) + _count(list(pyramid.highpasses))
    return Transform(
        forward=lambda: transform.forward(x, nlevels=NUM_SCALES),
        inverse=lambda: transform.inverse(pyramid),
        ncoef=ncoef,
        device="opencl",
        config=f"nlevels={NUM_SCALES}, OpenCL backend, NumPy arrays in and out",
    )


def build_dtcwt_tf(shape: tuple[int, ...], dev: str) -> Transform:
    """The ``dtcwt`` TensorFlow backend (2D).

    The backend is written for TensorFlow 1 (``tf.Session``, ``tf.variable_scope``),
    so TensorFlow 2 runs it in graph mode through the ``tf.compat.v1`` aliases.
    The graph is built once and run with ``Session.run``, which fetches the
    outputs to the host. (Running a ``tf.group`` of the outputs instead does
    not execute the transform: it costs the same at every size.)
    """
    import tensorflow as tf

    del dev
    v1 = tf.compat.v1
    v1.disable_eager_execution()
    aliases = {
        "variable_scope": v1.variable_scope,
        "Session": v1.Session,
        "global_variables_initializer": v1.global_variables_initializer,
        "real": tf.math.real,
        "imag": tf.math.imag,
    }
    for name, function in aliases.items():
        if not hasattr(tf, name):
            setattr(tf, name, function)
    tf.nn.conv2d = v1.nn.conv2d  # the TensorFlow 1 signature
    import dtcwt.tf

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    graph = tf.Graph()
    with graph.as_default():
        variable = tf.Variable(x)
        transform = dtcwt.tf.Transform2d()
        pyramid = transform.forward(variable, nlevels=NUM_SCALES)
        forward_ops = [pyramid.lowpass_op, *pyramid.highpasses_ops]
        inverse_op = transform.inverse(pyramid)
        init = v1.global_variables_initializer()
    session = v1.Session(graph=graph)
    session.run(init)
    values = session.run(forward_ops)
    _check_dtcwt(shape, values[0], values[1:])
    ncoef = _count(values)
    gpu = bool(tf.config.list_physical_devices("GPU"))
    return Transform(
        forward=lambda: session.run(forward_ops),
        inverse=lambda: session.run(inverse_op),
        ncoef=ncoef,
        device="cuda" if gpu else "cpu",
        config=f"nlevels={NUM_SCALES}, TensorFlow backend (graph mode), outputs fetched to the host",
    )


# name -> (builder, dimensions supported, uses the accelerator)
Builder = Callable[..., Transform]
LIBRARIES: dict[str, tuple[Builder, tuple[int, ...], bool]] = {
    "curvelets (torch, CUDA graphs)": (
        lambda s, d: build_curvelets(s, d, cuda_graphs=d == "cuda"),
        (2, 3),
        True,
    ),
    "curvelets (torch, CUDA graphs, no Triton)": (
        lambda s, d: build_curvelets(s, d, cuda_graphs=d == "cuda", triton=False),
        (2, 3),
        True,
    ),
    "curvelets (torch, eager)": (
        lambda s, d: build_curvelets(s, d, cuda_graphs=False),
        (2, 3),
        True,
    ),
    "curvelets (NumPy)": (build_curvelets_numpy, (2, 3), False),
    "Kymatio Scattering2D": (build_kymatio, (2,), True),
    "pytorch_wavelets DWT": (build_pw_dwt, (2,), True),
    "pytorch_wavelets DTCWT": (build_pw_dtcwt, (2,), True),
    "dtcwt (NumPy)": (build_dtcwt, (2, 3), False),
    "dtcwt (OpenCL)": (build_dtcwt_opencl, (2, 3), True),
    "dtcwt (TensorFlow)": (build_dtcwt_tf, (2,), True),
}


def run(shapes: list[tuple[int, ...]], dev: str, only: list[str] | None) -> list[Row]:
    """Time every available library on every shape."""
    rows: list[Row] = []
    for shape in shapes:
        size = int(np.prod(shape))
        for name, (builder, dims, accel) in LIBRARIES.items():
            if len(shape) not in dims or (only and name not in only):
                continue
            where = dev if accel else "cpu"
            try:
                t = builder(shape, dev)
                fwd = bench(t.forward, t.device)
                inv = bench(t.inverse, t.device) if t.inverse else None
            except ImportError as exc:
                rows.append(
                    Row(
                        name,
                        where,
                        list(shape),
                        "forward",
                        None,
                        None,
                        None,
                        None,
                        False,
                        "",
                        f"skipped: {exc}",
                    )
                )
                print(f"{name:34s} {shape}: skipped ({exc})", flush=True)
                continue
            except Exception as exc:  # pylint: disable=broad-except
                rows.append(
                    Row(
                        name,
                        where,
                        list(shape),
                        "forward",
                        None,
                        None,
                        None,
                        None,
                        False,
                        "",
                        f"failed: {type(exc).__name__}: {str(exc)[:160]}",
                    )
                )
                print(
                    f"{name:34s} {shape}: FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            red = t.ncoef / size
            rows.append(
                Row(
                    name,
                    t.device,
                    list(shape),
                    "forward",
                    fwd[0],
                    fwd[1],
                    t.ncoef,
                    red,
                    t.inverse is not None,
                    t.config,
                )
            )
            msg = f"{name:34s} {shape!s:16s} {t.device:4s} fwd {fwd[0]:9.3f} ms"
            if inv is not None:
                rows.append(
                    Row(
                        name,
                        t.device,
                        list(shape),
                        "inverse",
                        inv[0],
                        inv[1],
                        t.ncoef,
                        red,
                        True,
                        t.config,
                    )
                )
                msg += f"  inv {inv[0]:9.3f} ms"
            print(f"{msg}  redundancy {red:.1f}x", flush=True)
            del t
            if dev == "cuda":
                torch.cuda.empty_cache()
    return rows


def meta(dev: str) -> dict[str, Any]:
    """Versions and hardware, stored with the results."""
    versions: dict[str, str | None] = {}
    for pkg in ("torch", "numpy", "curvelets", "kymatio", "pytorch_wavelets", "dtcwt"):
        try:
            versions[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "date": dt.date.today().isoformat(),
        "device": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
        "versions": versions,
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "dtype": "float32",
        "batch": 1,
        "num_scales": NUM_SCALES,
        "wedges_per_direction": WEDGES_PER_DIRECTION,
    }


def markdown(data: dict[str, Any]) -> str:
    """Forward-time table, with the speedup of curvelets (CUDA graphs)."""
    rows = [r for r in data["results"] if r["direction"] == "forward"]
    ref_name = "curvelets (torch, CUDA graphs)"
    shapes = sorted({tuple(r["shape"]) for r in rows}, key=lambda s: (len(s), s))
    lines = [
        f"Forward transform, {data['meta']['device']}, float32, batch 1 (median ms)",
        "",
        "| library | device | "
        + " | ".join("x".join(map(str, s)) for s in shapes)
        + " | redundancy (largest) |",
        "|---|---|" + "---:|" * (len(shapes) + 1),
    ]
    for name in dict.fromkeys(r["library"] for r in rows):
        mine = {tuple(r["shape"]): r for r in rows if r["library"] == name}
        ref = {tuple(r["shape"]): r for r in rows if r["library"] == ref_name}
        cells = []
        for s in shapes:
            r = mine.get(s)
            if r is None or r["median_ms"] is None:
                cells.append("-")
                continue
            cell = f"{r['median_ms']:.3g}"
            if name != ref_name and s in ref and ref[s]["median_ms"]:
                cell += f" ({r['median_ms'] / ref[s]['median_ms']:.1f}x)"
            cells.append(cell)
        last = [r for r in mine.values() if r["redundancy"]]
        dev = next((r["device"] for r in mine.values() if r["median_ms"]), "-")
        red = f"{last[-1]['redundancy']:.1f}x" if last else "-"
        lines.append(f"| {name} | {dev} | " + " | ".join(cells) + f" | {red} |")
    lines += ["", "Parentheses: time relative to curvelets (torch, CUDA graphs)."]
    return "\n".join(lines)


# ------------------------------------------------------------- atoms, stats

ATOM_CROP = 64
# Relative soft threshold (L1 proximal step) of second-order atoms: 2% of the
# largest pixel after every update. 0.05 starts to truncate the lobes.
SOFT_THRESHOLD = 0.02


def _crop(img: np.ndarray, half: int = ATOM_CROP // 2) -> np.ndarray:
    cy, cx = (n // 2 for n in img.shape)
    return img[cy - half : cy + half, cx - half : cx + half]


def kymatio_ncoef_formula(
    shape: tuple[int, ...], j: int = NUM_SCALES, n_theta: int = 8
) -> int:
    """Output size of a second-order ``Scattering2D`` (no import needed)."""
    channels = 1 + j * n_theta + n_theta * n_theta * j * (j - 1) // 2
    return channels * int(np.prod([n // 2**j for n in shape]))


def wavelet_atoms(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Image-space atoms of pytorch_wavelets' DWT and DTCWT.

    One coefficient at the center of each subband of levels 3 and 4 (the
    coarser half of ``NUM_SCALES`` levels) is set to one and the transform is
    inverted. Keys are ``"<transform>/L<level>/o<orientation>"``.
    """
    from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse

    x = torch.zeros(1, 1, *shape)
    out: dict[str, np.ndarray] = {}

    yl, yh = DWTForward(J=NUM_SCALES, wave="db3", mode="zero")(x)
    inverse = DWTInverse(wave="db3", mode="zero")
    for level in (2, 3):
        for o in range(3):
            bands = [torch.zeros_like(y) for y in yh]
            h, w = bands[level].shape[-2:]
            bands[level][0, 0, o, h // 2, w // 2] = 1.0
            atom = inverse((torch.zeros_like(yl), bands))[0, 0].numpy()
            out[f"DWT/L{level + 1}/o{o}"] = _crop(atom)

    yl, yh = DTCWTForward(J=NUM_SCALES, biort="near_sym_a", qshift="qshift_a")(x)
    inverse = DTCWTInverse(biort="near_sym_a", qshift="qshift_a")
    for level in (2, 3):
        for o in range(6):
            bands = [torch.zeros_like(y) for y in yh]
            h, w = bands[level].shape[-3:-1]
            bands[level][0, 0, o, h // 2, w // 2, 0] = 1.0  # real part
            atom = inverse((torch.zeros_like(yl), bands))[0, 0].numpy()
            out[f"DTCWT/L{level + 1}/o{o}"] = _crop(atom)
    return out


def _progress(done: int, total: int, label: str) -> None:
    """One-line progress bar on stderr (no dependency)."""
    width = 30
    filled = width * done // total
    end = "\n" if done == total else ""
    print(
        f"\r[{'#' * filled}{'.' * (width - filled)}] {done}/{total} {label:<40s}",
        end=end,
        file=sys.stderr,
        flush=True,
    )


def _coefficient(scat: Any, x: torch.Tensor, channel: int) -> torch.Tensor:
    """One scattering coefficient at the center of its output."""
    y = scat(x)
    return y[0, 0, channel, y.shape[-2] // 2, y.shape[-1] // 2]


def kymatio_atoms(
    shape: tuple[int, ...], steps: int = 20, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Images that excite one scattering coefficient the most.

    The scattering transform is not invertible (it takes a modulus), so a
    coefficient has no synthesis atom. The closest equivalent is the unit-norm
    image ``x`` that maximizes that coefficient at the center of its output:

        maximize  S(x)[k, center]   subject to  ||x|| = 1

    solved from a random start by a fixed-point iteration. ``S`` is positively
    homogeneous (``S(c x) = c S(x)``), so the maximizer over the sphere of its
    linearization at ``x`` is ``grad S(x) / ||grad S(x)||``, and the update is

        x  <-  grad S(x) / ||grad S(x)||

    with no step size. Linearizing the modulus this way never decreases a
    first-order coefficient (it is convex in ``x``) and a few steps suffice.
    Second-order coefficients are not convex: the full jump overshoots and the
    coefficient bounces, so the update is damped,
    ``x <- normalize((1 - a) x + a grad / ||grad||)`` with ``a = 0.2``, which
    converges smoothly in 10-15 steps (and reached a coefficient about 50%
    higher than the full jump on the paths tried). An L1 proximal step (soft-threshold at 2% of the
    largest pixel) removes faint sidelobes at about 3% of the coefficient. The best
    iterate is kept. The gradient comes from PyTorch autograd; it is the adjoint of the
    linearized transform, so no explicit linear operator (for instance from
    PyLops) is needed. First-order paths use a ``max_order=1`` transform (33
    channels instead of 417). Keys are
    ``"Kymatio/order1/j<j>/o<theta>"`` and
    ``"Kymatio/order2/j<j1><j2>/o<theta1><theta2>"`` (scales 2 and 3: at finer scales
    the carrier oscillates every few pixels and the atom looks like stripes).

    Returns the atoms and, under the same keys, the convergence curves: the
    value of the coefficient before each step and after the last one
    (``steps + 1`` values).
    """
    scattering = _scattering2d()
    scats = {
        1: scattering(J=NUM_SCALES, shape=shape, L=8, max_order=1),
        2: scattering(J=NUM_SCALES, shape=shape, L=8, max_order=2),
    }
    j_max, n_theta = NUM_SCALES, 8
    # Output channel of every path, in the order Kymatio emits them: the
    # lowpass, then for each first-order path (j1, theta1) its own coefficient
    # followed by the second-order paths (j2 > j1, theta2) that extend it.
    index: dict[tuple[tuple[int, int], ...], int] = {(): 0}
    channel = 1
    for j1 in range(j_max):
        for t1 in range(n_theta):
            index[((j1, t1),)] = channel
            channel += 1
            for j2 in range(j1 + 1, j_max):
                for t2 in range(n_theta):
                    index[((j1, t1), (j2, t2))] = channel
                    channel += 1
    targets: dict[str, tuple[tuple[int, int], ...]] = {
        f"Kymatio/order1/j{j}/o{t}": ((j, t),) for j in (1, 2) for t in range(8)
    }
    targets.update({
        f"Kymatio/order2/j23/o{t1}{t2}": ((2, t1), (3, t2))
        for t1 in (0, 2)
        for t2 in range(8)
    })
    out: dict[str, np.ndarray] = {}
    curves: dict[str, np.ndarray] = {}
    total = len(targets) * steps
    done = 0
    for key, path in targets.items():
        values: list[float] = []
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 1, *shape, generator=gen)
        x = (x / x.norm()).requires_grad_()
        scat = scats[len(path)]
        # a max_order=1 transform lists first-order paths without the second-order
        # ones in between, so its channel is a plain offset
        channel = (
            1 + path[0][0] * n_theta + path[0][1] if len(path) == 1 else index[path]
        )
        # the full jump is exact for convex (first-order) coefficients; second-order
        # ones overshoot and bounce, and a damped step converges smoothly
        alpha = 1.0 if len(path) == 1 else 0.2
        best_value, best_x = -1.0, x.detach()
        for _ in range(steps + 1):
            value = _coefficient(scat, x, channel)
            values.append(float(value))
            if float(value) > best_value:  # second-order paths need not increase
                best_value, best_x = float(value), x.detach()
            if len(values) > steps:
                break
            (grad,) = torch.autograd.grad(value, x)
            with torch.no_grad():
                x = (1 - alpha) * x + alpha * grad / grad.norm()
                if len(path) == 2:  # L1: soft-threshold the sidelobes away
                    x = x.sign() * (x.abs() - SOFT_THRESHOLD * x.abs().max()).clamp(
                        min=0
                    )
                x = (x / x.norm()).requires_grad_()
            done += 1
            _progress(done, total, key)
        x = best_x
        curves[key] = np.array(values, dtype=np.float32)
        # the output pixel (center) sits at 2**J * (n // 2**J // 2) in the image
        half = 2**j_max * (shape[0] // 2**j_max // 2)
        img = x.detach()[0, 0].numpy()
        out[key] = np.roll(img, shape[0] // 2 - half, axis=(0, 1))[
            shape[0] // 2 - ATOM_CROP // 2 : shape[0] // 2 + ATOM_CROP // 2,
            shape[1] // 2 - ATOM_CROP // 2 : shape[1] // 2 + ATOM_CROP // 2,
        ]
    return out, curves


def register_stats(path: Path, size: int = 256, steps: int = 20) -> None:
    """Write atoms and redundancy of every transform that can be built here."""
    shape = (size, size)
    stats: dict[str, Any] = {"size": size, "redundancy": {}, "ncoef": {}, "notes": {}}
    for name, (builder, dims, _) in LIBRARIES.items():
        if (
            2 not in dims
            or "eager" in name
            or "no Triton" in name
            or ("NumPy" in name and "curvelets" in name)
        ):
            continue
        try:
            ncoef = builder(shape, "cpu").ncoef
        except Exception as exc:  # pylint: disable=broad-except
            if "Kymatio" in name:
                ncoef = kymatio_ncoef_formula(shape)
                stats["notes"][name] = (
                    f"from the output-size formula ({type(exc).__name__} on import)"
                )
            else:
                stats["notes"][name] = (
                    f"unavailable: {type(exc).__name__}: {str(exc)[:100]}"
                )
                continue
        stats["ncoef"][name] = ncoef
        stats["redundancy"][name] = ncoef / size**2
    atoms = wavelet_atoms(shape)
    curves: dict[str, np.ndarray] = {}
    stats["kymatio_steps"] = steps
    try:
        kymatio, curves = kymatio_atoms(shape, steps=steps)
        atoms.update(kymatio)
    except Exception as exc:  # pylint: disable=broad-except
        stats["notes"]["Kymatio atoms"] = (
            f"unavailable: {type(exc).__name__}: {str(exc)[:100]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"stats": json.dumps(stats)}
    payload.update({k: v.astype(np.float32) for k, v in atoms.items()})
    payload.update({f"curve/{k}": v for k, v in curves.items()})
    np.savez_compressed(path, **payload)
    print(json.dumps(stats, indent=1))
    print(f"wrote {path} ({len(atoms)} atoms)")


SPEED_SHAPES: list[tuple[int, ...]] = [(256, 256), (512, 512), (64, 64, 64)]
TORCH_LABEL = "Curvelets (torch)"
NUMPY_LABEL = "Curvelets (NumPy)"
_MODES = {
    "curvelets (torch, CUDA graphs)": "CUDA graphs",
    "curvelets (torch, CUDA graphs, no Triton)": "CUDA graphs, no Triton",
    "curvelets (torch, eager)": "eager",
}


def _label(library: str) -> str:
    if library == "curvelets (NumPy)":
        return NUMPY_LABEL
    return TORCH_LABEL if library.startswith("curvelets") else library


def _esc(row: list[str]) -> list[str]:
    """Escape the asterisk that marks a footnote in the library name."""
    return [row[0].replace("*", "\\*"), *row[1:]]


def _table(header: list[str], body: list[list[str]], fmt: str, text_cols: int) -> str:
    """A table as MyST/GitHub Markdown or as an RST ``list-table``.

    The first ``text_cols`` columns are left-aligned, the others right-aligned.
    """
    if fmt == "md":
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "---|" * text_cols + "---:|" * (len(header) - text_cols))
        lines += ["| " + " | ".join(_esc(row)) + " |" for row in body]
        return "\n".join(lines)
    lines = [".. list-table::", "   :header-rows: 1", ""]
    for raw in [header, *body]:
        row = _esc(raw)
        lines.append("   * - " + row[0])
        lines += [f"     - {cell}" for cell in row[1:]]
    return "\n".join(lines)


def speed_table(data: dict[str, Any], fmt: str) -> str:
    """Forward-transform speed, as multiples of the slowest library per column.

    The torch implementation of curvelets appears once, with its best mode in
    each column. The slowest entry of a column is 1.00x and the fastest is bold.
    """
    # label -> shape -> (median ms, device, library row that gave it)
    best: dict[str, dict[tuple[int, ...], tuple[float, str, str]]] = {}
    for r in data["results"]:
        shape = tuple(r["shape"])
        if r["direction"] != "forward" or r["median_ms"] is None:
            continue
        if shape not in SPEED_SHAPES:
            continue
        cell = best.setdefault(_label(r["library"]), {})
        if shape not in cell or r["median_ms"] < cell[shape][0]:
            cell[shape] = (r["median_ms"], r["device"], r["library"])
    order = [
        TORCH_LABEL,
        NUMPY_LABEL,
        "Kymatio Scattering2D",
        "pytorch_wavelets DWT",
        "pytorch_wavelets DTCWT",
        "dtcwt (NumPy)",
        "dtcwt (OpenCL)",
        "dtcwt (TensorFlow)",
    ]
    order += [n for n in best if n not in order]
    labels = sorted(
        best,
        key=lambda n: (
            n not in (TORCH_LABEL, NUMPY_LABEL),
            n != TORCH_LABEL,
            order.index(n),
        ),
    )
    slowest = {
        s: max((t[s][0], n) for n, t in best.items() if s in t) for s in SPEED_SHAPES
    }
    header = ["Library", "Device"] + ["x".join(map(str, s)) for s in SPEED_SHAPES]
    body = []
    for name in labels:
        devices = sorted({d for _, d, _ in best[name].values()})
        row = [name, "/".join(devices)]
        for s in SPEED_SHAPES:
            if s not in best[name]:
                row.append("-")
                continue
            factor = slowest[s][0] / best[name][s][0]
            fastest = factor == max(
                slowest[s][0] / t[s][0] for t in best.values() if s in t
            )
            text = f"{factor:.0f}x"
            row.append(f"**{text}**" if fastest else text)
        body.append(row)
    winners = {
        _MODES.get(best[TORCH_LABEL][s][2], best[TORCH_LABEL][s][2])
        for s in SPEED_SHAPES
        if s in best.get(TORCH_LABEL, {})
    }
    if winners == {"CUDA graphs"}:
        text = (
            "Best mode of the torch implementation in each column; "
            "it was CUDA graphs (``cuda_graphs=True``) in all of them."
        )
    else:
        text = (
            f"Best mode of the torch implementation in each column: {sorted(winners)}."
        )
    if fmt == "md":
        text = text.replace("``", "`")
        footnote = f"[^cudagraphs]: {text}"
    else:
        footnote = f".. [#cudagraphs] {text}"
    mark = "[^cudagraphs]" if fmt == "md" else " [#cudagraphs]_"
    for row in body:
        if row[0] == TORCH_LABEL:
            row[0] += mark
    return _table(header, body, fmt, 2) + "\n\n" + footnote


def speed_reference(data: dict[str, Any]) -> str:
    """The slowest library of each speed column, which is the 1x reference."""
    slowest: dict[tuple[int, ...], tuple[float, str]] = {}
    for r in data["results"]:
        shape = tuple(r["shape"])
        if r["direction"] != "forward" or r["median_ms"] is None:
            continue
        if shape in SPEED_SHAPES and (
            shape not in slowest or r["median_ms"] > slowest[shape][0]
        ):
            slowest[shape] = (r["median_ms"], _label(r["library"]))
    by_library: dict[str, list[str]] = {}
    for shape in SPEED_SHAPES:
        if shape in slowest:
            ms, name = slowest[shape]
            by_library.setdefault(name, []).append(
                f"{'x'.join(map(str, shape))} ({ms:.3g} ms)"
            )
    parts = []
    for name, cells in by_library.items():
        parts.append(f"{name} at {' and '.join(cells)}")
    return ", and ".join(parts) if len(parts) < 3 else "; ".join(parts)


def redundancy_table(data: dict[str, Any], fmt: str) -> str:
    """Coefficients per input sample of each transform."""
    shapes = [SPEED_SHAPES[0], SPEED_SHAPES[-1]]
    found: dict[str, dict[tuple[int, ...], float]] = {}
    for r in data["results"]:
        shape = tuple(r["shape"])
        if r["direction"] != "forward" or r["redundancy"] is None:
            continue
        if shape in shapes:
            name = r["library"]
            for prefix, short in (("curvelets", "Curvelets"), ("dtcwt", "dtcwt")):
                if name.startswith(prefix):
                    name = short
            found.setdefault(name, {})[shape] = r["redundancy"]
    header = ["Library"] + ["x".join(map(str, s)) for s in shapes]
    body = [
        [name] + [f"{vals[s]:.2f}x" if s in vals else "-" for s in shapes]
        for name, vals in found.items()
    ]
    return _table(header, body, fmt, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--export", type=Path, help="write results as JSON")
    p.add_argument("--markdown", type=Path, help="print a table from a JSON file")
    p.add_argument(
        "--table",
        choices=["speed", "redundancy", "speed-ref"],
        help="print a documentation table from --data",
    )
    p.add_argument("--format", choices=["rst", "md"], default="rst")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "library_comparison.json",
    )
    p.add_argument("--stats", type=Path, help="register atoms and redundancy (npz)")
    p.add_argument(
        "--steps", type=int, default=20, help="optimizer steps per Kymatio atom"
    )
    p.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    p.add_argument("--sizes-2d", type=int, nargs="*", default=DEFAULT_SHAPES_2D)
    p.add_argument("--sizes-3d", type=int, nargs="*", default=DEFAULT_SHAPES_3D)
    p.add_argument("--only", nargs="*", help="library names to run")
    args = p.parse_args()

    if args.stats:
        register_stats(args.stats, steps=args.steps)
        return

    if args.table:
        data = json.loads(args.data.read_text())
        if args.table == "speed-ref":
            print(speed_reference(data))
        else:
            table = speed_table if args.table == "speed" else redundancy_table
            print(table(data, args.format))
        return

    if args.markdown:
        print(markdown(json.loads(args.markdown.read_text())))
        return

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    shapes: list[tuple[int, ...]] = [(n, n) for n in args.sizes_2d]
    shapes += [(n, n, n) for n in args.sizes_3d]
    rows = run(shapes, dev, args.only)
    data = {"meta": meta(dev), "results": [asdict(r) for r in rows]}
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {args.export}")
    print()
    print(markdown(data))


if __name__ == "__main__":
    main()
