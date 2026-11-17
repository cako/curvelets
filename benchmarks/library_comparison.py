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
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import statistics
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
        return obj.numel() * (2 if obj.is_complex() else 1)
    if isinstance(obj, np.ndarray):
        return obj.size * (2 if np.iscomplexobj(obj) else 1)
    if isinstance(obj, (list, tuple)):
        return sum(_count(o) for o in obj)
    return 0


# ------------------------------------------------------------------ builders


def build_curvelets(
    shape: tuple[int, ...], dev: str, *, cuda_graphs: bool
) -> Transform:
    """``curvelets.torch.UDCT`` on the flat path that ``UDCTModule`` uses."""
    from curvelets.torch import UDCT

    udct = UDCT(
        shape=shape,
        num_scales=NUM_SCALES,
        wedges_per_direction=WEDGES_PER_DIRECTION,
        dtype=torch.float32,
        cuda_graphs=cuda_graphs,
    )
    udct.apply_to_tensors(lambda t: t.to(dev))
    x = torch.randn(shape, device=dev, dtype=torch.float32)
    forward = udct._forward_vect  # pylint: disable=protected-access
    backward = udct._backward_vect  # pylint: disable=protected-access
    v = forward(x)
    return Transform(
        forward=lambda: forward(x),
        inverse=lambda: backward(v),
        ncoef=_count(v),
        device=dev,
        config=f"num_scales={NUM_SCALES}, wedges_per_direction={WEDGES_PER_DIRECTION}"
        f", cuda_graphs={cuda_graphs}",
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


def build_kymatio(shape: tuple[int, ...], dev: str) -> Transform:
    """Kymatio's second-order 2D wavelet scattering (not invertible)."""
    from kymatio.torch import Scattering2D

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


# name -> (builder, dimensions supported, uses the accelerator)
Builder = Callable[..., Transform]
LIBRARIES: dict[str, tuple[Builder, tuple[int, ...], bool]] = {
    "curvelets (torch, CUDA graphs)": (
        lambda s, d: build_curvelets(s, d, cuda_graphs=d == "cuda"),
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
    versions = {}
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--export", type=Path, help="write results as JSON")
    p.add_argument("--markdown", type=Path, help="print a table from a JSON file")
    p.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    p.add_argument("--sizes-2d", type=int, nargs="*", default=DEFAULT_SHAPES_2D)
    p.add_argument("--sizes-3d", type=int, nargs="*", default=DEFAULT_SHAPES_3D)
    p.add_argument("--only", nargs="*", help="library names to run")
    args = p.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.markdown.read_text())))
        return

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    shapes = [(n, n) for n in args.sizes_2d] + [(n, n, n) for n in args.sizes_3d]
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
