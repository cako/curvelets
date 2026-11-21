"""
Like-for-like CPU benchmark of the NumPy UDCT, across versions and FFT engines.

Runs against whatever ``curvelets`` is importable, so the same script measures
``main`` (``--engines default``: no ``fft_engine`` argument) and a later branch
(``--engines numpy,scipy,fftw,mkl_fft,auto``). For each case it prints one JSON
line with the best-of-N wall time of ``forward`` and of ``backward``.

Usage::

    python benchmarks/numpy_engines_bench.py --label main --engines default
    python benchmarks/numpy_engines_bench.py --label pr --engines numpy,scipy,auto \\
        --workers 32

Results from a 32-vCPU AMD EPYC 9575F are in
``benchmarks/data/numpy_engines_epyc9575f.jsonl`` (one process per engine).
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from curvelets.numpy import UDCT

CASES: list[tuple[str, tuple[int, ...], list[list[int]]]] = [
    ("real", (512, 512), [[3, 3], [6, 6], [12, 12]]),
    ("real", (2048, 2048), [[3, 3], [6, 6], [12, 12], [24, 24]]),
    ("real", (128, 128, 128), [[3, 3, 3], [6, 6, 6]]),
    ("real", (256, 256, 256), [[3, 3, 3], [6, 6, 6]]),
    ("complex", (1024, 1024), [[3, 3], [6, 6], [12, 12]]),
    ("monogenic", (1024, 1024), [[3, 3], [6, 6], [12, 12]]),
]


def best_ms(fn: Callable[[], Any], budget_s: float = 8.0, max_runs: int = 9) -> float:
    """Best wall time in ms after one warm-up (at least 3 runs, ~budget_s)."""
    fn()
    times: list[float] = []
    start = time.perf_counter()
    while len(times) < max_runs and (
        len(times) < 3 or time.perf_counter() - start < budget_s
    ):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--engines",
        default="default",
        help="Comma-separated fft_engine values; 'default' passes none (main).",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="fft_workers (default: all)."
    )
    args = parser.parse_args()

    for kind, shape, wedges in CASES:
        for dtype in (np.float64, np.float32):
            rng = np.random.default_rng(0)
            x: np.ndarray = rng.standard_normal(shape).astype(dtype)
            if kind == "complex":
                x = x + 1j * rng.standard_normal(shape).astype(dtype)
            for engine in args.engines.split(","):
                kwargs: dict[str, Any] = {}
                if engine != "default":
                    kwargs = {"fft_engine": engine, "fft_workers": args.workers}
                udct = UDCT(
                    shape=shape,
                    angular_wedges_config=np.array(wedges),
                    transform_kind=kind,  # type: ignore[arg-type]
                    **kwargs,
                )
                coeffs = udct.forward(x)
                row = {
                    "label": args.label,
                    "kind": kind,
                    "shape": "x".join(map(str, shape)),
                    "dtype": np.dtype(dtype).name,
                    "engine": engine,
                    "resolved": getattr(udct, "fft_engine", "numpy"),
                    "workers": args.workers,
                    "fwd_ms": round(best_ms(functools.partial(udct.forward, x)), 2),
                    "bwd_ms": round(
                        best_ms(functools.partial(udct.backward, coeffs)), 2
                    ),
                }
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
