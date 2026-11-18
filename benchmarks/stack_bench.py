"""Like-for-like benchmark of the public torch UDCT across library versions.

Runs against whatever ``curvelets`` is importable, so the same script measures
``main`` and a later branch on one GPU. It uses only API that both have:

- ``fwd``: ``udct.vect(udct.forward(x))``;
- ``bwd``: ``udct.backward(coeffs)`` from nested coefficients;

for one image (float32 input, real transform, wall ms with a sync per call).
Where the checked-out version supports them (detected, not assumed), it also
times a batch of 8 natively and with ``cuda_graphs=True``; otherwise a batch of
8 is a Python loop over images, which is what that version offers.

Usage: python benchmarks/stack_bench.py --label main --export main.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from curvelets.torch import UDCT

CONFIGS = {
    "2d_512": ((512, 512), 3, 3),
    "2d_1024_w6": ((1024, 1024), 4, 6),
    "2d_2048_w6": ((2048, 2048), 5, 6),
    "3d_64": ((64, 64, 64), 3, 3),
    "3d_128": ((128, 128, 128), 3, 3),
}

_PARAMS = inspect.signature(UDCT.__init__).parameters
HAS_DTYPE = "dtype" in _PARAMS
HAS_GRAPHS = "cuda_graphs" in _PARAMS


def wall_ms(
    fn: Callable[[], object], min_time: float = 1.0, max_iter: int = 200
) -> float:
    """Median wall ms with a sync per call."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = time.perf_counter()
    while len(times) < max_iter and (
        time.perf_counter() - start < min_time or len(times) < 5
    ):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def build(shape: tuple[int, ...], ns: int, wpd: int, graphs: bool = False) -> UDCT:
    kwargs: dict[str, Any] = {
        "shape": shape,
        "num_scales": ns,
        "wedges_per_direction": wpd,
    }
    if HAS_DTYPE:
        kwargs["dtype"] = torch.float32
    if graphs:
        kwargs["cuda_graphs"] = True
    udct = UDCT(**kwargs)
    udct.apply_to_tensors(lambda t: t.cuda())
    return udct


def batched_ok(udct: UDCT, shape: tuple[int, ...]) -> bool:
    try:
        out = udct.vect(udct.forward(torch.zeros(2, *shape, device="cuda")))
    except Exception:  # older versions reject batch dimensions
        return False
    return bool(out.shape[0] == 2)


def run(name: str) -> dict[str, Any]:
    shape, ns, wpd = CONFIGS[name]
    t0 = time.perf_counter()
    udct = build(shape, ns, wpd)
    rec: dict[str, Any] = {"config": name, "init_s": time.perf_counter() - t0}
    x = torch.randn(*shape, device="cuda")
    coeffs = udct.forward(x)
    rec["fwd"] = wall_ms(lambda: udct.vect(udct.forward(x)))
    rec["bwd"] = wall_ms(lambda: udct.backward(coeffs))
    xs = torch.randn(8, *shape, device="cuda")
    if batched_ok(udct, shape):
        cb = udct.forward(xs)
        rec["fwd_b8"] = wall_ms(lambda: udct.vect(udct.forward(xs)))
        rec["bwd_b8"] = wall_ms(lambda: udct.backward(cb))
        rec["b8_mode"] = "native"
    else:
        cl = [udct.forward(xi) for xi in xs]
        rec["fwd_b8"] = wall_ms(lambda: [udct.vect(udct.forward(xi)) for xi in xs])
        rec["bwd_b8"] = wall_ms(lambda: [udct.backward(c) for c in cl])
        rec["b8_mode"] = "loop"
    if HAS_GRAPHS:
        g = build(shape, ns, wpd, graphs=True)
        cg = g.forward(x)
        rec["fwd_graphs"] = wall_ms(lambda: g.vect(g.forward(x)))
        rec["bwd_graphs"] = wall_ms(lambda: g.backward(cg))
    print(json.dumps(rec), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--export")
    args = ap.parse_args()
    rows = []
    for name in args.configs.split(","):
        rows.append(run(name))
        torch.cuda.empty_cache()
    if args.export:
        meta = {
            "label": args.label,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        }
        with Path(args.export).open("w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
