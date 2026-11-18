"""Benchmark the fused Triton fold against the torch gather fold (CUDA only).

For each configuration and batch size, on the same plan:

- ``fold``: the forward and backward folds alone (device ms, back to back);
- ``e2e``: ``UDCT._forward_vect`` / ``UDCT._backward_vect`` (wall ms with a sync
  per call, and device ms back to back), plain and with ``cuda_graphs=True``;
- peak extra memory of one end-to-end call, and the size of the fold tables.

``torch`` rows clear the plan's CSR maps, so both variants use the same plan,
windows and FFTs; only the fold implementation differs.

Usage: python benchmarks/fold_bench.py [--configs 2d_512,...] [--batches 1,8]
       [--export out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from functools import partial
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


def timeit(fn: Callable[[], object], iters: int = 30) -> tuple[float, float]:
    """(median wall ms with a sync per call, mean device ms back to back)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    walls.sort()
    # Queue a long sleep first so the events time the device, not the launches.
    torch.cuda._sleep(int(5e7))  # pylint: disable=protected-access
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return walls[len(walls) // 2], a.elapsed_time(b) / iters


def peak_mib(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() - base) / 2**20


def mib(tensors: list[torch.Tensor]) -> float:
    return float(sum(t.numel() * t.element_size() for t in tensors)) / 2**20


def level_tensors(fold: Any) -> list[torch.Tensor]:
    out = [fold.src0, fold.w0]
    for lvl in fold.levels:
        out += [lvl.targets, lvl.sources, lvl.weights]
    return out


def run_config(name: str, batches: list[int], graphs: bool) -> list[dict[str, Any]]:
    shape, ns, wpd = CONFIGS[name]
    udct = UDCT(
        shape=shape, num_scales=ns, wedges_per_direction=wpd, dtype=torch.float32
    )
    udct.apply_to_tensors(lambda t: t.cuda())
    plan = udct._get_plan()  # pylint: disable=protected-access
    fwd: Any = plan.fwd_map  # GatherFold with CSR maps on CUDA
    bwd: Any = plan.bwd_map
    csr = (fwd.csr, bwd.csr)
    assert csr[0] is not None, "Triton kernel unavailable"
    assert csr[1] is not None, "Triton kernel unavailable"
    # Plain calls keep only the CSR maps; the torch rows need the level tables,
    # which are built here (as a differentiable call would).
    tables = {
        "levels_mib": mib(level_tensors(fwd) + level_tensors(bwd)),
        "csr_mib": mib(csr[0].tensors() + csr[1].tensors()),
    }
    rows = []
    for batch in batches:
        x = torch.randn(batch, *shape, device="cuda")
        coeffs = udct._forward_vect(x)  # pylint: disable=protected-access
        spectrum = plan._spectrum(x)  # pylint: disable=protected-access
        small = plan._batched_ffts(coeffs, torch.fft.fftn, coeffs.dtype)  # pylint: disable=protected-access
        for variant in ("torch", "triton"):
            fwd.csr, bwd.csr = csr if variant == "triton" else (None, None)
            rec: dict[str, Any] = {"config": name, "batch": batch, "variant": variant}
            rec["fold_fwd_dev"] = timeit(partial(fwd.apply, spectrum))[1]
            rec["fold_bwd_dev"] = timeit(partial(bwd.apply, small))[1]
            calls: dict[str, Callable[[], object]] = {
                "fwd": partial(udct._forward_vect, x),  # pylint: disable=protected-access
                "inv": partial(udct._backward_vect, coeffs),  # pylint: disable=protected-access
            }
            for key, fn in calls.items():
                udct.cuda_graphs = False
                rec[f"{key}_wall"], rec[f"{key}_dev"] = timeit(fn)
                rec[f"{key}_peak_mib"] = peak_mib(fn)
                if graphs:
                    udct.cuda_graphs = True
                    rec[f"{key}_graph_wall"], _ = timeit(fn)
                    udct.cuda_graphs = False
            rec.update(tables)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        fwd.csr, bwd.csr = csr
        # Same results either way.
        ref = udct._forward_vect(x)  # pylint: disable=protected-access
        fwd.csr = bwd.csr = None
        err = ((ref - udct._forward_vect(x)).abs().max() / ref.abs().max()).item()  # pylint: disable=protected-access
        fwd.csr, bwd.csr = csr
        print(f"# {name} B={batch}: max rel diff triton vs torch {err:.1e}", flush=True)
    del udct, plan
    torch.cuda.empty_cache()
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    """Markdown: torch -> triton per config/batch."""
    out = [
        "| config | B | fold fwd dev | fold inv dev | fwd wall | inv wall | fwd dev | inv dev | fwd graphs | inv graphs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keyed = {(r["config"], r["batch"], r["variant"]): r for r in rows}
    for (cfg, b, v), r in keyed.items():
        if v != "torch":
            continue
        t = keyed[(cfg, b, "triton")]
        cells = []
        for k in (
            "fold_fwd_dev",
            "fold_bwd_dev",
            "fwd_wall",
            "inv_wall",
            "fwd_dev",
            "inv_dev",
            "fwd_graph_wall",
            "inv_graph_wall",
        ):
            if k in r:
                cells.append(
                    f"{r[k]:.3f} → **{t[k]:.3f}** ({(t[k] / r[k] - 1) * 100:+.0f}%)"
                )
            else:
                cells.append("—")
        out.append(f"| {cfg} | {b} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--export")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for name in args.configs.split(","):
        rows += run_config(
            name, [int(b) for b in args.batches.split(",")], not args.no_graphs
        )
    print(table(rows))
    if args.export:
        meta = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        }
        with Path(args.export).open("w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
