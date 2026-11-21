"""Per-stage GPU timing of the fused UDCT forward/backward (CUDA events).

Replays ``UDCTPlan.forward_flat`` / ``backward_flat`` (gather strategy, i.e. the
accelerator path) one stage at a time with the plan's own index maps, and puts a
CUDA event between stages. Each stage reports:

- ``gpu``: device time between its events. A long ``torch.cuda._sleep`` is queued
  first so the CPU runs ahead and the GPU never waits for a launch: this is pure
  device execution time.
- ``host``: CPU time to enqueue the stage (Python + dispatcher + launch).
- ``GB/s``: a lower bound on traffic (each input/index/weight read once, each
  output written once) over ``gpu``, i.e. the fraction of the roofline reached.

It also reports the uninstrumented end-to-end time (device and host) and the
kernel count, and checks that the replay matches the library output.
The replay runs the whole batch at once; the library splits large batches into
L2-sized row chunks (``UDCTPlan.batch_chunk_bytes``), so for those the
end-to-end figures are those of the chunked library call.

Usage: python benchmarks/stage_profile.py [--configs 2d_512,3d_128] [--batch 1]
       [--dtype float32] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.profiler import ProfilerActivity, profile

from curvelets.torch import UDCT
from curvelets.torch._foldmap import GatherFold

CONFIGS = {
    "mnist_28": ((28, 28), 2, 3),
    "2d_512": ((512, 512), 3, 3),
    "2d_1024_w6": ((1024, 1024), 4, 6),
    "2d_2048_w6": ((2048, 2048), 5, 6),
    "3d_64": ((64, 64, 64), 3, 3),
    "3d_128": ((128, 128, 128), 3, 3),
}

CSIZE = {torch.complex64: 8, torch.complex128: 16}


@dataclass
class Timeline:
    """Records (name, event, host time, min bytes) markers on the current stream."""

    names: list[str] = field(default_factory=list)
    events: list[torch.cuda.Event] = field(default_factory=list)
    host: list[float] = field(default_factory=list)
    nbytes: list[float] = field(default_factory=list)
    _t: float = 0.0

    def start(self) -> None:
        # Keep the GPU busy for ~20 ms so every launch below is queued before
        # the GPU reaches it: events then measure device time only.
        torch.cuda._sleep(int(2e7))  # pylint: disable=protected-access
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append(ev)
        self._t = time.perf_counter()

    def mark(self, name: str, nbytes: float = 0.0) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        now = time.perf_counter()
        self.names.append(name)
        self.events.append(ev)
        self.host.append((now - self._t) * 1e3)
        self.nbytes.append(nbytes)
        self._t = now

    def result(self) -> list[tuple[str, float, float, float]]:
        torch.cuda.synchronize()
        return [
            (n, self.events[i].elapsed_time(self.events[i + 1]), h, b)
            for i, (n, h, b) in enumerate(
                zip(self.names, self.host, self.nbytes, strict=True)
            )
        ]


def fold_bytes(fold: GatherFold, csize: int, rows: int) -> dict[str, float]:
    """Minimum traffic per fold level: index + weight(s) + gathered source + output."""
    wsize = (csize // 2) * (2 if fold.w0.ndim == 2 else 1)
    out: dict[str, float] = {"L0": fold.num_targets * (4 + wsize + rows * 2 * csize)}
    for i, lvl in enumerate(fold.levels, start=1):
        n = lvl.sources.numel()
        t = lvl.targets.numel()
        out[f"L{i}"] = n * (4 + wsize + rows * csize) + t * (4 + rows * 2 * csize)
    return out


def timed_fold(
    tl: Timeline, fold: GatherFold, src: torch.Tensor, tag: str
) -> torch.Tensor:
    """GatherFold.apply, one marker per level (gather*weight, then index_add)."""
    parts = torch.view_as_real(src)
    nb = fold_bytes(fold, CSIZE[src.dtype], src.shape[0])

    def weighted(index: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        return parts.index_select(1, index) * weights

    out = weighted(fold.src0, fold.w0)
    tl.mark(f"{tag} L0 gather (dense {fold.num_targets:,})", nb["L0"])
    for slot, lvl in enumerate(fold.levels, start=1):
        term = weighted(lvl.sources, lvl.weights)
        if lvl.width > 1:
            term = term.unflatten(1, (-1, lvl.width)).sum(2)
        out.index_add_(1, lvl.targets, term)
        kind = f"tail w={lvl.width}" if lvl.width > 1 else "sparse"
        tl.mark(f"{tag} L{slot} {kind} ({lvl.targets.numel():,} tgt)", nb[f"L{slot}"])
    return torch.view_as_complex(out)


def staged_forward(plan, image: torch.Tensor, tl: Timeline) -> torch.Tensor:
    dims = plan._dims()  # pylint: disable=protected-access
    rows = math.prod(image.shape[: image.ndim - plan.ndim])
    n = math.prod(plan.shape)
    nspec = math.prod(plan.spectrum_shape)
    rs = image.element_size()
    cs = 2 * rs
    tl.start()
    if plan.complex_output:
        spectrum = torch.fft.fftn(image, dim=dims)
        tl.mark(f"fftn full {plan.shape}", rows * n * 2 * cs)
    else:
        spectrum = torch.fft.rfftn(image, dim=dims)
        tl.mark(f"rfftn full {plan.shape}", rows * (n * rs + nspec * cs))
    flat = spectrum.reshape(-1, nspec)
    folded = timed_fold(tl, plan.fwd_map, flat, "fwd fold")
    # Like UDCTPlan._batched_ffts without autograd: each group writes into one buffer.
    coeffs = torch.empty_like(folded)
    for g in plan.groups:
        chunk = folded[:, g.offset : g.stop].view(rows, g.count, *g.shape)
        dst = coeffs[:, g.offset : g.stop].view(rows, g.count, *g.shape)
        torch.fft.ifftn(chunk, dim=dims, norm="forward", out=dst)
        tl.mark(f"ifftn {g.count}x{g.shape}", rows * g.count * g.size * 2 * cs)
    return coeffs


def staged_backward(plan, vector: torch.Tensor, tl: Timeline) -> torch.Tensor:
    dims = plan._dims()  # pylint: disable=protected-access
    rows = vector.shape[0]
    cs = vector.element_size()
    n = math.prod(plan.shape)
    nspec = math.prod(plan.spectrum_shape)
    tl.start()
    small = torch.empty_like(vector)
    for g in plan.groups:
        chunk = vector[:, g.offset : g.stop].view(rows, g.count, *g.shape)
        dst = small[:, g.offset : g.stop].view(rows, g.count, *g.shape)
        torch.fft.fftn(chunk, dim=dims, out=dst)
        tl.mark(f"fftn {g.count}x{g.shape}", rows * g.count * g.size * 2 * cs)
    spectrum = timed_fold(tl, plan.bwd_map, small, "bwd fold")
    spectrum = spectrum.reshape(rows, *plan.spectrum_shape)
    if plan.complex_output:
        out = torch.fft.ifftn(spectrum, dim=dims)
        tl.mark(f"ifftn full {plan.shape}", rows * n * 2 * cs)
    else:
        out = torch.fft.irfftn(spectrum, s=plan.shape, dim=dims)
        tl.mark(f"irfftn full {plan.shape}", rows * (nspec * cs + n * cs // 2))
    return out


def e2e(fn: Callable[[], object], iters: int) -> tuple[float, float]:
    """(device ms, host enqueue ms) per call, uninstrumented."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda._sleep(int(5e7))  # pylint: disable=protected-access
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host = (time.perf_counter() - t0) * 1e3 / iters
    b.record()
    torch.cuda.synchronize()
    # Wall-clock with a sync per call (what a user sees for one call).
    return a.elapsed_time(b) / iters, host


def wall(fn: Callable[[], object], iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def kernel_count(fn: Callable[[], object]) -> int:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def median_stages(runs: list[list[tuple[str, float, float, float]]]):
    out = []
    for i, (name, _, _, nb) in enumerate(runs[0]):
        g = sorted(r[i][1] for r in runs)[len(runs) // 2]
        h = sorted(r[i][2] for r in runs)[len(runs) // 2]
        out.append((name, g, h, nb))
    return out


def print_table(title: str, stages, total_dev: float, total_host: float) -> None:
    s = sum(g for _, g, _, _ in stages)
    print(f"\n  -- {title}: sum of stages {s:.3f} ms (device)")
    print(f"  {'stage':48s} {'gpu ms':>8s} {'%':>6s} {'host ms':>8s} {'GB/s':>7s}")
    for name, g, h, nb in stages:
        bw = nb / (g * 1e6) if g > 0 and nb else float("nan")
        print(f"  {name[:48]:48s} {g:8.3f} {100 * g / s:5.1f}% {h:8.3f} {bw:7.0f}")
    print(f"  {'e2e (uninstrumented)':48s} {total_dev:8.3f}        {total_host:8.3f}")


def run(
    name: str,
    batch: int,
    dtype: torch.dtype,
    kind: Literal["real", "complex"],
    iters: int,
) -> dict[str, Any]:
    shape, ns, wpd = CONFIGS[name]
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    udct = UDCT(
        shape=shape, num_scales=ns, wedges_per_direction=wpd, transform_kind=kind
    )
    udct.apply_to_tensors(lambda t: t.to(dev))
    init_s = time.perf_counter() - t0
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((batch, *shape), generator=gen, dtype=dtype).to(dev)
    if kind == "complex":
        x = torch.complex(x, torch.randn_like(x))
    t0 = time.perf_counter()
    vec = udct.forward_vect(x)
    torch.cuda.synchronize()
    plan_s = time.perf_counter() - t0
    plan = udct._get_plan()  # pylint: disable=protected-access

    # correctness of the replay
    ref_f = udct.forward_vect(x).reshape(batch, -1)
    tl = Timeline()
    got_f = staged_forward(plan, x, tl)
    tl.result()
    ref_b = udct.backward_vect(vec)
    tl = Timeline()
    got_b = staged_backward(plan, vec.reshape(batch, -1), tl)
    tl.result()
    err_f = ((got_f - ref_f).abs().max() / ref_f.abs().max()).item()
    err_b = (
        (got_b.reshape(ref_b.shape) - ref_b).abs().max() / ref_b.abs().max()
    ).item()

    fwd_runs, bwd_runs = [], []
    for _ in range(iters):
        tl = Timeline()
        staged_forward(plan, x, tl)
        fwd_runs.append(tl.result())
        tl = Timeline()
        staged_backward(plan, vec.reshape(batch, -1), tl)
        bwd_runs.append(tl.result())
    fwd = median_stages(fwd_runs)
    bwd = median_stages(bwd_runs)

    f_dev, f_host = e2e(lambda: udct.forward_vect(x), 20)
    b_dev, b_host = e2e(lambda: udct.backward_vect(vec), 20)
    f_wall = wall(lambda: udct.forward_vect(x), 20)
    b_wall = wall(lambda: udct.backward_vect(vec), 20)
    nested_wall = wall(lambda: udct.forward(x), 20)
    nested_bwd = udct.forward(x)
    nested_bwd_wall = wall(lambda: udct.backward(nested_bwd), 20)

    # Flat path with autograd (training-style use)
    mod = UDCT(
        shape=shape, num_scales=ns, wedges_per_direction=wpd, transform_kind=kind
    ).to(dev)
    xg = x.clone().requires_grad_(True)

    def train_step() -> None:
        mod.forward_vect(xg).abs().sum().backward()

    mod_fwd_wall = wall(lambda: mod.forward_vect(x), 20)
    train_wall = wall(train_step, 20)

    nw = len(plan.layout)
    print(
        f"\n=== {name} {shape} ns={ns} wpd={wpd} {kind} {dtype} batch={batch}: "
        f"{nw} wedges, {len(plan.groups)} groups, ncoef {plan.ncoef:,} "
        f"({plan.ncoef / math.prod(shape):.2f}x redundancy)"
    )
    print(
        f"  init {init_s:.2f}s, plan build {plan_s:.2f}s, replay err fwd {err_f:.1e} bwd {err_b:.1e}"
    )
    fm = plan.fwd_map
    assert isinstance(fm, GatherFold)  # CUDA plans use gather folds
    print(
        f"  fwd fold: {fm.num_targets:,} targets, levels "
        f"{[(lv.targets.numel(), lv.width) for lv in fm.levels]}"
    )
    print(
        f"  kernels/call: fwd {kernel_count(lambda: udct.forward_vect(x))}, "
        f"bwd {kernel_count(lambda: udct.backward_vect(vec))}"
    )
    print_table("FORWARD", fwd, f_dev, f_host)
    print_table("INVERSE", bwd, b_dev, b_host)
    print(
        f"\n  wall (sync per call): fwd_vect {f_wall:.3f}  bwd_vect {b_wall:.3f}  "
        f"forward(nested) {nested_wall:.3f}  backward(nested) {nested_bwd_wall:.3f}  "
        f"forward_vect {mod_fwd_wall:.3f}  fwd+autograd bwd {train_wall:.3f} ms"
    )
    return {
        "name": name,
        "shape": list(shape),
        "num_scales": ns,
        "wpd": wpd,
        "kind": kind,
        "dtype": str(dtype),
        "batch": batch,
        "n_wedges": nw,
        "groups": [[g.count, list(g.shape)] for g in plan.groups],
        "ncoef": plan.ncoef,
        "init_s": init_s,
        "plan_s": plan_s,
        "err": [err_f, err_b],
        "forward": [list(s) for s in fwd],
        "inverse": [list(s) for s in bwd],
        "e2e": {
            "fwd_dev": f_dev,
            "fwd_host": f_host,
            "bwd_dev": b_dev,
            "bwd_host": b_host,
            "fwd_wall": f_wall,
            "bwd_wall": b_wall,
            "forward_nested_wall": nested_wall,
            "backward_nested_wall": nested_bwd_wall,
            "module_fwd_wall": mod_fwd_wall,
            "module_train_wall": train_wall,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--batch", default="1")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--kind", default="real", choices=["real", "complex"])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--json")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
    results = []
    for name in args.configs.split(","):
        for b in (int(v) for v in args.batch.split(",")):
            kind = cast(Literal["real", "complex"], args.kind)
            results.append(run(name, b, dtype, kind, args.iters))
            torch.cuda.empty_cache()
    if args.json:
        with Path(args.json).open("w", encoding="utf-8") as fh:
            json.dump(
                {"device": torch.cuda.get_device_name(), "results": results},
                fh,
                indent=1,
            )


if __name__ == "__main__":
    main()
