"""torch.compile / CUDA graph / batch chunking variants for the fused UDCT.

Includes UDCT(cuda_graphs=True).

Usage: python benchmarks/compile_experiments.py [--configs 2d_512,3d_128] [--batch 1,8]
       [--modes default,reduce-overhead,max-autotune-no-cudagraphs,max-autotune]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from curvelets.torch import UDCT

CONFIGS = {
    "2d_512": ((512, 512), 3, 3),
    "2d_2048_w6": ((2048, 2048), 5, 6),
    "3d_128": ((128, 128, 128), 3, 3),
}


def timeit(fn: Callable[[], object], iters: int = 20) -> tuple[float, float]:
    """(wall ms with a sync per call, device ms back to back)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / iters
    torch.cuda._sleep(int(5e7))  # pylint: disable=protected-access
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return wall, a.elapsed_time(b) / iters


def graphed(
    fn: Callable[[torch.Tensor], object], inp: torch.Tensor
) -> Callable[[], object]:
    """Capture fn(static) in a CUDA graph; each call copies inp in and replays."""
    static = inp.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static)

    def run() -> object:
        static.copy_(inp)
        graph.replay()
        return out

    return run


def rowloop(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """Process the batch one row at a time (L2-sized working set)."""

    def run(x: torch.Tensor) -> list[torch.Tensor]:
        return [fn(x[i : i + 1]) for i in range(x.shape[0])]

    return run


def rel_err(a: Any, b: torch.Tensor) -> float:
    if isinstance(a, list):
        a = torch.cat(a)
    return float((a - b).abs().max() / b.abs().max())


def report(
    cfg: str,
    batch: int,
    direction: str,
    name: str,
    fn: Callable[[], object],
    ref: torch.Tensor,
    compile_s: float | None = None,
) -> None:
    try:
        out = fn()
        err = rel_err(out, ref)
        wall, dev = timeit(fn)
        extra = f"  compile {compile_s:6.1f}s" if compile_s is not None else ""
        print(
            f"{cfg:11s} B={batch} {direction:3s} {name:30s} wall {wall:8.3f}  dev {dev:8.3f}  err {err:.1e}{extra}",
            flush=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        msg = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
        print(
            f"{cfg:11s} B={batch} {direction:3s} {name:30s} FAILED {type(exc).__name__}: {msg}",
            flush=True,
        )


def compiled(
    fn: Callable[[torch.Tensor], torch.Tensor], mode: str, arg: torch.Tensor
) -> tuple[Callable[[], torch.Tensor], float]:
    torch._dynamo.reset()  # pylint: disable=protected-access
    kw = {} if mode == "default" else {"mode": mode}
    cfn = torch.compile(fn, **kw)
    cudagraphs = mode in ("reduce-overhead", "max-autotune")
    t0 = time.perf_counter()

    def call() -> torch.Tensor:
        if cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        return cfn(arg)

    call()
    torch.cuda.synchronize()
    return call, time.perf_counter() - t0


def graph_breaks(fn: Callable[[torch.Tensor], torch.Tensor], arg: torch.Tensor) -> str:
    torch._dynamo.reset()  # pylint: disable=protected-access
    try:
        ex = torch._dynamo.explain(fn)(arg)  # pylint: disable=protected-access
        reasons = sorted({
            str(r.reason).splitlines()[0][:110] for r in ex.break_reasons
        })
        return f"{ex.graph_count} graphs, {ex.graph_break_count} breaks: {reasons}"
    except Exception as exc:  # pylint: disable=broad-except
        return f"explain failed: {type(exc).__name__}: {str(exc)[:120]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="2d_512,2d_2048_w6,3d_128")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument(
        "--modes",
        default="default,reduce-overhead,max-autotune-no-cudagraphs,max-autotune",
    )
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    dev = torch.device("cuda")
    for cfg in args.configs.split(","):
        shape, ns, wpd = CONFIGS[cfg]
        udct = UDCT(shape=shape, num_scales=ns, wedges_per_direction=wpd)
        udct.apply_to_tensors(lambda t: t.to(dev))
        x1 = torch.randn((1, *shape), device=dev)
        udct.forward(x1, structured=False)
        plan = udct._get_plan()  # pylint: disable=protected-access
        fwd, bwd = plan.forward_flat, plan.backward_flat
        for batch in (int(b) for b in args.batch.split(",")):
            x = torch.randn((batch, *shape), device=dev)
            with torch.no_grad():
                v = fwd(x)
                ref_f, ref_b = v.clone(), bwd(v).clone()
                for direction, fn, arg, ref in (
                    ("fwd", fwd, x, ref_f),
                    ("inv", bwd, v, ref_b),
                ):
                    report(
                        cfg,
                        batch,
                        direction,
                        "eager",
                        partial(fn, arg),
                        ref,
                    )
                    report(
                        cfg,
                        batch,
                        direction,
                        "cuda graph (manual)",
                        graphed(fn, arg),
                        ref,
                    )
                    if batch > 1:
                        report(
                            cfg,
                            batch,
                            direction,
                            "eager, row loop",
                            partial(rowloop(fn), arg),
                            ref,
                        )
                        report(
                            cfg,
                            batch,
                            direction,
                            "cuda graph, row loop",
                            graphed(rowloop(fn), arg),
                            ref,
                        )
                    if "compile" in skip:
                        continue
                    if batch == 1:
                        print(
                            f"{cfg:11s} B={batch} {direction:3s} dynamo: {graph_breaks(fn, arg)}",
                            flush=True,
                        )
                    for mode in args.modes.split(","):
                        try:
                            call, cs = compiled(fn, mode, arg)
                        except Exception as exc:  # pylint: disable=broad-except
                            print(
                                f"{cfg:11s} B={batch} {direction:3s} compile[{mode}] FAILED {type(exc).__name__}: {str(exc)[:200]}",
                                flush=True,
                            )
                            continue
                        report(cfg, batch, direction, f"compile[{mode}]", call, ref, cs)
                graphed_udct = UDCT(
                    shape=shape,
                    num_scales=ns,
                    wedges_per_direction=wpd,
                    cuda_graphs=True,
                )
                graphed_udct.apply_to_tensors(lambda t: t.to(dev))
                fwd_g = lambda t: graphed_udct.forward(t, structured=False)
                bwd_g = graphed_udct.backward
                report(
                    cfg,
                    batch,
                    "fwd",
                    "UDCT(cuda_graphs=True)",
                    partial(fwd_g, x),
                    ref_f,
                )
                report(
                    cfg,
                    batch,
                    "inv",
                    "UDCT(cuda_graphs=True)",
                    partial(bwd_g, v),
                    ref_b,
                )
                del graphed_udct
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
