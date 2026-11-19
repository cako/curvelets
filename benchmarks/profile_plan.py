"""Per-kernel GPU time of the library forward/backward (torch.profiler / CUPTI).

Usage: python benchmarks/profile_plan.py [--shape 2048,2048] [--scales 5] [--wpd 6]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from curvelets.torch import UDCT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="2048,2048")
    ap.add_argument("--scales", type=int, default=5)
    ap.add_argument("--wpd", type=int, default=6)
    ap.add_argument("--kind", default="real")
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    udct = UDCT(
        shape=shape,
        num_scales=args.scales,
        wedges_per_direction=args.wpd,
        transform_kind=args.kind,
    )
    udct.apply_to_tensors(lambda t: t.to(dev))
    x = torch.randn(shape, device=dev)
    if args.kind == "complex":
        x = torch.complex(x, torch.randn(shape, device=dev))
    vec = udct._forward_vect(x)
    for _ in range(3):
        udct._backward_vect(udct._forward_vect(x))
    acts = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if dev.type == "cuda" else []
    )
    sort = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
    calls: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("forward", lambda: udct._forward_vect(x)),
        ("backward", lambda: udct._backward_vect(vec)),
    ]
    for name, fn in calls:
        with profile(activities=acts) as prof:
            for _ in range(10):
                fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
        print(f"\n=== {name} {shape} (10 calls)")
        print(prof.key_averages().table(sort_by=sort, row_limit=args.rows))


if __name__ == "__main__":
    main()
