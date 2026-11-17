"""GPU performance assessment for curvelets.torch.UDCT (real transform).

Compares, per configuration:

- ``library``   : the checked-out library code path (nested-list API) (``UDCT`` + ``apply_to_tensors(cuda)``, i.e. what
                  ``UDCTModule.to("cuda")`` does), forward = ``vect(forward(x))``,
                  backward = ``backward(c)``. (Reported as ``base`` in older results.)
- ``fix_maps``  : same, but re-attach the periodized index maps that
                  ``apply_to_tensors`` drops.
- ``fix_sync``  : ``fix_maps`` + decimation ratios kept on the CPU (removes the per-wedge
                  ``.item()`` device syncs used to compute scale factors).
                  ``fix_maps``/``fix_sync`` only differ from ``library`` before the P0
                  fixes; select variants with ``--variants``.
- ``fused``     : prototype that flattens every wedge into one gather, one ``index_add_``
                  and one batched FFT per run of equal-shape wedges; also supports a
                  leading batch dimension.
- ``fused_graph``: ``fused`` forward captured in a CUDA graph.

Also reports numpy/torch CPU timings, kernel launch and host-sync counts
(torch.profiler), peak memory, and window storage size.

Usage: python benchmarks/gpu_bench.py [--quick] [--export out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from curvelets.torch import UDCT

# ----------------------------------------------------------------------------- utils


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench(
    fn,
    dev: torch.device,
    *,
    warmup: int = 3,
    min_time: float = 0.5,
    max_iter: int = 200,
) -> float:
    """Median wall time in ms (with device sync around each call)."""
    for _ in range(warmup):
        fn()
    sync(dev)
    times: list[float] = []
    t_start = time.perf_counter()
    while len(times) < 5 or (
        time.perf_counter() - t_start < min_time and len(times) < max_iter
    ):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def profile_counts(fn, dev: torch.device) -> dict[str, int]:
    """Kernel launches and host<->device syncs for one call."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    sync(dev)
    acts = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if dev.type == "cuda" else []
    )
    with profile(activities=acts) as prof:
        fn()
        sync(dev)
    kernels = 0
    syncs = 0
    memcpy = 0
    for e in prof.events():
        name = e.name
        if name == "aten::_local_scalar_dense":
            syncs += 1
        dt = getattr(e, "device_type", None)
        if dt is not None and str(dt).endswith("CUDA"):
            if "Memcpy" in name or "memcpy" in name:
                if "DtoH" in name:
                    memcpy += 1
            elif "Memset" not in name:
                kernels += 1
    return {"kernels": kernels, "item_calls": syncs, "dtoh_syncs": memcpy}


def windows_bytes(udct: UDCT) -> int:
    total = 0
    for s in udct.windows:
        for d in s:
            for w in d:
                for t in (
                    w.indices,
                    w.values,
                    w.folded_indices,
                    w.flipped_indices,
                    w.flipped_folded_indices,
                ):
                    if t is not None:
                        total += t.numel() * t.element_size()
    return total


def decimation_for(udct: UDCT, s: int, d: int) -> torch.Tensor:
    dr = udct.decimation_ratios[s]
    return dr[0] if dr.shape[0] == 1 else dr[min(d, dr.shape[0] - 1)]


# ------------------------------------------------------------------ baseline variants


def move_base(udct: UDCT, dev: torch.device) -> UDCT:
    udct.apply_to_tensors(lambda t: t.to(dev))
    return udct


def move_fix_maps(udct: UDCT, dev: torch.device, *, keep_dec_on_cpu: bool) -> UDCT:
    dec_cpu = [d.clone() for d in udct.decimation_ratios]
    udct.apply_to_tensors(lambda t: t.to(dev))
    for s, sw in enumerate(udct.windows):
        for d, dw in enumerate(sw):
            dr = dec_cpu[s]
            dec = dr[0] if dr.shape[0] == 1 else dr[min(d, dr.shape[0] - 1)]
            for w in dw:
                w.attach_periodized([int(x) for x in dec.tolist()])
    if keep_dec_on_cpu:
        for i, d in enumerate(dec_cpu):
            udct._decimation_ratios[i] = d
    return udct


# --------------------------------------------------------------- fused prototype


@dataclass
class Group:
    offset: int  # into flat coefficient buffer
    count: int  # number of wedges in this run
    shape: tuple[int, ...]
    fwd_scale: torch.Tensor  # (count,) real
    bwd_scale: torch.Tensor  # (count,) real


class FusedUDCT:
    """Real-mode, curvelet-mode UDCT with all wedges fused into a few kernels.

    Coefficient layout equals ``UDCT.vect`` order, so outputs are comparable.
    Accepts a leading batch dimension: x of shape (B, *shape) -> (B, ncoef).
    """

    def __init__(self, udct: UDCT, dev: torch.device, real_dtype: torch.dtype):
        p = udct._parameters
        self.shape = tuple(p.shape)
        self.ndim = len(self.shape)
        self.N = math.prod(self.shape)
        ns = p.num_scales
        assert len(udct.windows[ns - 1]) > 1, (
            "prototype covers curvelet high-frequency mode"
        )
        scale_norm = math.sqrt(float(2 ** (ns - 2)) ** self.ndim)

        idx: list[torch.Tensor] = []
        gidx: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        # runs of consecutive wedges sharing a decimated shape:
        # [offset, count, shape, fwd scales, bwd scales]
        runs: list[tuple[int, int, tuple[int, ...], list[float], list[float]]] = []
        offset = 0
        entries = []  # (window, fwd_scale, bwd_scale)
        w0 = udct.windows[0][0][0]
        pd0 = float(np.prod(decimation_for(udct, 0, 0).cpu().numpy()))
        entries.append((w0, scale_norm / pd0, math.sqrt(pd0)))
        for s in range(1, ns):
            for d, dw in enumerate(udct.windows[s]):
                pd = float(np.prod(decimation_for(udct, s, d).cpu().numpy()))
                for w in dw:
                    entries.append((w, math.sqrt(2 * pd) / pd, 2 * math.sqrt(pd / 2)))
        for w, fs, bs in entries:
            assert w.folded_indices is not None
            assert w.out_shape is not None
            shp = tuple(w.out_shape)
            P = math.prod(shp)
            if runs and runs[-1][2] == shp:
                o, n, _, f_list, b_list = runs[-1]
                runs[-1] = (o, n + 1, shp, [*f_list, fs], [*b_list, bs])
            else:
                runs.append((offset, 1, shp, [fs], [bs]))
            gidx.append(w.folded_indices.cpu() + offset)
            idx.append(w.indices.cpu())
            vals.append(w.values.cpu())
            offset += P
        self.ncoef = offset
        cdtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        self.cdtype = cdtype
        self.idx = torch.cat(idx).to(dev)
        self.gidx = torch.cat(gidx).to(dev)
        self.vals = torch.cat(vals).to(dev, real_dtype)
        self.groups = [
            Group(
                o,
                n,
                shp,
                torch.tensor(f_list, dtype=real_dtype, device=dev),
                torch.tensor(b_list, dtype=real_dtype, device=dev),
            )
            for o, n, shp, f_list, b_list in runs
        ]
        self.dims = tuple(range(-self.ndim, 0))
        self.nnz = self.idx.numel()

    def _bshape(self, g: Group) -> tuple[int, ...]:
        return (g.count,) + (1,) * self.ndim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        F = torch.fft.fftn(x, dim=self.dims).reshape(B, self.N)
        v = F[:, self.idx] * self.vals
        buf = torch.zeros(B, self.ncoef, dtype=self.cdtype, device=x.device)
        buf.index_add_(1, self.gidx, v)
        out = torch.empty_like(buf)
        for g in self.groups:
            P = math.prod(g.shape)
            src = buf[:, g.offset : g.offset + g.count * P].view(B, g.count, *g.shape)
            dst = out[:, g.offset : g.offset + g.count * P].view(B, g.count, *g.shape)
            torch.fft.ifftn(src, dim=self.dims, out=dst)
            dst.mul_(g.fwd_scale.view(self._bshape(g)))
        return out

    def backward(self, c: torch.Tensor) -> torch.Tensor:
        B = c.shape[0]
        small = torch.empty_like(c)
        for g in self.groups:
            P = math.prod(g.shape)
            src = c[:, g.offset : g.offset + g.count * P].view(B, g.count, *g.shape)
            dst = small[:, g.offset : g.offset + g.count * P].view(B, g.count, *g.shape)
            torch.fft.fftn(src, dim=self.dims, out=dst)
            dst.mul_(g.bwd_scale.view(self._bshape(g)))
        v = small[:, self.gidx] * self.vals
        F = torch.zeros(B, self.N, dtype=self.cdtype, device=c.device)
        F.index_add_(1, self.idx, v)
        return torch.fft.ifftn(F.view(B, *self.shape), dim=self.dims).real


# ------------------------------------------------------------------------- driver

CONFIGS_FULL = [
    # name, shape, num_scales, wedges_per_direction, batch sizes for throughput
    ("mnist_28x28", (28, 28), 2, 3, [1, 64, 512]),
    ("2d_512", (512, 512), 3, 3, [1, 16]),
    ("2d_1024_w6", (1024, 1024), 4, 6, [1, 8]),
    ("2d_2048_w6", (2048, 2048), 5, 6, [1, 4]),
    ("3d_64", (64, 64, 64), 3, 3, [1, 8]),
    ("3d_128", (128, 128, 128), 3, 3, [1, 2]),
]
CONFIGS_QUICK = [
    ("mnist_28x28", (28, 28), 2, 3, [1, 64]),
    ("2d_256", (256, 256), 3, 3, [1, 4]),
    ("3d_32", (32, 32, 32), 3, 3, [1, 2]),
]


def run_config(
    name: str,
    shape: tuple[int, ...],
    ns: int,
    wpd: int,
    batches: list[int],
    dev: torch.device,
    dtype: torch.dtype,
    *,
    cpu_ref: bool,
    variant_names: tuple[str, ...] = ("library",),
) -> dict[str, Any]:
    res: dict[str, Any] = {
        "name": name,
        "shape": list(shape),
        "num_scales": ns,
        "wedges_per_direction": wpd,
        "dtype": str(dtype),
    }
    t0 = time.perf_counter()
    ref = UDCT(shape=shape, num_scales=ns, wedges_per_direction=wpd, dtype=dtype)
    res["init_s"] = time.perf_counter() - t0
    nwedges = sum(len(d) for s in ref.windows for d in s)
    res["n_wedges"] = nwedges
    res["window_bytes"] = windows_bytes(ref)
    gen = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(shape, generator=gen, dtype=dtype)
    v_ref = ref.vect(ref.forward(x_cpu))
    res["ncoef"] = v_ref.numel()
    res["redundancy"] = v_ref.numel() / x_cpu.numel()
    print(
        f"\n=== {name} {shape} ns={ns} wpd={wpd} {dtype}: {nwedges} wedges, init {res['init_s']:.2f}s, "
        f"windows {res['window_bytes'] / 2**20:.1f} MiB, ncoef {res['ncoef']}",
        flush=True,
    )

    if cpu_ref:
        from curvelets.numpy import UDCT as NPUDCT

        t0 = time.perf_counter()
        nu = NPUDCT(shape=shape, num_scales=ns, wedges_per_direction=wpd)
        res["numpy_init_s"] = time.perf_counter() - t0
        xn = x_cpu.numpy()
        cn = nu.forward(xn)
        res["numpy_cpu_fwd_ms"] = bench(
            lambda: nu.forward(xn), torch.device("cpu"), min_time=1.0, max_iter=20
        )
        res["numpy_cpu_bwd_ms"] = bench(
            lambda: nu.backward(cn), torch.device("cpu"), min_time=1.0, max_iter=20
        )
        cref = ref.forward(x_cpu)
        res["torch_cpu_fwd_ms"] = bench(
            lambda: ref.forward(x_cpu), torch.device("cpu"), min_time=1.0, max_iter=20
        )
        res["torch_cpu_bwd_ms"] = bench(
            lambda: ref.backward(cref), torch.device("cpu"), min_time=1.0, max_iter=20
        )
        print(
            f"  cpu: numpy fwd {res['numpy_cpu_fwd_ms']:.2f} bwd {res['numpy_cpu_bwd_ms']:.2f} | "
            f"torch fwd {res['torch_cpu_fwd_ms']:.2f} bwd {res['torch_cpu_bwd_ms']:.2f} ms",
            flush=True,
        )

    x = x_cpu.to(dev)
    # Best non-fused variant, reused below for the loop / vmap batch comparisons.
    u_keep = copy.deepcopy(ref)
    move_fix_maps(u_keep, dev, keep_dec_on_cpu=True)
    variants: dict[str, dict[str, Any]] = {}
    for vname in variant_names:
        u = copy.deepcopy(ref)
        if vname in ("library", "library_vect"):
            move_base(u, dev)
        else:
            move_fix_maps(u, dev, keep_dec_on_cpu=vname == "fix_sync")
        c = u.forward(x)
        v = u.vect(c)
        err = (v.cpu() - v_ref).abs().max().item() / v_ref.abs().max().item()
        if vname == "library_vect":
            # Flat path used by UDCTModule (no nested-list round trip).
            fwd = lambda u=u: u._forward_vect(x)  # noqa: E731
            bwd = lambda u=u, v=v: u._backward_vect(v)  # noqa: E731
        else:
            fwd = lambda u=u: u.vect(u.forward(x))  # noqa: E731
            # NB: backward is fed the nested coefficients directly; before the P0
            # fixes UDCT.struct() crashed on CUDA after apply_to_tensors.
            bwd = lambda u=u, c=c: u.backward(c)  # noqa: E731
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = {
            "fwd_ms": bench(fwd, dev),
            "bwd_ms": bench(bwd, dev),
            "rel_err_vs_cpu": err,
            "fwd_counts": profile_counts(fwd, dev),
            "bwd_counts": profile_counts(bwd, dev),
        }
        if dev.type == "cuda":
            r["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
        variants[vname] = r
        print(
            f"  {vname:9s} fwd {r['fwd_ms']:8.3f} ms  bwd {r['bwd_ms']:8.3f} ms  "
            f"fwd kernels {r['fwd_counts']['kernels']:5d} dtoh {r['fwd_counts']['dtoh_syncs']:4d}  "
            f"bwd kernels {r['bwd_counts']['kernels']:5d} dtoh {r['bwd_counts']['dtoh_syncs']:4d}  err {err:.1e}",
            flush=True,
        )
        del u

    # Does backprop through UDCTModule work on this device?
    try:
        from curvelets.torch import UDCTModule

        m = UDCTModule(
            shape=shape, num_scales=ns, wedges_per_direction=wpd, dtype=dtype
        ).to(dev)
        xg = x.clone().requires_grad_(True)
        m(xg).abs().sum().backward()
        res["udctmodule_autograd"] = "ok"
    except Exception as e:
        res["udctmodule_autograd"] = repr(e)[:200]
    print(f"  UDCTModule autograd on {dev}: {res['udctmodule_autograd']}", flush=True)

    # fused prototype
    fu = FusedUDCT(ref, dev, dtype)
    xb = x[None]
    vf = fu.forward(xb)
    err_f = (vf[0].cpu() - v_ref).abs().max().item() / v_ref.abs().max().item()
    xr = fu.backward(vf)
    err_rt = (xr[0] - x).abs().max().item() / x.abs().max().item()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    r = {
        "fwd_ms": bench(lambda: fu.forward(xb), dev),
        "bwd_ms": bench(lambda: fu.backward(vf), dev),
        "rel_err_vs_cpu": err_f,
        "roundtrip_err": err_rt,
        "fwd_counts": profile_counts(lambda: fu.forward(xb), dev),
        "bwd_counts": profile_counts(lambda: fu.backward(vf), dev),
        "n_groups": len(fu.groups),
        "nnz": fu.nnz,
    }
    if dev.type == "cuda":
        r["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
    variants["fused"] = r
    print(
        f"  {'fused':9s} fwd {r['fwd_ms']:8.3f} ms  bwd {r['bwd_ms']:8.3f} ms  "
        f"fwd kernels {r['fwd_counts']['kernels']:5d} dtoh {r['fwd_counts']['dtoh_syncs']:4d}  "
        f"bwd kernels {r['bwd_counts']['kernels']:5d}  err {err_f:.1e} rt {err_rt:.1e}  groups {len(fu.groups)}",
        flush=True,
    )

    # Bandwidth estimate for fused forward: bytes moved by the dominant ops
    csz = torch.empty(0, dtype=fu.cdtype).element_size()
    rsz = torch.empty(0, dtype=dtype).element_size()
    fft_bytes = 2 * fu.N * csz * 2  # fft read+write (~2 passes)
    gather_bytes = fu.nnz * (
        8 + 8 + rsz + csz * 2
    )  # idx, gidx, vals, gathered val, atomics
    small_fft = 2 * fu.ncoef * csz * 2
    r["est_fwd_bytes"] = fft_bytes + gather_bytes + small_fft

    # CUDA graph of fused forward
    if dev.type == "cuda":
        try:
            static_x = xb.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fu.forward(static_x)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = fu.forward(static_x)
            graph.replay()
            torch.cuda.synchronize()
            gerr = (static_out - vf).abs().max().item()
            variants["fused_graph"] = {
                "fwd_ms": bench(graph.replay, dev),
                "err_vs_fused": gerr,
            }
            print(
                f"  {'graph':9s} fwd {variants['fused_graph']['fwd_ms']:8.3f} ms",
                flush=True,
            )
        except Exception as e:
            variants["fused_graph"] = {"error": repr(e)[:300]}
            print(f"  graph failed: {e!r}"[:300], flush=True)

    # batched throughput (per-image ms)
    thr: dict[int, dict[str, float | None]] = {}
    for B in batches:
        if B == 1:
            continue
        xb_b = torch.randn((B, *shape), generator=gen, dtype=dtype).to(dev)
        try:
            t_fused = bench(lambda xb_b=xb_b: fu.forward(xb_b), dev) / B
            vb = fu.forward(xb_b)
            t_fused_b = bench(lambda vb=vb: fu.backward(vb), dev) / B
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        # current code: python loop over the batch (fix_sync is the best non-fused variant)
        t_loop_base = (
            bench(
                lambda xb_b=xb_b: [u_keep.vect(u_keep.forward(xi)) for xi in xb_b],
                dev,
                max_iter=20,
            )
            / B
        )
        # vmap via UDCTModule-style path
        try:
            vm = torch.vmap(lambda xi: u_keep.vect(u_keep.forward(xi)))
            vm(xb_b)
            t_vmap = bench(lambda vm=vm, xb_b=xb_b: vm(xb_b), dev, max_iter=20) / B
        except Exception as e:
            t_vmap = None
            print(f"  vmap failed: {e!r}"[:200])
        thr[B] = {
            "fused_fwd_ms_per_img": t_fused,
            "fused_bwd_ms_per_img": t_fused_b,
            "loop_fix_sync_fwd_ms_per_img": t_loop_base,
            "vmap_fix_sync_fwd_ms_per_img": t_vmap,
        }
        print(
            f"  batch {B:4d}: fused fwd {t_fused:.4f} bwd {t_fused_b:.4f} ms/img | loop {t_loop_base:.4f} | "
            f"vmap {t_vmap if t_vmap is None else round(t_vmap, 4)} ms/img",
            flush=True,
        )
        del xb_b
    res["batched"] = thr
    res["variants"] = variants
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--export", default=None)
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--no-cpu-ref", action="store_true")
    ap.add_argument(
        "--variants",
        default="library,library_vect",
        help="comma-separated subset of library,library_vect,fix_maps,fix_sync",
    )
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name()
        if dev.type == "cuda"
        else platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
    }
    print(meta, flush=True)
    configs = CONFIGS_QUICK if args.quick else CONFIGS_FULL
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c[0] in keep]
    results = []
    for dt_name in args.dtypes.split(","):
        dtype = getattr(torch, dt_name)
        for name, shape, ns, wpd, batches in configs:
            try:
                results.append(
                    run_config(
                        name,
                        shape,
                        ns,
                        wpd,
                        batches,
                        dev,
                        dtype,
                        cpu_ref=not args.no_cpu_ref and dt_name == "float32",
                        variant_names=tuple(args.variants.split(",")),
                    )
                )
            except Exception as e:
                print(f"!! {name} {dt_name} failed: {e!r}"[:400], flush=True)
                results.append({"name": name, "dtype": dt_name, "error": repr(e)[:400]})
            if args.export:
                with Path(args.export).open("w") as f:
                    json.dump(
                        {"meta": meta, "results": results}, f, indent=1, default=str
                    )


if __name__ == "__main__":
    main()
