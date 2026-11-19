# GPU performance assessment — `curvelets.torch`

Measured 2026-11-15 on an **NVIDIA RTX 2000 Ada** (16 GB, ~224 GB/s), torch
2.8.0+cu128. Raw data: [`data/gpu.json`](data/gpu.json). Reproduce on a machine
with a CUDA GPU:

```bash
python benchmarks/gpu_bench.py --export benchmarks/data/gpu.json
```

## TL;DR: top of the stack vs `main`

float32, `UDCTModule` forward + backward, single image.

| config | GPU (RTX 2000 Ada) | CPU (4 threads) |
|---|---:|---:|
| 512² | **11x** faster | **2.1x** faster |
| 1024², 85 wedges | **30x** faster | **1.3x** faster |
| 64³ | **67x** faster | **3.1x** faster |
| 128³ | **16x** faster | **1.5x** faster |

CPU raw data: [`data/cpu_vs_main.json`](data/cpu_vs_main.json) (forward + backward through autograd, min over repeated runs).

## Status after the GPU stack

The roadmap below has been implemented through P2, plus native batching (PRs
#82, #83, #84, #85, #86). Measured on the same RTX 2000 Ada as the baseline,
float32, `UDCTModule` flat path (raw data: [`data/final.json`](data/final.json)):

| config | before fwd / bwd (ms) | after fwd / bwd (ms) | speedup |
|---|---:|---:|---:|
| 28² (MNIST) | 2.06 / 1.71 | 0.33 / 0.55 | 6× / 3× |
| 512² | 6.53 / 5.35 | 0.39 / 0.69 | 17× / 8× |
| 1024², 85 wedges | 29.1 / 23.9 | 0.76 / 0.98 | 38× / 24× |
| 2048², 181 wedges | 64.1 / 52.8 | 4.15 / 4.29 | 15× / 12× |
| 64³ | 52.2 / 42.5 | 0.47 / 0.94 | 111× / 45× |
| 128³ | 53.5 / 44.2 | 3.21 / 2.92 | 17× / 15× |

Zero device-to-host copies per call, bitwise-deterministic results, and backprop
through `UDCTModule` works on CUDA. Not done yet: streaming `filter` for bounded
memory, building windows on the GPU (init still takes ~40 s for 2048²), and fusing the folds into the FFTs (Triton or cuFFT callbacks).


### CPU

The fold step is implemented per device class. On CPUs it is a sorted scatter-add
with a cache-blocked backward (wedges are transformed and folded into the spectrum
in chunks that stay in cache). GPU-style folds would cost redundant arithmetic there;
for example, their backward was about 2× slower at 64³ on this CPU.

float32, single image, `UDCTModule` path, min over repeated runs, ms, fwd / bwd
(raw data: [`data/cpu.json`](data/cpu.json)):

| config | per-wedge (#83), 4 threads | CPU path (#85), 4 threads | per-wedge, 1 thread | CPU path, 1 thread |
|---|---:|---:|---:|---:|
| 256² | 3.0 / 2.6 | 0.9 / 0.9 | 3.0 / 3.0 | 1.1 / 1.2 |
| 512² | 6.1 / 5.9 | 4.7 / 4.0 | 8.4 / 8.2 | 5.7 / 4.9 |
| 1024², 85 wedges | 28.7 / 29.3 | 20.4 / 19.0 | 38.5 / 37.6 | 24.4 / 24.0 |
| 64³ | 24.6 / 22.1 | 6.3 / 5.9 | 21.3 / 18.9 | 8.0 / 7.4 |
| 128³ | 98.6 / 69.1 | 59.3 / 52.5 | 124.4 / 83.9 | 93.1 / 67.1 |

At 128³ the forward varies by ±20% between runs (first-touch costs of ~66 MB
buffers), so compare it with care.

The original assessment follows.

## TL;DR

The current GPU path is **launch- and sync-bound**, not compute- or bandwidth-bound. Every
wedge issues ~20 tiny kernels plus several blocking `.item()` device→host copies, so the GPU
sits idle most of the time. 512² on the GPU (6.5 ms) is barely faster than NumPy on one CPU
(9.9 ms), and a 3D 64³ forward is *slower* on the GPU (52 ms) than in NumPy (20 ms).

A ~250-line fused prototype (`FusedUDCT` in `gpu_bench.py`) gives the same coefficients (rel.
err 2–6e-7 fp32, 1e-15 fp64) and runs **12–160× faster than today's GPU path** and
**40–70× faster than NumPy** at large sizes. It already reaches ~50% of the card's memory
bandwidth. The rest of the gain needs fewer bytes per transform (see P2).

There are also two **correctness bugs** that only show up on GPU (details under P0).

## Results (float32, ms per transform, batch 1)

| config | wedges | NumPy CPU fwd | **current** fwd / bwd | P0 fixes fwd / bwd | **fused** fwd / bwd | fused + CUDA graph fwd | fused vs current |
|---|---:|---:|---:|---:|---:|---:|---:|
| 28² (MNIST), 2 scales | 7 | 0.20 | 2.06 / 1.71 | 0.87 / 0.61 | 0.17 / 0.18 | **0.034** | 12× (60× w/ graph) |
| 512², 3 sc, 3 wpd | 19 | 9.9 | 6.53 / 5.35 | 2.19 / 1.51 | 0.25 / 0.26 | 0.17 | 26× |
| 1024², 4 sc, 6 wpd | 85 | 46.0 | 29.1 / 23.9 | 9.56 / 6.44 | 0.90 / 0.92 | 0.85 | 32× |
| 2048², 5 sc, 6 wpd | 181 | 229 | 64.1 / 52.8 | 21.4 / 15.2 | 5.40 / 5.35 | 5.35 | 12× |
| 64³, 3 sc, 3 wpd | 136 | 19.6 | 52.2 / 42.5 | 15.6 / 10.5 | 0.33 / 0.38 | 0.28 | **159×** |
| 128³, 3 sc, 3 wpd | 136 | 245 | 53.5 / 44.2 | 16.5 / 11.1 | 3.56 / 3.21 | 3.54 | 15× |

Launch and sync counts per forward (torch.profiler):

| config | current kernels / D→H syncs | P0 fixes | fused |
|---|---:|---:|---:|
| 512² | 386 / 96 | 177 / 1 | 22 / 0 |
| 2048² | 3613 / 902 | 1619 / 0 | 20 / 0 |
| 64³ | 3252 / 948 | 1345 / 0 | 7 / 0 |

Batched throughput (forward, ms per image):

| config | Python loop (P0) | `torch.vmap` (P0) | fused, native batch |
|---|---:|---:|---:|
| 28², B=512 | 0.855 | 0.0025 | **0.0005** (≈2 M img/s) |
| 512², B=16 | 2.18 | 0.30 | 0.34 ⚠ |
| 64³, B=8 | 15.5 | 2.90 | 0.50 |
| 2048², B=4 | 21.4 | 8.85 | 6.42 |

float64 on this consumer card is 3–4× slower than float32 for the fused path, because FP64
throughput is 1/64 of FP32 and the FFTs dominate. The unfused paths are launch-bound, so they
barely change with dtype. Datacenter GPUs (A100/H100) would close most of that gap; I didn't
measure them.

## Where the time goes today

Per wedge, `SparseWindow.analyze` / `synthesize` does: gather → cast window values to the
working dtype → multiply → `zeros` → `index_add_` → small `ifftn` → scale. In addition:

1. **`UDCT.apply_to_tensors` drops the periodized index maps.** It rebuilds every
   `SparseWindow(indices, values, shape)` without `folded_indices`, `flipped_*`, `decimation`
   or `out_shape`. So after `UDCTModule.to("cuda")` (or even `.to("cpu")`), **every call
   recomputes** unravel → mod → ravel for every wedge, plus `decimated_shape`, which calls
   `.item()` on each GPU element. That accounts for 2.2–2.8× of the gap between "current" and
   "P0 fixes" above.
2. **Per-wedge host syncs:** `_wedge_analyze_scale` / `_wedge_synthesize_scale` /
   `_forward_lowpass_periodized` compute a Python float from a CUDA tensor via `.item()`, and
   `scale_norm` does too. These are tiny constants that can be precomputed once at init.
3. **~10–20 kernel launches per wedge:** at 181 wedges that's ~3600 launches. On this GPU a
   launch plus Python dispatch costs ~5–10 µs, so the floor is ~20–30 ms no matter how small
   the data.
4. **Window values are stored as float64** and get `.to(dtype)` on every call (one extra kernel
   plus allocation per wedge for fp32 users). Indices are int64.
5. **`vect()` concatenates** all wedges into a new buffer, and `struct()` slices it back, so every
   `UDCTModule` call pays an extra full copy of the coefficients.

## Bugs found

- **`UDCTModule` autograd crashes on CUDA.** `UDCT.struct()` (and `coefficient_shapes()`)
  compute shapes as `torch.tensor(shape)  # CPU` `// decimation_ratio  # CUDA after .to()`,
  which raises *"Expected all tensors to be on the same device"*. The autograd backward calls
  `struct()`, so **backprop through `UDCTModule` on GPU always fails** (it failed in every config
  in `gpu.log`). The MNIST example doesn't catch this because it never backprops into the
  transform, and CI has no CUDA tests.
- `apply_to_tensors` losing the periodized maps (above). For `transform_kind="complex"` it also
  drops the flipped maps, so the negative-frequency wedges recompute two index maps per call.

## Roadmap

### P0: bug fixes and quick wins (~1 day, ≈3× today, no API change)

1. `apply_to_tensors`: move *all* `SparseWindow` tensors (`folded_indices`, `flipped_*`,
   `decimation`) instead of reconstructing the window. Simplest fix: give `SparseWindow` an
   `apply(fn)` method.
2. Fix `struct` / `coefficient_shapes`: precompute coefficient shapes as Python tuples at init
   (they're static), and slice with Python ints. This fixes the autograd crash.
3. Precompute every per-wedge scale as a Python float at init, and keep decimation ratios as
   Python ints / CPU tensors. That leaves zero `.item()` / `.tolist()` calls in the hot path,
   which also removes the `torch.compile` graph breaks.
4. Cache window values in the working dtype (lazily, per dtype), so there's no per-call cast.
5. Add a `cuda` pytest marker (round trip, gradcheck in fp64, `UDCTModule` backward), run through
   on a GPU machine (`benchmarks/gpu_check.py`). Every bug above would have been caught by a 30-second CUDA test.

The "P0 fixes" column above measures items 1 and 3 together. Items 2 and 4 add a little more.

### P1: fused execution engine (the big step, 10–160× today)

Replace the per-wedge Python loop with a precomputed **plan** built once at init (prototype:
`FusedUDCT`):

- Concatenate all wedges' `indices`, offset `folded_indices`, and pre-cast `values` into three
  flat tensors. Group consecutive wedges with the same decimated shape (2–9 groups in practice).
- **Forward:** `F = fftn(x)` → one gather `F[idx] * vals` → one fold into a flat coefficient
  buffer laid out in `vect()` order → one *batched* `ifftn` per shape group → in-place scale.
- **Backward:** the mirror: batched `fftn` per group → one gather → one `index_add_` → `ifftn`.
- Coefficients live in **one flat buffer**, and the nested-list API returns *views* into it. Then
  `vect()` is free, `struct()` is a set of views, and `UDCTModule` returns the buffer directly
  without `torch.cat`.
- **Native batch dimension** `(B, *shape)` instead of `vmap`. The prototype still regresses at
  mid sizes (512², B=16 is slower per image than B=1) because `F[:, idx]` / `index_add_(1, …)`
  on a `(B, N)` layout aren't coalesced. Gather on a flattened `B·N` index, or use an `(N, B)`
  layout so the batch is the contiguous axis.
- **Make the forward a pure gather (deterministic, no atomics, no zero-fill):** within a window
  the folded indices are unique in most configs, and the maximum fold multiplicity I observed is
  **2** (it shows up at 4–5 scales with 6 wedges). Precompute a padded inverse map `[ncoef, K=2]`
  of (source index, weight) and evaluate `(F[inv_idx] * inv_w).sum(-1)`. The backward is the
  transpose: each full-grid frequency is covered by **≤ 6** windows, so a padded `[N, 6]` map
  also makes it a pure gather. Both directions become deterministic. `index_add_` on CUDA
  floats is not, which matters for reproducible training and gradchecks.
- Optional: capture with **CUDA graphs** (or `torch.compile(mode="reduce-overhead")` once P0
  removes the `.item()` breaks) for small inputs. At MNIST size that's 0.17 → 0.034 ms (5×).
  Above ~512² it doesn't matter.

### P2: reach the bandwidth roofline (a further ~2–3× on large inputs)

The fused prototype moves ~600 MB per 2048² fp32 forward in 5.4 ms (~110 GB/s, ~50% of peak),
and ~445 MB per 128³ in 3.6 ms. Most remaining time is bytes, so cut bytes:

1. **Real input → `rfftn` / `irfftn`** for the full-size FFT. `F` is Hermitian, so the gather can
   read `conj(F_half[-k])` with a precomputed conjugate mask. That halves the largest
   transform's memory and FFT work, and on the backward only the half spectrum needs to be
   accumulated.
2. **int32 indices** (N < 2³¹ for every practical size) and a single combined index map. That cuts
   the per-nonzero traffic from ~36 B to ~16 B. Window storage drops too (125 MiB for 2048²,
   73 MiB for 128³ today).
3. **Fuse the gather into the FFT.** Use a Triton kernel for gather·window·scale that writes
   straight into the batched-FFT input, or a cuFFT load callback (`nvmath-python`), so the
   folded buffer never round-trips through HBM. On the backward, fuse the scatter into the inverse
   FFT's store callback.
4. Reuse workspaces (preallocated `out=` buffers, cached cuFFT plans per group) so steady-state
   calls allocate nothing. Fused peak memory is ~2.5× the current path today because of the
   temporary buffers.

### Also worth fixing

- **Init is slow and CPU-only:** 54 s for 2048² (6 wpd), 130 s for 4096², 84 s for 256³ (local
  CPU). The window math is all torch ops, so accept `device=` and build on the GPU, and/or cache
  the plan to disk keyed by the parameters (`torch.save` of the flat tensors). Today the init
  cost dwarfs thousands of transforms.
- **Complex mode** fits the same plan: flipped windows are just more entries in the index maps.
  **Monogenic** (not implemented in torch yet) fits as extra weight columns (Riesz samples) in
  the same gather.
- The NumPy backend has the same per-wedge structure. Its CPU hotspot is `np.add.at`, which a
  `bincount`-style fold or the same inverse-map gather would replace. That's out of scope for a
  GPU focus.

## Suggested order

P0 (1 day, fixes real bugs, ~3×) → P1 plan/flat-buffer engine with native batching and
gather-only maps (~1 week, 10–160×; this is where the API decision about flat coefficient
buffers and views gets made) → P2 rfft + int32 + Triton/callback fusion (as needed; ~2–3× more
for large 2D/3D).
