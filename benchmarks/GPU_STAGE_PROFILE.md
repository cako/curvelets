# Where the time goes: per-stage GPU profile of the fused UDCT

Measured on the native-batching code (#86; the code is unchanged by #87) with
`benchmarks/stage_profile.py`, on an **RTX 4090** (Ada AD102, 72 MB L2,
~1 TB/s), torch 2.8.0+cu128, float32, real transform, `forward(structured=False)` /
`backward` (the `UDCTModule` path).

## Method

`stage_profile.py` replays `UDCTPlan.forward_flat` / `backward_flat` (gather
strategy) stage by stage with the plan's own index maps, and records a CUDA
event between stages. The replay matches the library output bit-for-bit (error
0.0 in every config), and the stage sums match the uninstrumented end-to-end
device time to within 1–3%.

- **device ms**: a `torch.cuda._sleep` is queued first, so the CPU runs ahead
  and the GPU never waits for a launch. Event deltas are pure execution time.
- **host ms**: CPU time to enqueue a call (Python + dispatcher + launches).
- **wall ms**: one call followed by `synchronize`, i.e. what a user sees.
  wall ≈ max(device, host).
- **GB/s**: a lower bound on traffic (each input, index and weight read once,
  each output written once) divided by device time.

Rerun: `python benchmarks/stage_profile.py --batch 1,8 --json out.json`
(add `--configs 2d_2048_w6,3d_128` to restrict).

## 1. Two regimes

| config | B | fwd device | fwd host | fwd wall | inv device | inv host | inv wall | kernels fwd/inv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 28² (MNIST) | 1 | 0.027 | 0.240 | 0.241 | 0.071 | 0.485 | 0.487 | 12 / 34 |
| 28² (MNIST) | 8 | 0.032 | 0.322 | 0.334 | 0.097 | 0.780 | 0.822 | 14 / 43 |
| 512² | 1 | 0.050 | 0.410 | 0.414 | 0.114 | 0.831 | 0.831 | 18 / 44 |
| 512² | 8 | 0.274 | 0.483 | 0.476 | 0.349 | 1.061 | 1.033 | 22 / 56 |
| 1024², 85 wedges | 1 | 0.154 | 0.648 | 0.649 | 0.228 | 1.095 | 1.088 | 30 / 57 |
| 1024², 85 wedges | 8 | 1.578 | 0.785 | 1.581 | 1.549 | 1.384 | 1.566 | 37 / 72 |
| 2048², 181 wedges | 1 | 0.714 | 0.774 | 0.778 | 0.697 | 1.213 | 1.221 | 30 / 57 |
| 2048², 181 wedges | 8 | 8.862 | 0.948 | 8.865 | 8.874 | 4.151 | 8.890 | 34 / 69 |
| 64³ | 1 | 0.080 | 0.503 | 0.508 | 0.155 | 1.020 | 0.993 | 15 / 45 |
| 64³ | 8 | 0.612 | 0.616 | 0.618 | 0.610 | 1.261 | 1.260 | 23 / 62 |
| 128³ | 1 | 0.637 | 0.513 | 0.630 | 0.545 | 1.062 | 1.005 | 13 / 43 |
| 128³ | 8 | 8.024 | 0.629 | 8.024 | 7.307 | 2.602 | 7.311 | — |

(ms per call)

- **Launch-bound (host > device):** every single-image call up to 1024², all
  of 64³, and the *inverse* at 2048² and 128³. The GPU finishes in 0.03–0.7 ms
  but Python takes 0.24–1.2 ms to enqueue it, about 20 µs of host time per
  kernel. At 512² the forward does 50 µs of GPU work in 414 µs of wall time.
  The inverse costs about twice the forward in host time because it launches
  2–3× more kernels (the sparse fold levels; see §3).
- **Bandwidth-bound (device > host):** batched work at ≥1024² and ≥128³. The
  breakdown in §2 applies here.
- `UDCTModule` forward wall ≈ flat `forward` wall: the autograd wrapper adds
  nothing measurable.
- The nested API costs host time only: at 2048² B=1, `forward()` adds 0.32 ms
  over the flat `forward` (181 `unbind` views) and `backward()` adds 0.65 ms (the
  `flatten` cat). At 128³ B=1 it adds +0.12 / +0.49 ms.

## 2. Stage breakdown (device ms)

Grouped by stage type. "fold" covers every level (dense L0, the sparse levels
and the tail). "wedge FFTs" covers every per-group batched FFT.

### Forward

| config | B | rfftn (full) | cat [F, conj F] | fold | wedge iFFTs | total |
|---|---:|---:|---:|---:|---:|---:|
| 512² | 1 | 0.008 (13%) | 0.006 (10%) | 0.011 (18%) | 0.036 (59%) | 0.061 |
| 512² | 8 | 0.027 (10%) | 0.018 (7%) | 0.084 (31%) | 0.141 (52%) | 0.269 |
| 2048² | 1 | 0.068 (9%) | 0.037 (5%) | 0.337 (47%) | 0.280 (39%) | 0.721 |
| 2048² | 8 | 0.678 (8%) | 0.880 (10%) | 3.083 (35%) | 4.245 (48%) | 8.887 |
| 128³ | 1 | 0.035 (6%) | 0.019 (3%) | 0.293 (46%) | 0.289 (45%) | 0.636 |
| 128³ | 8 | 0.357 (4%) | 0.435 (5%) | 2.875 (36%) | 4.388 (54%) | 8.054 |

### Inverse

| config | B | wedge FFTs | fold A (plain) | fold B (conj) | A + conj(B) | irfftn (full) | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512² | 1 | 0.036 (26%) | 0.047 (34%) | 0.037 (27%) | 0.005 (4%) | 0.011 (8%) | 0.137 |
| 512² | 8 | 0.155 (41%) | 0.082 (22%) | 0.072 (19%) | 0.016 (4%) | 0.035 (9%) | 0.374 |
| 2048² | 1 | 0.278 (37%) | 0.172 (23%) | 0.188 (25%) | 0.043 (6%) | 0.075 (10%) | 0.756 |
| 2048² | 8 | 4.244 (48%) | 1.217 (14%) | 1.532 (17%) | 0.720 (8%) | 1.193 (13%) | 8.907 |
| 128³ | 1 | 0.287 (48%) | 0.140 (23%) | 0.113 (19%) | 0.019 (3%) | 0.040 (7%) | 0.599 |
| 128³ | 8 | 4.392 (60%) | 1.123 (15%) | 0.863 (12%) | 0.362 (5%) | 0.586 (8%) | 7.327 |

The largest single stages:

- **Forward fold, dense level L0:** 2048² 0.322 ms (B=1) / 3.05 ms (B=8);
  128³ 0.293 / ~2.9 ms. It reaches 627 → 375 GB/s, against ~1 TB/s peak.
- **The largest wedge-FFT groups:** `48x(1024,64)` and `48x(64,1024)` at
  2048² take 1.75 ms *each* at B=8 (230 GB/s). At 128³, `36x(32,32,64)` takes
  1.65 ms, against 1.13 ms for the same-size `36x(64,32,32)`.
- **Small inverse fold levels:** each sparse level costs ≥10 µs of device time
  even with only 100–2,000 targets (3 kernels per level). At 512² B=1, the 9
  fold levels are 61% of the inverse device time.

## 3. What the kernels show (torch.profiler, `benchmarks/profile_plan.py`)

- **The fold's weight multiply costs more than its gather.** `out = src[idx] * w`
  runs as `index_select` followed by a separate complex×real `mul`, and the
  `mul` goes through the slow non-vectorized `unrolled_elementwise_kernel`
  (type promotion). Per forward call: 2048², mul 0.18 ms vs gather 0.15 ms;
  128³, mul 0.17 ms vs gather 0.12 ms. A microbenchmark of the 8.4M-target L0
  confirms it: `index_select` alone takes 0.154 ms, `index_select * w` takes
  0.340 ms (B=1). At B=8 the numbers are 1.21 vs 2.67 ms. The library's B>1
  path (`row[:, idx]`) is another ~15% slower than `index_select(1, idx)`
  (3.06 ms).
- **ifftn normalization is a separate pass.** Each `ifftn` group launches an
  extra elementwise scale kernel (9 per forward at 2048², ~47 µs/call at
  B=1). The plan already folds per-wedge scales into the fold weights, and the
  `1/size` factor could be folded in the same way.
- **The full-size FFTs clone their input:** 2 `aten::clone` per forward call
  (~43 µs at 2048² B=1), and `irfftn` clones its input because C2R overwrites
  it.
- **Inverse kernel count:** two folds × (L0 + 3 sparse levels + tail) ×
  (gather, mul, index_add), plus the combine. That is about 30 of the 43–57
  kernels in the inverse, and the main reason it is host-bound.

## 4. Why batch 8 costs more than 8× batch 1

At B=8 the wedge FFTs are 2–2.5× slower *per image* than at B=1. For 2048²,
`ifftn 48x(1024,64)` takes 0.089 ms at B=1 and 1.754 ms at B=8. Splitting
the two causes with a microbenchmark (same group, 2048² layout):

| `ifftn 48x(1024,64)` | B=1 | B=8 |
|---|---:|---:|
| strided group view (as in the plan: `(B, ncoef)` rows) | 0.061 | 1.756 |
| same data, contiguous | 0.052 | 1.317 |

| 128³ group, contiguous, B=8 | ms |
|---|---:|
| `36x(32,32,64)` | 1.319 |
| `36x(64,32,32)` | 0.785 |

- **About 25% comes from layout.** With a `(B, ncoef)` buffer, each group's
  slice is non-contiguous across the batch.
- **The rest is cache.** At B=1 the large configs fit, or nearly fit, in the
  4090's 72 MB L2. At B=8 they go to DRAM. B=1 stage numbers on this card are
  therefore optimistic for cards with a smaller L2, such as the RTX 2000 Ada
  baseline used in #81–#86.
- **Axis order matters for cuFFT:** the same 3D wedge size is 1.7× faster when
  the long axis is outermost. ⚠️ This is one microbenchmark of two group shapes
  on one card and one cuFFT version; see the caveats under §5, item 3(c).

## 5. Where the next gains are (ranked)

1. **Remove the launch overhead (B=1 and small/medium sizes).** Capture
   `forward_flat` / `backward_flat` in a CUDA graph per (shape, batch, dtype),
   or use `torch.compile(mode="reduce-overhead")`. The plan is static, so this
   fits well. Expected wall time ≈ device time: 512² forward 0.41 → ~0.05 ms
   and inverse 0.83 → ~0.12 ms; 2048² inverse 1.22 → ~0.70 ms; 128³ inverse
   1.0 → ~0.55 ms. Fewer kernels (item 2) also helps without graphs.
2. **One fused fold kernel (Triton or CUDA), both directions.** Store the
   contributions as CSR by target and have each output element loop over its
   sources. Weight, conjugation (a sign flip on the imaginary part, which
   removes `cat [F, conj F]`), the `A + conj(B)` combine and the ifft `1/size`
   factor can all go into the same kernel. It stays gather-only and
   deterministic, with no atomics.
   - Forward: replaces cat + gather + mul (+ the sparse level) with one kernel.
   - Inverse: replaces ~30 kernels with one.
   - Rough device savings at 2048² B=8: forward ~3.9 → ~1.3 ms, inverse
     ~3.5 → ~1.0 ms (about 30% of each transform). At B=1 it removes most of
     the inverse's launches.
   - A cheaper first step: make the weight multiply a vectorized complex×complex
     op, or apply the weight with `torch.view_as_real`. That alone should
     recover much of the `mul` cost.
3. **Wedge FFTs (48–60% of device time when batched).**
   - (a) Lay out batched coefficients group-major (`[group][B][count][shape]`)
     so each group's FFT input is contiguous: about −25% on these FFTs. This
     changes the batched `vect` layout or needs a permute at the API boundary.
   - (b) Fold the ifft normalization into the weights (`norm=` choice): one
     fewer full pass over the coefficients.
   - (c) ⚠️ **Not a free win.** Store 3D wedges with the long axis outermost
     when cuFFT prefers it (1.7× on the `x,x,2x` groups in one microbenchmark).
     - It changes the coefficient layout: the `vect` order and the nested wedge
       shapes would no longer match the NumPy backend or saved coefficients.
     - Keeping the public layout means transposing back at the API boundary,
       an extra read and write of those coefficients that can cancel most of
       the gain on a memory-bound transform.
     - The 1.7× is from two group shapes on one GPU (RTX 4090) and one cuFFT
       version; cuFFT's algorithm choice varies by GPU and version.
     Only worth pursuing as an internal layout (e.g. written by a fused fold
     into scratch space) and after measuring it on more than one card.
   - (d) Longer term, cuFFT load/store callbacks (nvmath-python) to fuse the
     fold into the FFT so the folded buffer never goes through DRAM.
4. **Full-size FFTs (13–21% at 2048² B=8).** Mostly at cuFFT speed already.
   Avoiding the `irfftn` input clone (with an `out=`/scratch buffer) is the
   only cheap win.
5. **Nested API (host only):** build the nested lists with one `split` per
   group instead of per-wedge `unbind`, and skip the `flatten` cat in
   `backward()` when the coefficients are already views of one buffer.
   Worth 0.3–0.65 ms per call at 2048² B=1.

## 6. Follow-up: compilation, CUDA graphs, batch chunking, on-the-fly conjugation

Same RTX 4090, float32, real transform. Scripts: `benchmarks/compile_experiments.py`
(library code as is) and a prototype "real plan" built from the library plan. Its folds use real
arithmetic: a gather from `view_as_real(spectrum)` times weight pairs
`(w, ±w)`. This applies the conjugation on the fly, so there is no
`[F, conj F]`. The `ifftn` `1/size` factor goes into the weights, and the
plain and conjugated backward folds are merged into one fold. (The prototype
became the library change in the next PR of this stack.) Every variant
matches the library to ≤2e-7 relative error.

### Best wall time per case (ms, sync per call)

| case | library eager | best | how |
|---|---:|---:|---|
| 512² B=1 fwd | 0.370 | **0.049** (7.5×) | real plan, `torch.compile` (default) inside a manual CUDA graph |
| 512² B=1 inv | 0.695 | **0.057** (12×) | same |
| 512² B=8 fwd | 0.426 | **0.222** | same |
| 512² B=8 inv | 0.886 | **0.296** | same |
| 2048² B=1 fwd | 0.753 | **0.618** | same |
| 2048² B=1 inv | 1.060 | **0.753** | same |
| 2048² B=8 fwd | 8.924 | **5.042** (−43%) | same, one batch row per call (row loop) |
| 2048² B=8 inv | 9.143 | **6.015** (−34%) | same |
| 128³ B=1 fwd | 0.688 | **0.575** | compile + manual graph |
| 128³ B=1 inv | 0.849 | **0.623** | library + manual graph |
| 128³ B=8 fwd | 8.069 | **4.532** (−44%) | compile + graph + row loop |
| 128³ B=8 inv | 7.387 | **5.150** (−30%) | library + graph + row loop |

### Findings

1. **`torch.compile` on the library code as it is now makes things worse** in
   every mode: up to 2–4× slower at 2048² and 128³. Two reasons:
   `_functorch_wrapped` calls a builtin that dynamo can't trace (3 graph
   breaks), and Inductor generates no code for complex operators ("Performance
   may be worse than eager").
2. **With real-arithmetic folds it compiles to a single graph (0 breaks)**,
   and Inductor fuses gather × weight (+ conj) into Triton kernels. Default
   mode is the best mode. `reduce-overhead` and `max-autotune` (both use
   CUDA-graph trees) are consistently slower on device. `max-autotune` finds
   nothing for these memory-bound gathers and costs 4–9 s of compilation.
   Compile times in default mode are 0.6–2 s.
3. **A manual `torch.cuda.CUDAGraph` beats `reduce-overhead`** in the
   launch-bound regime (512² B=1: 0.049 vs 0.127 ms). Above that, graph
   capture alone is neutral or slightly *slower*: the graph's private memory
   pool changes buffer reuse, and the batched inverse loses about 10%.
4. **Batch chunking is the easy batch-friendly win.** Processing a batch one
   row at a time (graphed, so the extra launches cost nothing) keeps the
   working set in L2. It is 30–44% faster at 2048² and 128³ with B=8. At 512²,
   where B=8 already fits in L2, it is slower: the chunk size should depend on
   the working-set size versus L2, as the CPU path already does.
5. **On-the-fly conjugation (fold L0 alone):** 2048² B=8 goes 4.84 → 4.02 ms
   eager and 2.90 ms compiled. Extra memory goes 1798 → 1542 → 1028 MiB. Peak
   memory of the whole compiled forward is 1221 MiB vs 1796 MiB (−32%) at
   2048² B=8, and 129 vs 201 MiB (−36%) at B=1. Uncompiled, the
   `view_as_real` gather with an inner size of 2 is no faster than the
   complex gather at B=1, and the merged backward fold is slower than the
   two separate folds. The gains come from the fusion.

## 7. Follow-up: fused Triton fold, grouped layout, FFT fusion bound (A40)

NVIDIA A40 (6 MB L2), torch 2.15 (cu132), Triton 3.9, float32, real transform.
On this card every batch of 8 runs row by row (§6, item 4 and the L2 rule).
Raw fold results: `benchmarks/data/fold_bench_a40.json` (`benchmarks/fold_bench.py`).

- **Fused fold (item 2):** one Triton kernel per direction (CSR by target) takes
  43–61% off the forward fold and 35–81% off the backward fold (§5 predicted
  ~2/3 off at 2048² B=8; measured −59% / −75%). End to end see the PR.
- **Grouped layout (item 3a): not worth doing after row-by-row batching.** Per-group
  FFTs on the strided `(B, ncoef)` view vs a contiguous group-major copy, device ms:

  | config | B=1 strided / contiguous | B=8 strided / contiguous | regroup copy, B=8 |
  |---|---:|---:|---:|
  | 512² | 0.053 / 0.053 | 0.500 / 0.369 | 0.139 |
  | 1024², 85 wedges | 0.168 / 0.167 | 1.954 / 1.474 | 0.502 |
  | 2048², 181 wedges | 0.725 / 0.727 | 7.706 / 5.830 | 1.926 |
  | 64³ | 0.100 / 0.100 | 1.204 / 0.958 | 0.249 |
  | 128³ | 0.958 / 0.961 | 9.405 / 7.623 | 1.872 |

  With one row the group slices are already contiguous, so the ~25% gap only
  exists for whole-batch runs. Large rows no longer run whole-batch, and where
  batches still do (rows that fit in L2), copying back to the `vect` layout costs
  as much as the contiguous FFTs save (e.g. 1024² B=8: −0.48 ms FFT, +0.50 ms copy).
- **Fold-into-FFT fusion (item 3d): bounded at ~15–20%.** A cuFFT load callback
  removes one write and one read of the folded `(rows, ncoef)` buffer, about one
  copy of it: 0.24 ms of 1.26 ms (19%) for the 2048² B=1 forward, 0.24 / 1.35 ms
  (18%) at 128³, 4–6% for small batched inputs. That is an upper bound; gathers
  inside callbacks usually cost cuFFT some of its own throughput.
