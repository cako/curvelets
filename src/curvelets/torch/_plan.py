"""Fused execution plan for the PyTorch UDCT.

The reference implementation (``_forward_transform`` / ``_backward_transform``)
handles one wedge at a time: a gather, a fold, a small FFT and a scale per wedge,
i.e. roughly ten kernel launches per wedge. On a GPU that makes the transform
launch-bound: thousands of tiny kernels for large configurations.

:class:`UDCTPlan` precomputes the same arithmetic for *all* wedges at once:

- forward: one full-size FFT, one fold of every window sample into a flat
  buffer laid out in :meth:`UDCT.vect` order, and one batched inverse FFT per run
  of consecutive equal-shape wedges;
- backward: the mirror image (batched FFTs, one fold, one full-size inverse FFT).

The fold step has one implementation per device class:

- GPUs and other accelerators: one deterministic gather-only fold over all
  wedges (:class:`~._foldmap.GatherFold`) and batched FFTs, i.e. few launches.
- CPUs: sorted scatter-adds (:class:`~._foldmap.ScatterFold`), applied row by
  row, with no redundant arithmetic. The forward folds all wedges at once (it
  has to read the whole spectrum anyway). The backward is cache-blocked: wedges
  are processed in chunks of about :data:`CPU_CHUNK_ELEMENTS` coefficients,
  each transformed and folded into the spectrum while it is still in cache, so
  the large intermediate buffer of all coefficient spectra never exists.
The real transform works on half spectra: the forward gathers from ``rfftn`` of
the (real) input, reading ``conj(F[-k])`` for frequencies outside the stored
half (from a ``[F, conj(F)]`` buffer, so every term is a plain gather), and the
backward accumulates the Hermitian part of the spectrum, ``A + conj(B)``, whose
``irfftn`` equals ``ifftn(F).real``. That halves the full-size FFTs and buffers.

The launch count no longer grows with the number of wedges. Every op is
out-of-place and differentiable, so the plan works under autograd and
``torch.vmap``, and it accepts arbitrary leading batch dimensions.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ._foldmap import GatherFold, ScatterFold, _functorch_wrapped, _index_dtype
from ._periodized_fft import _ravel_multi_index, _unravel_index
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows

CPU_CHUNK_ELEMENTS = 1 << 19  # coefficients per CPU chunk (~4 MiB as complex64)

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def _direct_ok(tensor: torch.Tensor) -> bool:
    """Whether FFT results may be written in place into a preallocated buffer.

    ``out=`` variants are not differentiable and are not supported inside
    functorch transforms (``vmap``, ``grad``), so fall back in those cases.
    """
    if torch.is_grad_enabled() and tensor.requires_grad:
        return False
    return not _functorch_wrapped(tensor)


@dataclass(frozen=True)
class _Group:
    """Run of consecutive wedges (in ``vect`` order) that share a decimated shape."""

    offset: int
    count: int
    shape: tuple[int, ...]
    fwd_scale: tuple[float, ...]
    bwd_scale: tuple[float, ...]

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def stop(self) -> int:
        return self.offset + self.count * self.size


@dataclass(frozen=True)
class _Entry:
    window: SparseWindow
    flip: bool
    fwd_scale: float
    bwd_scale: float  # 0.0: coefficient is not used by the backward transform
    position: tuple[int, int, int]  # (scale, direction, wedge) in the output


@dataclass(frozen=True)
class _CpuChunk:
    """Consecutive wedges of one group, folded and transformed together on CPU."""

    start: int  # slot range [start, stop) in the flat coefficient vector
    stop: int
    count: int
    shape: tuple[int, ...]
    bwd: tuple[ScatterFold, ...]  # this chunk's spectra -> spectrum (plain, conj)


@dataclass
class UDCTPlan:
    """
    Flattened index maps and scale factors for a fused forward/backward UDCT.

    Build it with :meth:`build`; the coefficient layout matches :meth:`UDCT.vect`
    and the results match the per-wedge reference implementation.
    """

    shape: tuple[int, ...]
    complex_output: bool
    groups: list[_Group]
    layout: list[tuple[int, int, int]]
    num_directions: list[int]
    ncoef: int
    fwd_map: GatherFold | ScatterFold  # spectrum ([F, conj F] or full) -> folded
    # Accelerator backward (gather strategy): one fold over all wedges.
    bwd_map: GatherFold | None  # coefficient spectra -> spectrum (half or full)
    bwd_conj_map: GatherFold | None  # half-spectrum terms that enter conjugated
    # CPU backward (scatter strategy): cache-sized chunks of wedges.
    cpu_chunks: list[_CpuChunk] | None
    # CPU forward scratch space ([F, conj F]), reused across calls per thread.
    # Freshly allocated large buffers cost about as much as the FFTs on CPUs,
    # because every page is faulted in again; GPU memory is recycled by PyTorch.
    _scratch: threading.local = field(
        default_factory=threading.local, repr=False, compare=False
    )

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        parameters: ParamUDCT,
        windows: UDCTWindows,
        decimation_ratios: list[torch.Tensor],
        *,
        complex_mode: bool,
        fold_strategy: str | None = None,
    ) -> UDCTPlan:
        """
        Build the plan from windows with attached periodized maps.

        Parameters
        ----------
        parameters : ParamUDCT
            Transform parameters.
        windows : UDCTWindows
            Windows with ``attach_periodized`` maps (and flipped maps in complex
            mode).
        decimation_ratios : list[torch.Tensor]
            Decimation ratios (CPU tensors preferred; only read via ``tolist``).
        complex_mode : bool
            Complex transform (separate positive/negative frequency bands).
        fold_strategy : {"scatter", "gather"}, optional
            Execution strategy. By default CPUs get ``"scatter"`` (cache-blocked
            chunks) and every other device (CUDA, MPS, XPU, ...) ``"gather"``.

        Returns
        -------
        UDCTPlan
            The plan. Entry order, scale factors and wavelet-mode handling mirror
            ``_apply_forward_transform_*`` and ``_apply_backward_transform_*``.
        """
        entries = _plan_entries(
            parameters, windows, decimation_ratios, complex_mode=complex_mode
        )
        groups: list[_Group] = []
        idx: list[torch.Tensor] = []
        gidx: list[torch.Tensor] = []
        fwd_w: list[torch.Tensor] = []
        bwd_w: list[torch.Tensor] = []
        used: list[torch.Tensor] = []
        offset = 0
        for entry in entries:
            window = entry.window
            if window.out_shape is None:
                msg = "windows must have periodized maps attached"
                raise ValueError(msg)
            indices, folded = window.resolve_indices(flip=entry.flip)
            out_shape = tuple(window.out_shape)
            last = groups[-1] if groups else None
            if last is not None and last.shape == out_shape:
                groups[-1] = _Group(
                    last.offset,
                    last.count + 1,
                    out_shape,
                    (*last.fwd_scale, entry.fwd_scale),
                    (*last.bwd_scale, entry.bwd_scale),
                )
            else:
                groups.append(
                    _Group(offset, 1, out_shape, (entry.fwd_scale,), (entry.bwd_scale,))
                )
            idx.append(indices)
            gidx.append(folded + offset)
            # Per-wedge normalizations are folded into the fold weights: the
            # FFTs are linear, so scaling before them saves a full pass over
            # the coefficients in each direction.
            fwd_w.append(window.values * entry.fwd_scale)
            bwd_w.append(window.values * entry.bwd_scale)
            used.append(
                torch.full(
                    (indices.numel(),),
                    entry.bwd_scale != 0.0,
                    dtype=torch.bool,
                    device=indices.device,
                )
            )
            offset += math.prod(out_shape)

        fwd_idx = torch.cat(idx)
        fwd_gidx = torch.cat(gidx)
        fwd_vals = torch.cat(fwd_w)
        bwd_vals = torch.cat(bwd_w)
        if all(entry.bwd_scale != 0.0 for entry in entries):
            bwd_idx, bwd_gidx = fwd_idx, fwd_gidx
        else:
            mask = torch.cat(used)
            bwd_idx, bwd_gidx, bwd_vals = fwd_idx[mask], fwd_gidx[mask], bwd_vals[mask]

        shape = tuple(parameters.shape)
        numel = math.prod(shape)
        if complex_mode:
            fwd_c = (fwd_gidx, fwd_idx, fwd_vals)
            bwd_c = [(bwd_idx, bwd_gidx, bwd_vals)]
            num_spectrum, num_fwd_sources = numel, numel
        else:
            fwd_c, bwd_c, num_spectrum = _half_spectrum_contributions(
                shape, (fwd_idx, fwd_gidx, fwd_vals), (bwd_idx, bwd_gidx, bwd_vals)
            )
            num_fwd_sources = 2 * num_spectrum

        strategy = fold_strategy
        if strategy is None:
            strategy = "scatter" if fwd_idx.device.type == "cpu" else "gather"
        fwd_map: GatherFold | ScatterFold
        bwd_map = bwd_conj_map = None
        cpu_chunks = None
        if strategy == "gather":
            fwd_map = GatherFold.build(*fwd_c, offset, num_fwd_sources)
            bwd_folds = [GatherFold.build(*c, num_spectrum, offset) for c in bwd_c]
            bwd_map = bwd_folds[0]
            bwd_conj_map = bwd_folds[1] if len(bwd_folds) > 1 else None
        elif strategy == "scatter":
            fwd_map = ScatterFold.build(*fwd_c, offset, num_fwd_sources)
            cpu_chunks = _cpu_chunks(groups, bwd_c, num_spectrum)
        else:
            msg = f"unknown fold strategy {strategy!r}"
            raise ValueError(msg)

        num_directions = [1] + [
            2 * parameters.ndim if complex_mode else len(windows[scale_idx])
            for scale_idx in range(1, parameters.num_scales)
        ]
        return cls(
            shape=shape,
            complex_output=complex_mode,
            groups=groups,
            layout=[entry.position for entry in entries],
            num_directions=num_directions,
            ncoef=offset,
            fwd_map=fwd_map,
            bwd_map=bwd_map,
            bwd_conj_map=bwd_conj_map,
            cpu_chunks=cpu_chunks,
        )

    # Thread-local scratch cannot be pickled or deep-copied (``copy.deepcopy``,
    # ``torch.save`` of a module): drop it, and start empty on the copy.
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        del state["_scratch"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._scratch = threading.local()

    # ---------------------------------------------------------------- helpers
    @property
    def ndim(self) -> int:
        """Number of transform dimensions."""
        return len(self.shape)

    @property
    def spectrum_shape(self) -> tuple[int, ...]:
        """Shape of the stored spectrum (half spectrum for the real transform)."""
        if self.complex_output:
            return self.shape
        return (*self.shape[:-1], self.shape[-1] // 2 + 1)

    def _dims(self) -> tuple[int, ...]:
        return tuple(range(-self.ndim, 0))

    def _buffer(
        self,
        name: str,
        like: torch.Tensor,
        factor: int,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Thread-local reusable buffer (CPU forward scratch space)."""
        if shape is None:
            shape = (like.shape[0], like.shape[1] * factor)
        cached: torch.Tensor | None = getattr(self._scratch, name, None)
        if (
            cached is None
            or tuple(cached.shape) != shape
            or cached.dtype != like.dtype
            or cached.device != like.device
        ):
            cached = torch.empty(shape, dtype=like.dtype, device=like.device)
            setattr(self._scratch, name, cached)
        return cached

    def _batched_ffts(
        self,
        flat: torch.Tensor,
        transform: Callable[..., torch.Tensor],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply ``transform`` (fftn/ifftn) to every group of a ``(B, ncoef)`` buffer.

        Without autograd or functorch in the way, each group is written straight
        into one output buffer (``out=``); otherwise the groups are concatenated.
        """
        rows = flat.shape[0]
        dims = self._dims()
        chunks = [
            flat[:, g.offset : g.stop].view(rows, g.count, *g.shape)
            for g in self.groups
        ]
        if _direct_ok(flat):
            out = torch.empty(rows, self.ncoef, dtype=out_dtype, device=flat.device)
            for g, chunk in zip(self.groups, chunks, strict=True):
                transform(
                    chunk,
                    dim=dims,
                    out=out[:, g.offset : g.stop].view(rows, g.count, *g.shape),
                )
            return out
        parts = [transform(chunk, dim=dims).reshape(rows, -1) for chunk in chunks]
        return torch.cat(parts, dim=1)

    # ---------------------------------------------------------------- forward
    def forward_flat(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward transform as a ``(*batch, ncoef)`` tensor in ``vect`` order.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *shape)``.

        Returns
        -------
        torch.Tensor
            Flattened coefficients.
        """
        batch = image.shape[: image.ndim - self.ndim]
        if self.complex_output:
            spectrum = torch.fft.fftn(image, dim=self._dims())  # pylint: disable=not-callable
        else:
            spectrum = torch.fft.rfftn(image, dim=self._dims())  # pylint: disable=not-callable
        flat = spectrum.reshape(-1, math.prod(self.spectrum_shape))
        reuse = isinstance(self.fwd_map, ScatterFold) and _direct_ok(flat)
        if not self.complex_output:
            pair = [flat, flat.conj()]
            if reuse:
                flat = torch.cat(pair, dim=-1, out=self._buffer("ext", flat, 2))
            else:
                flat = torch.cat(pair, dim=-1)
        folded = self.fwd_map.apply(flat)
        if reuse:
            # The folded buffer becomes the result: inverse FFTs run in place, so
            # each call touches exactly one fresh coefficient-sized buffer.
            dims = self._dims()
            for g in self.groups:
                view = folded[:, g.offset : g.stop].view(-1, g.count, *g.shape)
                torch.fft.ifftn(view, dim=dims, out=view)  # pylint: disable=not-callable
            coeffs = folded
        else:
            coeffs = self._batched_ffts(folded, torch.fft.ifftn, folded.dtype)
        return coeffs.reshape(*batch, self.ncoef)

    def forward_groups(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward transform, one view per group.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *shape)``.

        Returns
        -------
        list[torch.Tensor]
            Group ``g`` has shape ``(*batch, g.count, *g.shape)``; all groups are
            views of one flat buffer.
        """
        flat = self.forward_flat(image)
        return [
            flat[..., g.offset : g.stop].unflatten(-1, (g.count, *g.shape))
            for g in self.groups
        ]

    def forward_nested(self, image: torch.Tensor) -> UDCTCoefficients:
        """
        Forward transform in the nested ``[scale][direction][wedge]`` layout.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *shape)``.

        Returns
        -------
        UDCTCoefficients
            Each coefficient is a view of shape ``(*batch, *wedge_shape)``.
        """
        coeffs: UDCTCoefficients = [
            [[] for _ in range(ndir)] for ndir in self.num_directions
        ]
        entry = 0
        for tensor in self.forward_groups(image):
            for wedge in tensor.unbind(dim=-self.ndim - 1):
                scale_idx, dir_idx, _ = self.layout[entry]
                coeffs[scale_idx][dir_idx].append(wedge)
                entry += 1
        return coeffs

    # --------------------------------------------------------------- backward
    def flatten(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """
        Flatten nested coefficients to ``(*batch, ncoef)`` in ``vect`` order.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Nested coefficients, each of shape ``(*batch, *wedge_shape)``.

        Returns
        -------
        torch.Tensor
            Flattened coefficients.
        """
        parts = []
        for scale_idx, dir_idx, wedge_idx in self.layout:
            coeff = coefficients[scale_idx][dir_idx][wedge_idx]
            batch = coeff.shape[: coeff.ndim - self.ndim]
            parts.append(coeff.reshape(*batch, -1))
        return torch.cat(parts, dim=-1)

    def backward_flat(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Backward transform of ``(*batch, ncoef)`` coefficients.

        Parameters
        ----------
        vector : torch.Tensor
            Flattened coefficients in ``vect`` order.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(*batch, *shape)``; real for the real
            transform, complex for the complex transform.
        """
        batch = vector.shape[:-1]
        flat = vector.reshape(-1, self.ncoef)
        cdtype = flat.dtype if flat.is_complex() else _COMPLEX_DTYPE[flat.dtype]
        if self.cpu_chunks is not None:
            spectrum = _stack_rows([
                self._cpu_backward_row(row, cdtype) for row in flat
            ])
        else:
            assert self.bwd_map is not None
            small = self._batched_ffts(flat, torch.fft.fftn, cdtype)
            spectrum = self.bwd_map.apply(small)
            if self.bwd_conj_map is not None:
                spectrum = spectrum + self.bwd_conj_map.apply(small).conj()
        spectrum = spectrum.reshape(*batch, *self.spectrum_shape)
        if self.complex_output:
            return torch.fft.ifftn(spectrum, dim=self._dims())  # pylint: disable=not-callable
        return torch.fft.irfftn(spectrum, s=self.shape, dim=self._dims())  # pylint: disable=not-callable

    def _cpu_backward_row(self, row: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
        """One coefficient vector -> (half) spectrum, chunk by chunk (CPU)."""
        assert self.cpu_chunks is not None
        dims = self._dims()
        size = math.prod(self.spectrum_shape)
        terms = 1 if self.complex_output else 2
        acc = [torch.zeros(size, dtype=cdtype) for _ in range(terms)]
        for chunk in self.cpu_chunks:
            spec = torch.fft.fftn(  # pylint: disable=not-callable
                row[chunk.start : chunk.stop].view(chunk.count, *chunk.shape), dim=dims
            ).reshape(-1)
            for k, fold in enumerate(chunk.bwd):
                acc[k] = fold.add_row(acc[k], spec)
        return acc[0] if terms == 1 else acc[0] + acc[1].conj()


def _stack_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    return rows[0].unsqueeze(0) if len(rows) == 1 else torch.stack(rows)


def _cpu_chunks(
    groups: list[_Group],
    bwd: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_spectrum: int,
) -> list[_CpuChunk]:
    """Split the backward contributions into cache-sized chunks of wedges.

    Each ``bwd`` entry is ``(spectrum target, slot source, weight)``. Each list
    is sorted once by (chunk, target), and the chunk folds are views into the
    sorted tensors: a handful of allocations instead of several per chunk, which
    keeps the CPU heap unfragmented for the large per-call buffers.
    """
    bounds: list[
        tuple[int, int, int, tuple[int, ...]]
    ] = []  # start, stop, count, shape
    for group in groups:
        per_chunk = max(1, CPU_CHUNK_ELEMENTS // group.size)
        for first in range(0, group.count, per_chunk):
            count = min(per_chunk, group.count - first)
            start = group.offset + first * group.size
            bounds.append((start, start + count * group.size, count, group.shape))
    num_chunks = len(bounds)
    ncoef = bounds[-1][1]
    target_dtype = _index_dtype(num_spectrum)
    source_dtype = _index_dtype(ncoef)

    folds_per_chunk: list[list[ScatterFold]] = [[] for _ in range(num_chunks)]
    for targets, raw_sources, weights in bwd:
        device = raw_sources.device
        starts = torch.tensor([b[0] for b in bounds], dtype=torch.int64, device=device)
        sources = raw_sources.to(torch.int64)
        chunk_of = torch.searchsorted(starts, sources, right=True) - 1
        order = torch.argsort(chunk_of * num_spectrum + targets.to(torch.int64))
        chunk_sorted = chunk_of[order]
        sorted_targets = targets[order].to(target_dtype)
        local_sources = (sources - starts[chunk_of])[order].to(source_dtype)
        sorted_weights = weights[order]
        ends = torch.cumsum(
            torch.bincount(chunk_sorted, minlength=num_chunks), dim=0
        ).tolist()
        lo = 0
        for chunk_idx, hi in enumerate(ends):
            start, stop, _, _ = bounds[chunk_idx]
            folds_per_chunk[chunk_idx].append(
                ScatterFold(
                    num_targets=num_spectrum,
                    num_sources=stop - start,
                    targets=sorted_targets[lo:hi],
                    sources=local_sources[lo:hi],
                    weights=sorted_weights[lo:hi],
                )
            )
            lo = hi

    return [
        _CpuChunk(start, stop, count, shape, tuple(folds))
        for (start, stop, count, shape), folds in zip(
            bounds, folds_per_chunk, strict=True
        )
    ]


def _half_spectrum_contributions(
    shape: tuple[int, ...],
    fwd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bwd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    int,
]:
    """
    Fold contributions that read from / write to the ``rfftn`` half spectrum.

    A full-grid frequency ``k`` lives in the stored half when its last coordinate
    is at most ``n_last // 2``; otherwise ``F[k] = conj(F[-k])`` for real input.
    The backward produces ``H[k] = (F[k] + conj(F[-k])) / 2`` on the stored half,
    whose ``irfftn`` is ``ifftn(F).real``.

    Returns the forward contributions ``(target slot, source, weight)`` with
    sources in ``[F_half, conj(F_half)]``, the backward contributions for ``A``
    and ``B`` with ``H = A(c) + conj(B(c))``, and the half-spectrum length.
    """
    fwd_idx, fwd_gidx, fwd_vals = fwd
    bwd_idx, bwd_gidx, bwd_vals = bwd
    half_shape = (*shape[:-1], shape[-1] // 2 + 1)
    half_numel = math.prod(half_shape)
    half_last = half_shape[-1]

    def locate(
        full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = _unravel_index(full, shape)
        neg = tuple((-c) % n for c, n in zip(coords, shape, strict=True))
        own_in_half = coords[-1] < half_last
        neg_in_half = neg[-1] < half_last
        zero = torch.zeros_like(coords[-1])
        own = _ravel_multi_index(
            (*coords[:-1], torch.where(own_in_half, coords[-1], zero)), half_shape
        )
        mirror = _ravel_multi_index(
            (*neg[:-1], torch.where(neg_in_half, neg[-1], zero)), half_shape
        )
        return own, own_in_half, mirror, neg_in_half

    own, own_in_half, mirror, _ = locate(fwd_idx)
    # Conjugated samples come from the second half of [F_half, conj(F_half)].
    fwd_src = torch.where(own_in_half, own, mirror + half_numel)

    own, own_in_half, mirror, neg_in_half = locate(bwd_idx)
    plain = (own[own_in_half], bwd_gidx[own_in_half], 0.5 * bwd_vals[own_in_half])
    conj = (mirror[neg_in_half], bwd_gidx[neg_in_half], 0.5 * bwd_vals[neg_in_half])
    return (fwd_gidx, fwd_src, fwd_vals), [plain, conj], half_numel


def _plan_entries(
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
    *,
    complex_mode: bool,
) -> list[_Entry]:
    """Wedges in ``vect`` order with their forward/backward scale factors."""
    ndim = parameters.ndim
    num_scales = parameters.num_scales
    highest = num_scales - 1
    wavelet_highest = len(windows[highest]) == 1
    half = math.sqrt(0.5) if complex_mode else 1.0

    def dec_prod(scale_idx: int, dir_idx: int) -> float:
        ratios = decimation_ratios[scale_idx]
        row = ratios[0] if ratios.shape[0] == 1 else ratios[dir_idx]
        return float(math.prod(int(v) for v in row.tolist()))

    pd0 = dec_prod(0, 0)
    scale_norm = math.sqrt(float(2 ** (num_scales - 2)) ** ndim)
    entries = [
        _Entry(windows[0][0][0], False, scale_norm / pd0, math.sqrt(pd0), (0, 0, 0))
    ]

    def add(scale_idx: int, dir_idx: int, out_dir: int, flip: bool, used: bool) -> None:
        win_dir = min(dir_idx, len(windows[scale_idx]) - 1)
        pd = dec_prod(scale_idx, win_dir)
        fwd = math.sqrt(2.0 * pd) / pd * half
        # Wavelet mode: the backward weighs the other scales by 2 and the
        # ring-shaped highest scale by 1.
        factor = 1.0 if (wavelet_highest and scale_idx == highest) else 2.0
        bwd = math.sqrt(pd / 2.0) * half * factor if used else 0.0
        for wedge_idx, window in enumerate(windows[scale_idx][win_dir]):
            entries.append(
                _Entry(window, flip, fwd, bwd, (scale_idx, out_dir, wedge_idx))
            )

    for scale_idx in range(1, num_scales):
        if not complex_mode:
            for dir_idx in range(len(windows[scale_idx])):
                add(scale_idx, dir_idx, dir_idx, False, True)
            continue
        for flip in (False, True):
            for dir_idx in range(ndim):
                # Wavelet mode repeats the single ring window over the ndim
                # directions; the backward only uses direction 0.
                used = not (wavelet_highest and scale_idx == highest and dir_idx != 0)
                out_dir = dir_idx + ndim if flip else dir_idx
                add(scale_idx, dir_idx, out_dir, flip, used)
    return entries
