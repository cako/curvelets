"""Fused execution plan for the PyTorch UDCT.

The reference implementation (``_forward_transform`` / ``_backward_transform``)
handles one wedge at a time: a gather, a fold, a small FFT and a scale per wedge,
i.e. roughly ten kernel launches per wedge. On a GPU that makes the transform
launch-bound: thousands of tiny kernels for large configurations.

:class:`UDCTPlan` precomputes the same arithmetic for *all* wedges at once:

- forward: one full-size FFT, one gather of every window sample, one fold
  (``index_add``) into a flat buffer laid out in :meth:`UDCT.vect` order, and one
  batched inverse FFT per run of consecutive equal-shape wedges;
- backward: the mirror image (batched FFTs, one gather, one scatter, one
  full-size inverse FFT).

The launch count no longer grows with the number of wedges. Every op is
out-of-place and differentiable, so the plan works under autograd and
``torch.vmap``, and it accepts arbitrary leading batch dimensions.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ._sparse_window import _REAL_DTYPE, SparseWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows


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
    fwd_idx: torch.Tensor
    fwd_gidx: torch.Tensor
    fwd_vals: torch.Tensor
    bwd_idx: torch.Tensor
    bwd_gidx: torch.Tensor
    bwd_vals: torch.Tensor
    _cache: dict[tuple[str, torch.dtype, torch.device], torch.Tensor] = field(
        default_factory=dict, repr=False
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
        vals: list[torch.Tensor] = []
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
            vals.append(window.values)
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
        fwd_vals = torch.cat(vals)
        if all(entry.bwd_scale != 0.0 for entry in entries):
            bwd_idx, bwd_gidx, bwd_vals = fwd_idx, fwd_gidx, fwd_vals
        else:
            mask = torch.cat(used)
            bwd_idx, bwd_gidx, bwd_vals = fwd_idx[mask], fwd_gidx[mask], fwd_vals[mask]

        num_directions = [1] + [
            2 * parameters.ndim if complex_mode else len(windows[scale_idx])
            for scale_idx in range(1, parameters.num_scales)
        ]
        return cls(
            shape=tuple(parameters.shape),
            complex_output=complex_mode,
            groups=groups,
            layout=[entry.position for entry in entries],
            num_directions=num_directions,
            ncoef=offset,
            fwd_idx=fwd_idx,
            fwd_gidx=fwd_gidx,
            fwd_vals=fwd_vals,
            bwd_idx=bwd_idx,
            bwd_gidx=bwd_gidx,
            bwd_vals=bwd_vals,
        )

    # ---------------------------------------------------------------- helpers
    @property
    def ndim(self) -> int:
        """Number of transform dimensions."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Number of samples in one input."""
        return math.prod(self.shape)

    def _dims(self) -> tuple[int, ...]:
        return tuple(range(-self.ndim, 0))

    def _cached(
        self,
        key: str,
        dtype: torch.dtype,
        device: torch.device,
        make: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        cache_key = (key, dtype, device)
        out = self._cache.get(cache_key)
        if out is None:
            out = make()
            self._cache[cache_key] = out
        return out

    def _values(self, which: str, complex_dtype: torch.dtype) -> torch.Tensor:
        real = _REAL_DTYPE.get(complex_dtype, complex_dtype)
        src = self.fwd_vals if which == "fwd" else self.bwd_vals
        if src.dtype == real:
            return src
        return self._cached(f"{which}_vals", real, src.device, lambda: src.to(real))

    def _scale(
        self,
        group_idx: int,
        which: str,
        complex_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        group = self.groups[group_idx]
        real = _REAL_DTYPE.get(complex_dtype, complex_dtype)
        values = group.fwd_scale if which == "fwd" else group.bwd_scale
        return self._cached(
            f"{which}_scale_{group_idx}",
            real,
            device,
            lambda: torch.tensor(values, dtype=real, device=device).view(
                group.count, *([1] * self.ndim)
            ),
        )

    @staticmethod
    def _gather(flat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """``flat[:, index]`` for a ``(batch, n)`` tensor.

        A single row is gathered as a 1-D tensor, which is the fastest form;
        advanced indexing beats ``index_select(1, ...)`` for wider batches.
        """
        if flat.shape[0] == 1:
            return flat[0].index_select(0, index).unsqueeze(0)
        return flat[:, index]

    @staticmethod
    def _scatter(size: int, index: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """Sum ``samples`` (``(batch, nnz)``) into a zero ``(batch, size)`` tensor."""
        if samples.shape[0] == 1:
            out = torch.zeros(size, dtype=samples.dtype, device=samples.device)
            return out.index_add(0, index, samples[0]).unsqueeze(0)
        out = torch.zeros(
            samples.shape[0], size, dtype=samples.dtype, device=samples.device
        )
        return out.index_add(1, index, samples)

    # ---------------------------------------------------------------- forward
    def forward_groups(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward transform, one tensor per group.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape ``(*batch, *shape)``.

        Returns
        -------
        list[torch.Tensor]
            Group ``g`` has shape ``(*batch, g.count, *g.shape)``.
        """
        batch = image.shape[: image.ndim - self.ndim]
        spectrum = torch.fft.fftn(image, dim=self._dims())  # pylint: disable=not-callable
        flat = spectrum.reshape(-1, self.numel)
        cdtype = flat.dtype
        samples = self._gather(flat, self.fwd_idx) * self._values("fwd", cdtype)
        folded = self._scatter(self.ncoef, self.fwd_gidx, samples)
        out = []
        for group_idx, group in enumerate(self.groups):
            chunk = folded[:, group.offset : group.stop].reshape(
                -1, group.count, *group.shape
            )
            coeff = torch.fft.ifftn(chunk, dim=self._dims())  # pylint: disable=not-callable
            coeff = coeff * self._scale(group_idx, "fwd", cdtype, flat.device)
            out.append(coeff.reshape(*batch, group.count, *group.shape))
        return out

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
        parts = [g.reshape(*batch, -1) for g in self.forward_groups(image)]
        return torch.cat(parts, dim=-1)

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
        parts = []
        cdtype: torch.dtype | None = None
        for group_idx, group in enumerate(self.groups):
            chunk = flat[:, group.offset : group.stop].reshape(
                -1, group.count, *group.shape
            )
            spec = torch.fft.fftn(chunk, dim=self._dims())  # pylint: disable=not-callable
            cdtype = spec.dtype
            spec = spec * self._scale(group_idx, "bwd", cdtype, flat.device)
            parts.append(spec.reshape(flat.shape[0], -1))
        assert cdtype is not None
        small = torch.cat(parts, dim=1)
        samples = self._gather(small, self.bwd_gidx) * self._values("bwd", cdtype)
        spectrum = self._scatter(self.numel, self.bwd_idx, samples)
        image = torch.fft.ifftn(  # pylint: disable=not-callable
            spectrum.reshape(*batch, *self.shape), dim=self._dims()
        )
        return image if self.complex_output else image.real


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
