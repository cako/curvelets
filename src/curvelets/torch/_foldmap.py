"""Sparse reductions ("folds") used by the fused UDCT plan, one per device class.

A fold evaluates ``out[t] = sum_j w_j * src[s_j]`` with real weights for a fixed
list of contributions ``(t_j, s_j, w_j)``. The best way to do that depends on the
hardware, so there are two implementations:

- :class:`ScatterFold` (CPU): one ``index_add`` over the contributions sorted by
  target, so writes walk memory in order, applied one row at a time. It does no
  redundant arithmetic, which is what matters on a CPU. The plan builds one per
  cache-sized chunk of wedges (see ``UDCTPlan``).
- :class:`GatherFold` (GPUs and other accelerators): trades extra arithmetic for
  fewer launches and no atomics, as described below.

GatherFold: instead of a scatter-add (``index_add`` over every contribution, which needs
atomics on GPU, a zero-filled output, and gives run-to-run differences in
floating-point summation order), the contributions are split into *levels*:
level ``l`` holds the ``l``-th contribution of every target, so each level has
unique targets.

- Level 0 is stored densely over all targets: one gather writes the whole
  output (empty targets read element 0 with weight 0), so there is no zero fill.
- Higher levels are sparse and usually small. Their targets are unique, so the
  ``index_add`` that applies them never collides and is deterministic.
- 3D windows overlap more, so a target can have a dozen or more contributions
  with a long thin tail. After :data:`MAX_LEVELS` explicit levels, the remaining
  contributions are packed into one zero-padded ``(targets, width)`` block that
  is reduced with a single ``sum``; it keeps the launch count bounded.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
from dataclasses import dataclass, field

import torch

from ._sparse_window import _REAL_DTYPE

MAX_LEVELS = 3  # explicit sparse levels before the padded tail block


def _index_dtype(size: int) -> torch.dtype:
    """int32 indices halve index traffic; fall back to int64 for huge grids."""
    return torch.int32 if size < 2**31 - 1 else torch.int64


def _functorch_wrapped(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is wrapped by a functorch transform (vmap, grad, ...)."""
    is_wrapped = getattr(torch._C._functorch, "is_functorch_wrapped_tensor", None)  # pylint: disable=protected-access
    return is_wrapped is not None and bool(is_wrapped(tensor))


def _ranks(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable sort order of ``targets`` and each element's rank among equals."""
    order = torch.argsort(targets, stable=True)
    sorted_t = targets[order]
    n = sorted_t.numel()
    is_start = torch.ones(n, dtype=torch.bool, device=targets.device)
    if n > 1:
        is_start[1:] = sorted_t[1:] != sorted_t[:-1]
    positions = torch.arange(n, device=targets.device)
    run_start = torch.where(is_start, positions, torch.zeros_like(positions))
    run_start = torch.cummax(run_start, dim=0).values
    return order, positions - run_start


@dataclass
class _Level:
    """Unique ``targets``; each gathers ``width`` sources (zero-padded) and sums."""

    targets: torch.Tensor
    sources: torch.Tensor  # (len(targets) * width,)
    weights: torch.Tensor
    width: int = 1


@dataclass
class GatherFold:
    """
    Gather-only fold for accelerators: ``(batch, num_sources) -> (batch, num_targets)``.

    Build it with :meth:`build`; apply it with :meth:`apply`.
    """

    num_targets: int
    num_sources: int
    src0: torch.Tensor
    w0: torch.Tensor
    levels: list[_Level]
    _cast: dict[tuple[int, torch.dtype], torch.Tensor] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def build(
        cls,
        targets: torch.Tensor,
        sources: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
        num_sources: int,
    ) -> GatherFold:
        """
        Build the map from contribution lists.

        Parameters
        ----------
        targets, sources : torch.Tensor
            Integer tensors of equal length: contribution ``j`` adds
            ``weights[j] * src[sources[j]]`` to ``out[targets[j]]``.
        weights : torch.Tensor
            Real weights.
        num_targets, num_sources : int
            Output and input lengths.

        Returns
        -------
        GatherFold
            The map, on the device of ``targets``.
        """
        device = targets.device
        order, rank = _ranks(targets.to(torch.int64))
        t_sorted = targets.to(torch.int64)[order]
        s_sorted = sources.to(torch.int64)[order]
        w_sorted = weights[order]
        idx_t = _index_dtype(num_targets)
        idx_s = _index_dtype(num_sources)

        first = rank == 0
        src0 = torch.zeros(num_targets, dtype=idx_s, device=device)
        w0 = torch.zeros(num_targets, dtype=weights.dtype, device=device)
        src0[t_sorted[first]] = s_sorted[first].to(idx_s)
        w0[t_sorted[first]] = w_sorted[first]

        levels = []
        num_levels = int(rank.max().item()) + 1 if rank.numel() else 1
        explicit = min(num_levels - 1, MAX_LEVELS)
        for level in range(1, explicit + 1):
            sel = rank == level
            levels.append(
                _Level(
                    targets=t_sorted[sel].to(idx_t),
                    sources=s_sorted[sel].to(idx_s),
                    weights=w_sorted[sel],
                )
            )
        if num_levels - 1 > explicit:
            first_tail = explicit + 1
            tail_targets = t_sorted[rank == first_tail]  # sorted, unique
            sel = rank >= first_tail
            width = num_levels - first_tail
            rows = torch.searchsorted(tail_targets, t_sorted[sel])
            flat = rows * width + (rank[sel] - first_tail)
            size = tail_targets.numel() * width
            tail_src = torch.zeros(size, dtype=idx_s, device=device)
            tail_w = torch.zeros(size, dtype=weights.dtype, device=device)
            tail_src[flat] = s_sorted[sel].to(idx_s)
            tail_w[flat] = w_sorted[sel]
            levels.append(
                _Level(
                    targets=tail_targets.to(idx_t),
                    sources=tail_src,
                    weights=tail_w,
                    width=width,
                )
            )
        return cls(
            num_targets=num_targets,
            num_sources=num_sources,
            src0=src0,
            w0=w0,
            levels=levels,
        )

    def _weights(
        self, slot: int, tensor: torch.Tensor, real: torch.dtype
    ) -> torch.Tensor:
        if tensor.dtype == real:
            return tensor
        key = (slot, real)
        cast = self._cast.get(key)
        if cast is None:
            cast = tensor.to(real)
            self._cast[key] = cast
        return cast

    def apply(self, src: torch.Tensor) -> torch.Tensor:
        """
        Apply the map to ``src`` of shape ``(batch, num_sources)``.

        Parameters
        ----------
        src : torch.Tensor
            Complex input.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, num_targets)``.
        """
        real = _REAL_DTYPE.get(src.dtype, src.dtype)
        single = src.shape[0] == 1
        row = src[0] if single else src

        def take(index: torch.Tensor) -> torch.Tensor:
            return row.index_select(0, index) if single else row[:, index]

        out = take(self.src0) * self._weights(0, self.w0, real)
        dim = 0 if single else 1
        for slot, lvl in enumerate(self.levels, start=1):
            term = take(lvl.sources) * self._weights(slot, lvl.weights, real)
            if lvl.width > 1:
                term = term.unflatten(-1, (-1, lvl.width)).sum(-1)
            # ``out`` is a fresh tensor owned by this call, so the level is added
            # in place (an out-of-place index_add would copy the whole output).
            out.index_add_(dim, lvl.targets, term)
        return out.unsqueeze(0) if single else out


@dataclass
class ScatterFold:
    """
    Scatter-add fold for CPUs: ``(batch, num_sources) -> (batch, num_targets)``.

    Contributions are sorted by target once, at build time, so the ``index_add``
    writes sequentially; every contribution is touched exactly once.
    """

    num_targets: int
    num_sources: int
    targets: torch.Tensor
    sources: torch.Tensor
    weights: torch.Tensor
    _cast: dict[torch.dtype, torch.Tensor] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        targets: torch.Tensor,
        sources: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
        num_sources: int,
    ) -> ScatterFold:
        """
        Build the map from contribution lists (same contract as GatherFold.build).

        Parameters
        ----------
        targets, sources : torch.Tensor
            Integer tensors of equal length: contribution ``j`` adds
            ``weights[j] * src[sources[j]]`` to ``out[targets[j]]``.
        weights : torch.Tensor
            Real weights.
        num_targets, num_sources : int
            Output and input lengths.

        Returns
        -------
        ScatterFold
            The map, on the device of ``targets``.
        """
        order = torch.argsort(targets.to(torch.int64), stable=True)
        return cls(
            num_targets=num_targets,
            num_sources=num_sources,
            targets=targets.to(torch.int64)[order].to(_index_dtype(num_targets)),
            sources=sources.to(torch.int64)[order].to(_index_dtype(num_sources)),
            weights=weights[order],
        )

    def _terms(self, row: torch.Tensor) -> torch.Tensor:
        real = _REAL_DTYPE.get(row.dtype, row.dtype)
        weights = self.weights
        if weights.dtype != real:
            cached = self._cast.get(real)
            if cached is None:
                cached = self.weights.to(real)
                self._cast[real] = cached
            weights = cached
        return row.index_select(0, self.sources) * weights

    def add_row(self, out: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        """
        Add this fold of the 1-D ``row`` into the 1-D ``out`` and return it.

        ``out`` is updated in place unless a functorch transform (vmap, grad) is
        active, in which case a new tensor is returned.

        Parameters
        ----------
        out : torch.Tensor
            Accumulator of length ``num_targets``.
        row : torch.Tensor
            Input of length ``num_sources``.

        Returns
        -------
        torch.Tensor
            The accumulator.
        """
        terms = self._terms(row)
        if _functorch_wrapped(row) or _functorch_wrapped(out):
            return out.index_add(0, self.targets, terms)
        return out.index_add_(0, self.targets, terms)

    def apply(self, src: torch.Tensor) -> torch.Tensor:
        """
        Apply the map to ``src`` of shape ``(batch, num_sources)``, row by row.

        Parameters
        ----------
        src : torch.Tensor
            Complex input.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, num_targets)``.
        """
        if _functorch_wrapped(src):
            # vmap/grad cannot write batched values into a plain buffer in place.
            zeros = torch.zeros(self.num_targets, dtype=src.dtype, device=src.device)
            return torch.stack([self.add_row(zeros, row) for row in src])
        out = torch.zeros(
            src.shape[0], self.num_targets, dtype=src.dtype, device=src.device
        )
        for row in range(src.shape[0]):
            self.add_row(out[row], src[row])
        return out
