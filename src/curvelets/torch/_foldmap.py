"""Sparse reductions ("folds") used by the fused UDCT plan, one per device class.

A fold evaluates ``out[t] = sum_j w_j * src[s_j]`` with real weights for a fixed
list of contributions ``(t_j, s_j, w_j)``. :class:`CsrFold` also accepts a
weight *pair* ``(a_j, b_j)`` per contribution, which scales the real and
imaginary parts separately: ``a_j * Re(src[s_j]) + i * b_j * Im(src[s_j])``. With
``b_j = -a_j`` that is ``a_j * conj(src[s_j])``, so conjugated terms need no
separate conjugated copy of the input. The best way to do that depends on the
hardware, so there are two implementations:

- :class:`ScatterFold` (CPU): one ``index_add`` over the contributions sorted by
  target, so writes walk memory in order, applied one row at a time. It does no
  redundant arithmetic, which is what matters on a CPU. The plan builds one per
  cache-sized chunk of wedges (see ``UDCTPlan``).
- :class:`CsrFold` (GPUs and other accelerators): compressed sparse rows by
  target, evaluated by one fused Triton kernel launch per direction (no atomics,
  deterministic). Outside Triton or under functorch transforms (``vmap``), it
  falls back to PyTorch indexing. Automatic differentiation uses the transpose
  CSR fold as the adjoint.
"""

from __future__ import annotations

# pylint: disable=duplicate-code
from dataclasses import dataclass

import torch

from ._triton_fold import CsrFold, _functorch_wrapped, _index_dtype

__all__ = ["CsrFold", "ScatterFold"]


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
        Build the map from contribution lists (same contract as CsrFold.build).

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
        terms = row.index_select(0, self.sources) * self.weights
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
