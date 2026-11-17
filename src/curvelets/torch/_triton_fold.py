"""Fused Triton kernel for the accelerator fold (:class:`~._foldmap.GatherFold`).

The torch implementation of a gather fold runs one gather-multiply per level
(the dense level 0, a few sparse levels and a padded tail), each followed by an
``index_add`` into the output: 3 to 4 kernels and one temporary per level, and
the output is read and written once per level. That is ~5 kernels in the
forward and ~15 in the backward, and the backward is launch-bound for small
inputs and bandwidth-bound for large ones.

:class:`CsrFold` stores the same contributions in compressed sparse rows (CSR)
by target, and one kernel evaluates the whole fold: each program owns a tile of
targets and of batch rows, loops over its targets' contributions, and writes
every output element exactly once. So the fold is

- one launch per direction;
- deterministic, with no atomics and a fixed summation order per target;
- one pass over the output, and one pass over the index and weight tables per
  tile of up to :data:`MAX_BATCH_TILE` batch rows (the table entries are loaded
  once and reused for every row of the tile).

The kernel is used only where it is safe and useful: CUDA tensors (or CPU
tensors through Triton's interpreter, in the tests), outside autograd and
functorch transforms. It is registered as a ``torch.library.triton_op``, so
``torch.compile`` and CUDA graphs see through it. Everything else falls back to
the torch implementation, which is also used when Triton is not installed.
Set ``CURVELETS_TRITON=0`` to disable the kernel.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

MAX_BATCH_TILE = 8  # batch rows per program; the tables are reused across them
_TARGETS_PER_PROGRAM = 1024  # targets x batch rows handled by one program


@cache
def _triton_installed() -> bool:
    return importlib.util.find_spec("triton") is not None


def triton_available() -> bool:
    """Whether the fused fold kernel can be used at all (Triton importable, not disabled)."""
    if os.environ.get("CURVELETS_TRITON", "1").strip().lower() in {"0", "false", "off"}:
        return False
    return _triton_installed()


def _cpu_testing() -> bool:
    """CPU tensors through Triton's interpreter: for tests only.

    Needs both ``CURVELETS_TRITON_CPU=1`` and ``TRITON_INTERPRET=1``; the latter
    must be set before Triton is first imported (``tests/torch/conftest.py``).
    """
    return (
        os.environ.get("CURVELETS_TRITON_CPU", "0") == "1"
        and os.environ.get("TRITON_INTERPRET", "0") == "1"
    )


def triton_device(device: torch.device) -> bool:
    """Whether fold maps on ``device`` should carry a :class:`CsrFold`."""
    if not triton_available():
        return False
    return device.type == "cuda" or (device.type == "cpu" and _cpu_testing())


def _kernel() -> Any:
    """The jitted kernel (importing it imports Triton)."""
    from ._triton_kernels import (  # pylint: disable=import-outside-toplevel
        csr_fold_kernel,
    )

    return csr_fold_kernel


@cache
def _op():  # type: ignore[no-untyped-def]
    """Register the ``curvelets::csr_fold`` op on first use (imports Triton)."""
    import triton  # pylint: disable=import-outside-toplevel

    @torch.library.triton_op("curvelets::csr_fold", mutates_args={})  # type: ignore[misc]
    def csr_fold(
        src: torch.Tensor,
        row_ptr: torch.Tensor,
        col: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        """``(rows, num_sources)`` complex -> ``(rows, num_targets)`` complex."""
        src = src.contiguous()
        rows = src.shape[0]
        out = torch.empty(rows, num_targets, dtype=src.dtype, device=src.device)
        if rows == 0 or num_targets == 0:
            return out
        src_parts = torch.view_as_real(src)
        out_parts = torch.view_as_real(out)
        block_b = min(MAX_BATCH_TILE, triton.next_power_of_2(rows))
        block_t = max(32, _TARGETS_PER_PROGRAM // block_b)
        grid = (triton.cdiv(num_targets, block_t), triton.cdiv(rows, block_b))
        torch.library.wrap_triton(_kernel())[grid](
            src_parts,
            out_parts,
            row_ptr,
            col,
            weights,
            rows,
            num_targets,
            src_parts.stride(0),
            out_parts.stride(0),
            BLOCK_T=block_t,
            BLOCK_B=block_b,
            PAIR=weights.ndim == 2,
        )
        return out

    return csr_fold


@dataclass
class CsrFold:
    """A fold in compressed sparse rows by target, applied by one Triton kernel."""

    num_targets: int
    num_sources: int
    row_ptr: torch.Tensor  # (num_targets + 1,) offsets into col / weights
    col: torch.Tensor  # (nnz,) source of each contribution, grouped by target
    weights: torch.Tensor  # (nnz,) or (nnz, 2) real weights / weight pairs

    @classmethod
    def from_sorted(
        cls,
        targets: torch.Tensor,
        sources: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
        num_sources: int,
        index_dtype: torch.dtype,
    ) -> CsrFold:
        """Build from contributions already sorted by target (stable)."""
        counts = torch.bincount(targets, minlength=num_targets)
        row_ptr = torch.zeros(num_targets + 1, dtype=torch.int64, device=targets.device)
        torch.cumsum(counts, 0, out=row_ptr[1:])
        nnz_dtype = torch.int32 if targets.numel() < 2**31 - 1 else torch.int64
        return cls(
            num_targets=num_targets,
            num_sources=num_sources,
            row_ptr=row_ptr.to(nnz_dtype),
            col=sources.to(index_dtype),
            weights=weights.contiguous(),
        )

    def apply(self, src: torch.Tensor) -> torch.Tensor:
        """Apply to complex ``src`` of shape ``(rows, num_sources)``.

        The weights are cast to the working precision inside the kernel.
        """
        return _op()(src, self.row_ptr, self.col, self.weights, self.num_targets)

    def tensors(self) -> list[torch.Tensor]:
        """Index and weight tensors (for size accounting)."""
        return [self.row_ptr, self.col, self.weights]
