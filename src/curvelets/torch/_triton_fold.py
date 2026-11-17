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

The kernel is used only where it is safe and useful: CUDA tensors (or any device
under ``TRITON_INTERPRET=1``, for testing), outside autograd and functorch
transforms. It is registered as a ``torch.library.triton_op``, so
``torch.compile`` and CUDA graphs see through it. Everything else falls back to
the torch implementation, which is also used when Triton is not installed.
Set ``CURVELETS_TRITON=0`` to disable the kernel.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import cache

import torch

MAX_BATCH_TILE = 8  # batch rows per program; the tables are reused across them
_TARGETS_PER_PROGRAM = 1024  # targets x batch rows handled by one program


@cache
def triton_available() -> bool:
    """Whether the fused fold kernel can be used at all (Triton importable, not disabled)."""
    if os.environ.get("CURVELETS_TRITON", "1").strip().lower() in {"0", "false", "off"}:
        return False
    return importlib.util.find_spec("triton") is not None


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def triton_device(device: torch.device) -> bool:
    """Whether fold maps on ``device`` should carry a :class:`CsrFold`."""
    if not triton_available():
        return False
    return device.type == "cuda" or _interpreting()


@cache
def _op():  # type: ignore[no-untyped-def]  # noqa: ANN202
    """Build the kernel and its ``torch.library`` op on first use (imports Triton)."""
    import triton  # pylint: disable=import-outside-toplevel
    import triton.language as tl  # pylint: disable=import-outside-toplevel

    @triton.jit
    def _csr_fold_kernel(  # noqa: PLR0913
        src_ptr,
        out_ptr,
        row_ptr,
        col_ptr,
        w_ptr,
        rows,
        num_targets,
        src_row_stride,
        out_row_stride,
        BLOCK_T: tl.constexpr,  # noqa: N803
        BLOCK_B: tl.constexpr,  # noqa: N803
        PAIR: tl.constexpr,  # noqa: N803
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t < num_targets
        start = tl.load(row_ptr + t, mask=t_mask, other=0)
        stop = tl.load(row_ptr + t + 1, mask=t_mask, other=0)
        count = stop - start
        b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b < rows
        src_rows = src_ptr + b.to(tl.int64)[:, None] * src_row_stride
        acc_dtype = src_ptr.dtype.element_ty
        acc_re = tl.zeros((BLOCK_B, BLOCK_T), dtype=acc_dtype)
        acc_im = tl.zeros((BLOCK_B, BLOCK_T), dtype=acc_dtype)
        for k in range(0, tl.max(count, axis=0)):
            m = t_mask & (k < count)
            j = start + k
            s = tl.load(col_ptr + j, mask=m, other=0).to(tl.int64)
            if PAIR:
                w_re = tl.load(w_ptr + 2 * j, mask=m, other=0.0).to(acc_dtype)
                w_im = tl.load(w_ptr + 2 * j + 1, mask=m, other=0.0).to(acc_dtype)
            else:
                w_re = tl.load(w_ptr + j, mask=m, other=0.0).to(acc_dtype)
                w_im = w_re
            mm = b_mask[:, None] & m[None, :]
            p = src_rows + 2 * s[None, :]
            acc_re += tl.load(p, mask=mm, other=0.0) * w_re[None, :]
            acc_im += tl.load(p + 1, mask=mm, other=0.0) * w_im[None, :]
        o = out_ptr + b.to(tl.int64)[:, None] * out_row_stride + 2 * t[None, :]
        om = b_mask[:, None] & t_mask[None, :]
        tl.store(o, acc_re, mask=om)
        tl.store(o + 1, acc_im, mask=om)

    @torch.library.triton_op("curvelets::csr_fold", mutates_args={})
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
        torch.library.wrap_triton(_csr_fold_kernel)[grid](
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
