"""Fused Triton kernel and accelerator fold (:class:`CsrFold`).

:class:`CsrFold` stores the fold contributions in compressed sparse rows (CSR)
by target, and one fused Triton kernel evaluates the whole fold: each program
owns a tile of targets and of batch rows, loops over its targets' contributions,
and writes every output element exactly once. So the fold is

- one launch per direction;
- deterministic, with no atomics and a fixed summation order per target;
- one pass over the output, and one pass over the index and weight tables per
  tile of up to :data:`MAX_BATCH_TILE` batch rows (the table entries are loaded
  once and reused for every row of the tile).

The kernel is used where it is safe and useful: CUDA tensors (or CPU tensors
through Triton's interpreter, in the tests). Outside Triton or under functorch
transforms (``vmap``), it falls back to PyTorch indexing. Automatic
differentiation uses the transpose CSR fold as the adjoint.
Set ``CURVELETS_TRITON=0`` to disable the kernel.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

MAX_BATCH_TILE = 8  # batch rows per program; the tables are reused across them
_TARGETS_PER_PROGRAM = 1024  # targets x batch rows handled by one program


def _index_dtype(size: int) -> torch.dtype:
    """int32 indices halve index traffic; fall back to int64 for huge grids."""
    return torch.int32 if size < 2**31 - 1 else torch.int64


def _functorch_wrapped(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is wrapped by a functorch transform (vmap, grad, ...).

    Always ``False`` while TorchDynamo traces (``torch.compile``): the check is an
    opaque builtin that would break the graph, and traced code takes the
    functional paths anyway.
    """
    if torch.compiler.is_compiling():
        return False
    is_wrapped = getattr(torch._C._functorch, "is_functorch_wrapped_tensor", None)  # pylint: disable=protected-access
    return is_wrapped is not None and bool(is_wrapped(tensor))


_TRITON_INSTALLED: bool = importlib.util.find_spec("triton") is not None


def triton_available() -> bool:
    """Whether the fused fold kernel can be used at all (Triton importable, not disabled)."""
    if os.environ.get("CURVELETS_TRITON", "1").strip().lower() in {"0", "false", "off"}:
        return False
    return _TRITON_INSTALLED


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
    """Whether fold maps on ``device`` should run the Triton kernel."""
    if not triton_available():
        return False
    return device.type == "cuda" or (device.type == "cpu" and _cpu_testing())


def _direct(kernel: Any) -> Any:
    """Launch a kernel as is (eager calls)."""
    return kernel


def _kernel() -> Any:
    """The jitted kernel (importing it imports Triton)."""
    from ._triton_kernels import (  # pylint: disable=import-outside-toplevel
        csr_fold_kernel,
    )

    return csr_fold_kernel


_OP_REGISTERED = False


def _register_op() -> None:
    """Register ``curvelets::csr_fold`` once (eagerly, when CSR maps are built).

    Calls then go through ``torch.ops.curvelets.csr_fold``, which TorchDynamo
    traces natively; it cannot trace into a cached registration helper.
    """
    global _OP_REGISTERED  # noqa: PLW0603  # pylint: disable=global-statement
    if not _OP_REGISTERED:
        _op()
        _OP_REGISTERED = True


def _fold(
    launch: Callable[[Any], Any],
    src: torch.Tensor,
    row_ptr: torch.Tensor,
    col: torch.Tensor,
    weights: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """Allocate the output and launch the kernel through ``launch(kernel)``."""
    src = src.contiguous()
    rows = src.shape[0]
    out = torch.empty(rows, num_targets, dtype=src.dtype, device=src.device)
    if rows == 0 or num_targets == 0:
        return out
    src_parts = torch.view_as_real(src)
    out_parts = torch.view_as_real(out)
    block_b = min(MAX_BATCH_TILE, 1 << (rows - 1).bit_length())  # next power of 2
    block_t = max(32, _TARGETS_PER_PROGRAM // block_b)
    grid = (-(-num_targets // block_t), -(-rows // block_b))
    launch(_kernel())[grid](
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
        # Explicit (Triton 3.9 warns when unset): fused multiply-adds for
        # ``acc += x * w``, Triton's default.
        enable_fp_fusion=True,
    )
    return out


def _op():  # type: ignore[no-untyped-def]
    """Register the ``curvelets::csr_fold`` op (for ``torch.compile``)."""

    @torch.library.triton_op("curvelets::csr_fold", mutates_args={})  # type: ignore[untyped-decorator]
    def csr_fold(
        src: torch.Tensor,
        row_ptr: torch.Tensor,
        col: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        """``(rows, num_sources)`` complex -> ``(rows, num_targets)`` complex."""
        return _fold(torch.library.wrap_triton, src, row_ptr, col, weights, num_targets)

    return csr_fold


class _CsrFoldFunction(torch.autograd.Function):  # type: ignore[misc]  # pylint: disable=abstract-method
    """Autograd wrapper for CsrFold using the transpose fold as the adjoint."""

    @staticmethod
    def forward(ctx: Any, src: torch.Tensor, csr: CsrFold) -> torch.Tensor:  # pylint: disable=arguments-differ
        ctx.csr = csr
        return csr._apply_triton(src)  # pylint: disable=protected-access

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        csr: CsrFold = ctx.csr
        grad_src = csr.get_transpose()._apply_triton(  # pylint: disable=protected-access
            grad_outputs[0].contiguous()
        )
        return grad_src, None


@dataclass
class CsrFold:
    """A fold in compressed sparse rows by target.

    Applied by one Triton kernel where available (CUDA), or via PyTorch
    indexing where not (non-Triton devices, functorch transforms).
    """

    num_targets: int
    num_sources: int
    row_ptr: torch.Tensor  # (num_targets + 1,) offsets into col / weights
    col: torch.Tensor  # (nnz,) source of each contribution, grouped by target
    weights: torch.Tensor  # (nnz,) or (nnz, 2) real weights / weight pairs
    _targets: torch.Tensor | None = None
    _transpose: CsrFold | None = None

    @classmethod
    def build(
        cls,
        targets: torch.Tensor,
        sources: torch.Tensor,
        weights: torch.Tensor,
        num_targets: int,
        num_sources: int,
    ) -> CsrFold:
        """Build the map from contribution lists.

        Parameters
        ----------
        targets, sources : torch.Tensor
            Integer tensors of equal length: contribution ``j`` adds
            ``weights[j] * src[sources[j]]`` to ``out[targets[j]]``.
        weights : torch.Tensor
            Real weights, shape ``(n,)``, or weight pairs, shape ``(n, 2)``, for
            the real and imaginary parts of each source.
        num_targets, num_sources : int
            Output and input lengths.

        Returns
        -------
        CsrFold
            The map, on the device of ``targets``.
        """
        order = torch.argsort(targets.to(torch.int64), stable=True)
        t_sorted = targets.to(torch.int64)[order]
        s_sorted = sources.to(torch.int64)[order]
        w_sorted = weights[order]
        return cls.from_sorted(
            t_sorted,
            s_sorted,
            w_sorted,
            num_targets,
            num_sources,
            _index_dtype(num_sources),
        )

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
        if triton_device(targets.device):
            _register_op()
        return cls(
            num_targets=num_targets,
            num_sources=num_sources,
            row_ptr=row_ptr.to(nnz_dtype),
            col=sources.to(index_dtype),
            weights=weights.contiguous(),
            _targets=targets.to(torch.int64),
        )

    def transpose(self) -> CsrFold:
        """The transpose fold: (batch, num_targets) -> (batch, num_sources).

        Used by autograd: the adjoint of ``out[t] = sum_j w_j * src[s_j]`` is
        ``grad_src[s] = sum_j w_j * grad_out[t_j]``.
        """
        if self._targets is not None:
            targets_expanded = self._targets
        else:
            row_ptr = self.row_ptr.to(torch.int64)
            counts = row_ptr[1:] - row_ptr[:-1]
            targets = torch.arange(
                self.num_targets, device=row_ptr.device, dtype=self.col.dtype
            )
            targets_expanded = torch.repeat_interleave(targets, counts)
        order = torch.argsort(self.col, stable=True)
        return self.from_sorted(
            self.col[order],
            targets_expanded[order],
            self.weights[order],
            self.num_sources,
            self.num_targets,
            self.col.dtype,
        )

    def get_transpose(self) -> CsrFold:
        """Return the cached transpose fold, building it on first use."""
        if self._transpose is None:
            self._transpose = self.transpose()
        return self._transpose

    def _apply_torch(self, src: torch.Tensor) -> torch.Tensor:
        """Apply via PyTorch indexing (for non-Triton devices and functorch)."""
        parts = torch.view_as_real(src)
        w = self.weights if self.weights.ndim == 2 else self.weights.unsqueeze(-1)
        terms = parts.index_select(1, self.col.to(torch.int64)) * w
        if self._targets is None:
            row_ptr = self.row_ptr.to(torch.int64)
            counts = row_ptr[1:] - row_ptr[:-1]
            self._targets = torch.repeat_interleave(
                torch.arange(self.num_targets, device=row_ptr.device), counts
            ).to(torch.int64)
        out = torch.zeros(
            parts.shape[0], self.num_targets, 2, dtype=parts.dtype, device=parts.device
        )
        if _functorch_wrapped(src):
            out = out.index_add(1, self._targets, terms)
        else:
            out.index_add_(1, self._targets, terms)
        return torch.view_as_complex(out)

    def _apply_triton(self, src: torch.Tensor) -> torch.Tensor:
        """Launch the fused Triton kernel or trace the registered op."""
        args = (src, self.row_ptr, self.col, self.weights, self.num_targets)
        if torch.compiler.is_compiling():
            if not _OP_REGISTERED:  # e.g. unpickled maps; a constant once registered
                _register_op()
            return torch.ops.curvelets.csr_fold(*args)
        return _fold(_direct, *args)

    def apply(self, src: torch.Tensor) -> torch.Tensor:
        """Apply the fold to complex ``src`` of shape ``(rows, num_sources)``.

        Parameters
        ----------
        src : torch.Tensor
            Complex input.

        Returns
        -------
        torch.Tensor
            Complex output of shape ``(rows, num_targets)``.
        """
        if not triton_device(src.device) or _functorch_wrapped(src):
            return self._apply_torch(src)
        if src.requires_grad and torch.is_grad_enabled():
            if torch.compiler.is_compiling():
                return self._apply_torch(src)
            return _CsrFoldFunction.apply(src, self)
        return self._apply_triton(src)

    def tensors(self) -> list[torch.Tensor]:
        """Index and weight tensors (for size accounting)."""
        return [self.row_ptr, self.col, self.weights]
