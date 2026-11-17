"""Triton kernels for :mod:`curvelets.torch._triton_fold` (imported only with Triton)."""

# mypy: disable-error-code="misc, no-untyped-def"
# ruff: noqa: N803  (Triton constexpr parameters are upper case by convention)
# pylint: disable=invalid-name
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def csr_fold_kernel(
    src_ptr,
    out_ptr,
    row_ptr,
    col_ptr,
    w_ptr,
    rows,
    num_targets,
    src_row_stride,
    out_row_stride,
    BLOCK_T: tl.constexpr,
    BLOCK_B: tl.constexpr,
    PAIR: tl.constexpr,
):
    """``out[b, t] = sum_k w[k] * src[b, col[k]]`` over ``k`` in ``row_ptr[t]:row_ptr[t+1]``.

    Complex values are interleaved real pairs; ``PAIR`` weights scale the real and
    imaginary parts separately. Each program owns ``BLOCK_B`` rows x ``BLOCK_T``
    targets and loads every table entry once for all of its rows.
    """
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
    for k in range(tl.max(count, axis=0)):
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
