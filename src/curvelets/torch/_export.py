"""Plan export utilities for loading a UDCT in Python-free C++ (libtorch)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ._monogenic import MonogenicLayout, riesz_spectra
from ._plan import UDCTPlan
from ._triton_fold import CsrFold

if TYPE_CHECKING:
    from ._udct import UDCT

_KIND_TO_CODE: dict[str, int] = {"real": 0, "complex": 1, "monogenic": 2}


def export_udct_plan_dict(udct: UDCT) -> dict[str, torch.Tensor]:
    """
    Serialize a :class:`~curvelets.torch.UDCT` into a flat dictionary of tensors.

    The returned dictionary can be saved with :func:`torch.save` and loaded
    directly in Python-free C++ via ``curvelets::UDCT`` (``libtorch``).
    """
    # pylint: disable=protected-access
    params = udct._params
    windows = udct._windows
    decimation_ratios = udct._decimation_ratios_cpu
    kind = udct._transform_kind
    # pylint: enable=protected-access

    device = windows[0][0][0].values.device
    dtype = udct.dtype
    plan = UDCTPlan.build(
        params,
        windows,
        decimation_ratios,
        complex_mode=kind == "complex",
        fold_strategy="csr",
    )
    assert isinstance(plan.fwd_map, CsrFold)
    assert plan.bwd_map is not None

    group_offsets = [g.offset for g in plan.groups]
    group_counts = [g.count for g in plan.groups]
    group_shapes = [dim for g in plan.groups for dim in g.shape]
    kind_code = _KIND_TO_CODE[kind]

    if kind == "monogenic":
        layout = MonogenicLayout.build(plan, params.ndim + 2).to(device)
        # Store Riesz filters as a real (..., 2) float tensor so .to(dtype) on
        # C++ modules preserves imaginary components.
        riesz_filters = torch.view_as_real(
            riesz_spectra(params.shape, dtype, device)
        ).contiguous()
        monogenic_perm = layout.perm.contiguous()
        monogenic_inverse = layout.inverse.contiguous()
        even_axes = [k for k, n in enumerate(params.shape) if n % 2 == 0]
    else:
        riesz_filters = torch.empty((0,), dtype=dtype, device=device)
        monogenic_perm = torch.empty((0,), dtype=torch.int64, device=device)
        monogenic_inverse = torch.empty((0,), dtype=torch.int64, device=device)
        even_axes = []

    return {
        "meta": torch.tensor(
            [1, plan.ncoef, int(plan.complex_output), kind_code], dtype=torch.int64
        ),
        "shape": torch.tensor(list(plan.shape), dtype=torch.int64),
        "group_offsets": torch.tensor(group_offsets, dtype=torch.int64),
        "group_counts": torch.tensor(group_counts, dtype=torch.int64),
        "group_shapes": torch.tensor(group_shapes, dtype=torch.int64),
        "num_directions": torch.tensor(plan.num_directions, dtype=torch.int64),
        "layout": torch.tensor(plan.layout, dtype=torch.int64),
        "even_axes": torch.tensor(even_axes, dtype=torch.int64),
        "fwd_row_ptr": plan.fwd_map.row_ptr.detach().contiguous(),
        "fwd_col": plan.fwd_map.col.detach().contiguous(),
        "fwd_weights": plan.fwd_map.weights.detach().contiguous(),
        "bwd_row_ptr": plan.bwd_map.row_ptr.detach().contiguous(),
        "bwd_col": plan.bwd_map.col.detach().contiguous(),
        "bwd_weights": plan.bwd_map.weights.detach().contiguous(),
        "riesz_filters": riesz_filters.detach(),
        "monogenic_perm": monogenic_perm.detach(),
        "monogenic_inverse": monogenic_inverse.detach(),
    }


def save_udct_plan(udct: UDCT, path: str | Path) -> Path:
    """
    Export a :class:`~curvelets.torch.UDCT` plan to a ``.pt`` file for C++ ``libtorch``.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan_dict = {
        k: v.cpu().contiguous() for k, v in export_udct_plan_dict(udct).items()
    }
    torch.save(plan_dict, out_path)
    return out_path
