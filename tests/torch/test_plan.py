"""Tests for the fused execution plan (curvelets.torch._plan)."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from curvelets.torch import UDCT, UDCTModule
from curvelets.torch._backward_transform import _apply_backward_transform
from curvelets.torch._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_real,
)

Kind = Literal["real", "complex"]
Mode = Literal["curvelet", "wavelet"]

CONFIGS = [
    ((32, 32), 3, 3),
    ((48, 40), 3, 6),
    ((64, 64), 4, 6),
    ((16, 16, 16), 3, 3),
]


def _input(kind: Kind, shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn(shape, generator=gen, dtype=torch.float64))
    return x


@pytest.mark.parametrize(("shape", "num_scales", "wpd"), CONFIGS)  # type: ignore[misc]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
@pytest.mark.parametrize("mode", ["curvelet", "wavelet"])  # type: ignore[misc]
def test_plan_matches_per_wedge_reference(
    shape: tuple[int, ...], num_scales: int, wpd: int, kind: Kind, mode: Mode
) -> None:
    """Fused forward/backward equal the per-wedge reference implementation."""
    udct = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=wpd,
        transform_kind=kind,
        high_frequency_mode=mode,
    )
    params = udct._parameters  # pylint: disable=protected-access
    decim = udct._decimation_ratios_cpu  # pylint: disable=protected-access
    x = _input(kind, shape)

    reference_fwd = (
        _apply_forward_transform_complex
        if kind == "complex"
        else _apply_forward_transform_real
    )
    expected = reference_fwd(x, params, udct.windows, decim)
    got = udct.forward(x)
    for s_exp, s_got in zip(expected, got, strict=True):
        for d_exp, d_got in zip(s_exp, s_got, strict=True):
            for w_exp, w_got in zip(d_exp, d_got, strict=True):
                torch.testing.assert_close(w_got, w_exp, rtol=1e-12, atol=1e-12)

    expected_bwd = _apply_backward_transform(
        expected, params, udct.windows, decim, use_complex_transform=kind == "complex"
    )
    torch.testing.assert_close(udct.backward(got), expected_bwd, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[misc]
def test_plan_batched_matches_single(kind: Kind) -> None:
    """Leading batch dimensions give the same result as a Python loop."""
    udct = UDCT(
        shape=(32, 32), num_scales=3, wedges_per_direction=3, transform_kind=kind
    )
    plan = udct._get_plan()  # pylint: disable=protected-access
    batch = torch.stack([_input(kind, (32, 32), seed) for seed in range(3)]).reshape(
        3, 1, 32, 32
    )
    flat = plan.forward_flat(batch)
    assert flat.shape == (3, 1, plan.ncoef)
    for i in range(3):
        torch.testing.assert_close(flat[i, 0], udct.vect(udct.forward(batch[i, 0])))
    recon = plan.backward_flat(flat)
    torch.testing.assert_close(recon, batch)


def test_vect_fast_path_matches_vect_forward() -> None:
    """``UDCT._forward_vect`` equals ``vect(forward(x))``."""
    udct = UDCT(shape=(48, 40), num_scales=3, wedges_per_direction=6)
    x = _input("real", (48, 40))
    torch.testing.assert_close(
        udct._forward_vect(x),  # pylint: disable=protected-access
        udct.vect(udct.forward(x)),
    )


def test_udct_module_vmap_matches_loop() -> None:
    """``torch.vmap`` over ``UDCTModule`` still works with the fused plan."""
    module = UDCTModule(shape=(28, 28), num_scales=2, wedges_per_direction=3)
    batch = torch.randn(4, 28, 28, dtype=torch.float64)
    batched = torch.vmap(module)(batch)
    for i in range(4):
        torch.testing.assert_close(batched[i], module(batch[i]))


@pytest.mark.parametrize("wpd", [3, 6])  # type: ignore[misc]
def test_forward_launch_count_does_not_grow_with_wedges(wpd: int) -> None:
    """One fold and one gather per forward, regardless of the number of wedges."""
    udct = UDCT(shape=(64, 64), num_scales=4, wedges_per_direction=wpd)
    x = _input("real", (64, 64))
    udct.forward(x)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        udct.forward(x)
    names = [event.name for event in prof.events()]
    assert names.count("aten::index_add") == 1
    assert names.count("aten::index_select") == 1
