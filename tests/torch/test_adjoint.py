"""Adjoint, Parseval tight-frame, and gradcheck tests for PyTorch UDCT."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
import torch

from curvelets.torch import UDCT
from curvelets.torch._plan import UDCTPlan

Mode = Literal["curvelet", "wavelet"]
Kind = Literal["real", "complex"]
AllKind = Literal["real", "complex", "monogenic"]


def _randn_like_kind(
    shape: tuple[int, ...],
    *,
    is_complex: bool,
    gen: torch.Generator,
) -> torch.Tensor:
    real = torch.randn(shape, generator=gen, dtype=torch.float64)
    if not is_complex:
        return real
    imag = torch.randn(shape, generator=gen, dtype=torch.float64)
    return torch.complex(real, imag)


def _directional_grad_error(
    fn: Callable[[torch.Tensor], torch.Tensor],
    inp: torch.Tensor,
    *,
    gen: torch.Generator,
) -> float:
    inp_req = inp.detach().clone().requires_grad_(True)
    out = fn(inp_req)
    g = _randn_like_kind(tuple(out.shape), is_complex=out.is_complex(), gen=gen)
    loss = (out.conj() * g).real.sum() if out.is_complex() else (out * g).sum()
    (grad_inp,) = torch.autograd.grad(loss, inp_req)

    dinp = _randn_like_kind(tuple(inp.shape), is_complex=inp.is_complex(), gen=gen)
    eps = 1e-6
    with torch.no_grad():
        out_p = fn(inp.detach() + eps * dinp)
        out_m = fn(inp.detach() - eps * dinp)
        diff = out_p - out_m
        df_fd_tensor = (
            (diff.conj() * g).real.sum() if out.is_complex() else (diff * g).sum()
        )
        df_fd = float(df_fd_tensor.item()) / (2.0 * eps)
    df_an_tensor = (
        (dinp.conj() * grad_inp).real.sum()
        if inp.is_complex()
        else (dinp * grad_inp).sum()
    )
    df_an = float(df_an_tensor.item())
    return abs(df_fd - df_an) / max(abs(df_fd), abs(df_an), 1e-30)


@pytest.mark.parametrize("shape", [(32, 32), (16, 16, 16)])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("num_scales", [2, 3])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("high_frequency_mode", ["curvelet", "wavelet"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("transform_kind", ["real", "complex"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("strategy", ["scatter", "csr"])  # type: ignore[untyped-decorator]
def test_torch_adjoint_and_parseval(
    shape: tuple[int, ...],
    num_scales: int,
    high_frequency_mode: Mode,
    transform_kind: Kind,
    strategy: str,
) -> None:
    """Forward and backward form an exact tight frame and adjoint pair."""
    gen = torch.Generator().manual_seed(0)
    udct = UDCT(
        shape=shape,
        num_scales=num_scales,
        wedges_per_direction=3,
        high_frequency_mode=high_frequency_mode,
        transform_kind=transform_kind,
        dtype=torch.float64,
    )
    udct._plan = UDCTPlan.build(  # pylint: disable=protected-access
        udct._params,  # pylint: disable=protected-access
        udct.windows,
        udct._decimation_ratios_cpu,  # pylint: disable=protected-access
        complex_mode=transform_kind == "complex",
        fold_strategy=strategy,
    )

    x = _randn_like_kind(shape, is_complex=transform_kind == "complex", gen=gen)
    u_x = udct.forward(x, structured=False)
    recon = udct.backward(u_x)
    assert float((recon - x).norm().item()) / float(x.norm().item()) < 1e-12

    energy_x = float(x.abs().pow(2).sum().item())
    energy_u_x = float(u_x.abs().pow(2).sum().item())
    assert abs(energy_u_x - energy_x) / energy_x < 1e-12

    y = _randn_like_kind(tuple(u_x.shape), is_complex=True, gen=gen)
    uh_y = udct.backward(y)
    lhs = (y.conj() * u_x).sum()
    rhs = (uh_y.conj() * x).sum()
    if transform_kind == "real":
        lhs_val = float(lhs.real.item())
        rhs_val = float(rhs.real.item())
        assert abs(lhs_val - rhs_val) / abs(lhs_val) < 1e-12
    else:
        assert float((lhs - rhs).abs().item()) / float(lhs.abs().item()) < 1e-12


@pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])  # type: ignore[untyped-decorator]
def test_torch_wavelet_gradcheck(transform_kind: AllKind) -> None:
    """Exact gradcheck holds for the flat forward and backward in wavelet mode."""
    gen = torch.Generator().manual_seed(0)
    udct = UDCT(
        shape=(8, 8),
        num_scales=2,
        high_frequency_mode="wavelet",
        transform_kind=transform_kind,
        dtype=torch.float64,
    )
    x = _randn_like_kind((8, 8), is_complex=transform_kind == "complex", gen=gen)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: udct.forward(t, structured=False), (x,), fast_mode=False
    )

    template = udct.forward(x.detach(), structured=False)
    c = _randn_like_kind(
        tuple(template.shape), is_complex=template.is_complex(), gen=gen
    )
    c.requires_grad_(True)
    assert torch.autograd.gradcheck(udct.backward, (c,), fast_mode=False)


@pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])  # type: ignore[untyped-decorator]
def test_torch_wavelet_multiscale_directional_grad(transform_kind: AllKind) -> None:
    """Adjoint autograd matches finite differences for 3-scale wavelet mode."""
    gen = torch.Generator().manual_seed(1)
    udct = UDCT(
        shape=(16, 16),
        num_scales=3,
        high_frequency_mode="wavelet",
        transform_kind=transform_kind,
        dtype=torch.float64,
    )
    x = _randn_like_kind((16, 16), is_complex=transform_kind == "complex", gen=gen)
    assert (
        _directional_grad_error(lambda t: udct.forward(t, structured=False), x, gen=gen)
        < 1e-7
    )

    template = udct.forward(x, structured=False)
    c = _randn_like_kind(
        tuple(template.shape), is_complex=template.is_complex(), gen=gen
    )
    assert _directional_grad_error(udct.backward, c, gen=gen) < 1e-7
