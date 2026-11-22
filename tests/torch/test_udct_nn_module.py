"""UDCT as a torch.nn.Module, and the deprecated UDCTModule."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch import nn

from curvelets.torch import UDCT, UDCTModule

SHAPE = (32, 32)
Kind = Literal["real", "complex"]


def _udct(kind: Kind = "real") -> UDCT:
    return UDCT(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
        dtype=torch.float64,
    )


def _image(kind: Kind, batch: tuple[int, ...] = ()) -> torch.Tensor:
    if kind == "complex":
        return torch.randn(*batch, *SHAPE, dtype=torch.complex128)
    return torch.randn(*batch, *SHAPE, dtype=torch.float64)


def test_is_a_module_without_state() -> None:
    """UDCT is an nn.Module with no parameters and an empty state_dict."""
    udct = _udct()
    assert isinstance(udct, nn.Module)
    assert list(udct.parameters()) == []
    assert udct.state_dict() == {}


def test_call_is_nested_forward() -> None:
    """Calling the module runs forward and returns nested coefficients."""
    udct = _udct()
    x = _image("real")
    coeffs = udct(x)
    assert isinstance(coeffs, list)
    torch.testing.assert_close(udct.vect(coeffs), udct.forward(x, structured=False))


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_nested_gradients_match_flat(kind: Kind) -> None:
    """Gradients through forward/backward (nested) equal those of the flat path."""
    udct = _udct(kind)
    x = _image(kind, (2,))
    xn = x.clone().requires_grad_(True)
    coeffs = udct(xn)
    loss = sum(c.abs().pow(2).sum() for s in coeffs for d in s for c in d)
    loss = loss + udct.backward(coeffs).abs().sum()
    loss.backward()
    xf = x.clone().requires_grad_(True)
    vec = udct.forward(xf, structured=False)
    (vec.abs().pow(2).sum() + udct.backward(vec).abs().sum()).backward()
    assert xn.grad is not None
    assert xf.grad is not None
    torch.testing.assert_close(xn.grad, xf.grad)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_nested_forward_gradcheck(kind: Kind) -> None:
    """gradcheck holds for the nested forward and the backward transform,
    including real inputs to the complex transform."""
    udct = UDCT(shape=(16, 16), num_scales=2, transform_kind=kind, dtype=torch.float64)
    x = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)

    def nested(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(c for s in udct(t) for d in s for c in d)

    assert torch.autograd.gradcheck(nested, (x,), fast_mode=True)
    v = udct.forward(x.detach(), structured=False).requires_grad_(True)
    assert torch.autograd.gradcheck(udct.backward, (v,), fast_mode=True)
    # Real coefficients through the backward transform get real gradients.
    r = v.detach().real.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(udct.backward, (r,), fast_mode=True)


def test_works_inside_sequential() -> None:
    """The flat forward composes with ordinary layers."""
    udct = _udct()

    class Features(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.udct = udct
            self.head = nn.LazyLinear(3, dtype=torch.float64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.udct.forward(x, structured=False).abs())

    model = Features()
    model(_image("real", (4,))).sum().backward()
    assert model.head.weight.grad is not None
    # The transform is a submodule: .to() reaches it.
    model.to(torch.float32)
    assert udct.dtype == torch.float32


def test_udct_module_is_deprecated_and_returns_flat() -> None:
    """UDCTModule warns and keeps returning flat coefficients from __call__."""
    with pytest.deprecated_call():
        module = UDCTModule(
            shape=SHAPE,
            num_scales=3,
            wedges_per_direction=3,
            transform_type="complex",
            dtype=torch.float64,
        )
    assert isinstance(module, UDCT)
    x = _image("complex", (2,))
    torch.testing.assert_close(module(x), _udct("complex").forward(x, structured=False))
