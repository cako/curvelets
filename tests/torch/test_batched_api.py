"""Native batching of the public torch API and UDCT.filter."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from curvelets.torch import UDCT

Kind = Literal["real", "complex"]
SHAPE = (32, 32)


def _batch(kind: Kind, *lead: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((*lead, *SHAPE), generator=gen, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn(x.shape, generator=gen, dtype=torch.float64))
    return x


def _udct(kind: Kind) -> UDCT:
    return UDCT(shape=SHAPE, num_scales=3, wedges_per_direction=3, transform_kind=kind)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_batched_forward_backward_match_loop(kind: Kind) -> None:
    """forward/backward/vect/struct on (*batch, *shape) equal a per-sample loop."""
    udct = _udct(kind)
    x = _batch(kind, 2, 3)
    coeffs = udct.forward(x)
    assert coeffs[1][0][0].shape[:2] == (2, 3)
    vec = udct.vect(coeffs)
    assert vec.shape[:2] == (2, 3)
    for i in range(2):
        for j in range(3):
            single = udct.vect(udct.forward(x[i, j]))
            torch.testing.assert_close(vec[i, j], single)
    restored = udct.struct(vec)
    torch.testing.assert_close(udct.vect(restored), vec)
    torch.testing.assert_close(udct.backward(restored), x)


def test_forward_rejects_wrong_trailing_shape() -> None:
    """Inputs must end with the transform shape."""
    with pytest.raises(ValueError, match="Expected input of shape"):
        _udct("real").forward(torch.randn(4, 32, 31))


def test_forward_vect_batched_matches_vmap() -> None:
    """flat forward on a batch equals vmap over single images, gradients included."""
    udct = UDCT(shape=SHAPE, num_scales=3, wedges_per_direction=3)

    def module(t: torch.Tensor) -> torch.Tensor:
        return udct.forward(t, structured=False)

    x = _batch("real", 5)
    torch.testing.assert_close(module(x), torch.vmap(module)(x))

    xb = x.clone().requires_grad_(True)
    module(xb).abs().pow(2).sum().backward()
    xs = x.clone().requires_grad_(True)
    torch.vmap(module)(xs).abs().pow(2).sum().backward()
    assert xb.grad is not None
    assert xs.grad is not None
    torch.testing.assert_close(xb.grad, xs.grad)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_filter(kind: Kind) -> None:
    """filter(identity) reconstructs; filter(fn) equals forward -> fn -> backward."""
    udct = _udct(kind)
    x = _batch(kind, 4)
    torch.testing.assert_close(udct.filter(x, lambda c: c), x)

    def hard_threshold(c: torch.Tensor) -> torch.Tensor:
        return c * (c.abs() > 0.5)

    expected = udct.backward(udct.struct(hard_threshold(udct.vect(udct.forward(x)))))
    torch.testing.assert_close(udct.filter(x, hard_threshold), expected)
