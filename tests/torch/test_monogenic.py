"""Monogenic curvelet transform in torch: parity with numpy, layout, gradients."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

from curvelets.numpy import UDCT as NpUDCT
from curvelets.torch import UDCT

HighFrequencyMode = Literal["curvelet", "wavelet"]

# Even and odd axes (the Riesz images are complex along even axes), 2D and 3D,
# and the wavelet mode at the highest scale.
CASES = [
    ((32, 32), "curvelet"),
    ((33, 30), "curvelet"),
    ((31, 33), "curvelet"),
    ((32, 32), "wavelet"),
    ((16, 18, 12), "curvelet"),
]
# Shapes the transform reconstructs exactly (divisible by the decimation).
RECONSTRUCTING = [
    ((32, 32), "curvelet"),
    ((32, 32), "wavelet"),
    ((32, 32, 32), "curvelet"),
]
# The torch windows differ from numpy's by ~1e-6 for every transform kind.
ATOL = 1e-5


def _pair(
    shape: tuple[int, ...], high: HighFrequencyMode = "curvelet"
) -> tuple[UDCT, NpUDCT]:
    kwargs = {
        "shape": shape,
        "num_scales": 3,
        "wedges_per_direction": 3,
        "high_frequency_mode": high,
        "transform_kind": "monogenic",
    }
    return UDCT(dtype=torch.float64, **kwargs), NpUDCT(**kwargs)  # type: ignore[arg-type]


def _wedges(coeffs: list[list[list[torch.Tensor]]]) -> list[torch.Tensor]:
    return [c for s in coeffs for d in s for c in d]


@pytest.mark.parametrize(("shape", "high"), CASES)  # type: ignore[untyped-decorator]
def test_matches_numpy(shape: tuple[int, ...], high: HighFrequencyMode) -> None:
    """Coefficients, vect layout, reconstruction and monogenic() match numpy."""
    udct, ref = _pair(shape, high)
    x = np.random.default_rng(0).standard_normal(shape)
    xt = torch.from_numpy(x)

    coeffs = udct(xt)
    coeffs_ref = ref.forward(x)
    for c, c_ref in zip(_wedges(coeffs), _wedges(coeffs_ref), strict=True):
        assert c.shape == c_ref.shape
        np.testing.assert_allclose(c.numpy(), c_ref, atol=ATOL)
    np.testing.assert_allclose(
        udct.forward_vect(xt).numpy(), ref.vect(coeffs_ref), atol=ATOL
    )

    # numpy's tuples of components are the first axis of the torch results.
    recon = udct.backward(coeffs)
    assert recon.shape == (len(shape) + 1, *shape)
    recon_ref = ref.backward(coeffs_ref)
    assert isinstance(recon_ref, tuple)
    np.testing.assert_allclose(recon.numpy(), np.stack(recon_ref), atol=ATOL)
    np.testing.assert_allclose(
        udct.monogenic(xt).numpy(), np.stack(ref.monogenic(x)), atol=1e-12
    )


@pytest.mark.parametrize(("shape", "high"), RECONSTRUCTING)  # type: ignore[untyped-decorator]
def test_round_trip_scalar(shape: tuple[int, ...], high: HighFrequencyMode) -> None:
    """backward(forward(f)) returns f first, then one component per dimension.

    Like numpy, the Riesz components only approximate -R_k f (each Riesz
    channel keeps the real part of its coefficients), so they are compared
    with numpy in test_matches_numpy rather than with monogenic(f).
    """
    udct, _ = _pair(shape, high)
    x = torch.randn(shape, dtype=torch.float64)
    recon = udct.backward(udct(x))
    assert recon.shape == (len(shape) + 1, *shape)
    torch.testing.assert_close(recon[0], x, atol=1e-4, rtol=1e-4)


def test_layout_and_batching() -> None:
    """Batched calls equal per-image calls; vect/struct/forward_vect agree."""
    udct, _ = _pair((32, 30))
    x = torch.randn(2, 3, 32, 30, dtype=torch.float64)
    coeffs = udct(x)
    shapes = udct.coefficient_shapes()
    for c, s in zip(
        _wedges(coeffs), [w for sc in shapes for d in sc for w in d], strict=True
    ):
        assert tuple(c.shape) == (2, 3, *s)

    vec = udct.forward_vect(x)
    assert vec.shape == (
        2,
        3,
        sum(int(np.prod(s)) for s in [w for sc in shapes for d in sc for w in d]),
    )
    torch.testing.assert_close(udct.vect(coeffs), vec)
    for a, b in zip(_wedges(udct.struct(vec)), _wedges(coeffs), strict=True):
        torch.testing.assert_close(a, b)

    single = udct.forward_vect(x[1, 2])
    torch.testing.assert_close(vec[1, 2], single)
    recon = udct.backward_vect(vec)
    assert recon.shape == (3, 2, 3, 32, 30)
    torch.testing.assert_close(recon[:, 1, 2], udct.backward_vect(single))
    torch.testing.assert_close(udct.filter(x, lambda c: c), recon)
    monogenic = udct.monogenic(x)
    assert monogenic.shape == (3, 2, 3, 32, 30)
    torch.testing.assert_close(monogenic[:, 1, 2], udct.monogenic(x[1, 2]))


def test_invalid_coefficients() -> None:
    """Complex or wrongly sized coefficients are rejected."""
    udct, _ = _pair((32, 32))
    vec = udct.forward_vect(torch.randn(32, 32, dtype=torch.float64))
    with pytest.raises(ValueError, match="real-valued"):
        udct.backward_vect(vec.to(torch.complex128))
    with pytest.raises(ValueError, match="Expected monogenic coefficients"):
        udct.backward_vect(vec[:-1])
    coeffs = udct.struct(vec)
    coeffs[0][0][0] = coeffs[0][0][0][..., :2]
    with pytest.raises(ValueError, match="Expected monogenic coefficients"):
        udct.backward(coeffs)
    with pytest.raises(ValueError, match="real-valued input"):
        udct.monogenic(torch.randn(32, 32, dtype=torch.complex128))


def test_gradcheck() -> None:
    """gradcheck holds for the nested and flat forward and the backward transform."""
    udct = UDCT(
        shape=(16, 18), num_scales=2, transform_kind="monogenic", dtype=torch.float64
    )
    x = torch.randn(16, 18, dtype=torch.float64, requires_grad=True)

    def nested(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(_wedges(udct(t)))

    assert torch.autograd.gradcheck(nested, (x,), fast_mode=True)
    assert torch.autograd.gradcheck(udct.forward_vect, (x,), fast_mode=True)
    v = udct.forward_vect(x.detach()).requires_grad_(True)
    assert torch.autograd.gradcheck(udct.backward_vect, (v,), fast_mode=True)
    assert torch.autograd.gradcheck(udct.monogenic, (x,), fast_mode=True)


def test_nested_gradients_match_flat() -> None:
    """Gradients through the nested coefficients equal those of the flat path."""
    udct, _ = _pair((32, 32))
    x = torch.randn(2, 32, 32, dtype=torch.float64)
    xn = x.clone().requires_grad_(True)
    coeffs = udct(xn)
    loss = sum(c.pow(2).sum() for c in _wedges(coeffs))
    loss = loss + udct.backward(coeffs).abs().sum()
    loss.backward()
    xf = x.clone().requires_grad_(True)
    vec = udct.forward_vect(xf)
    (vec.pow(2).sum() + udct.backward_vect(vec).abs().sum()).backward()
    assert xn.grad is not None
    assert xf.grad is not None
    torch.testing.assert_close(xn.grad, xf.grad)


def test_to_rebuilds_caches() -> None:
    """Changing the precision with .to() rebuilds the Riesz filters."""
    udct = UDCT(shape=(32, 32), transform_kind="monogenic", dtype=torch.float32)
    x = torch.randn(32, 32, dtype=torch.float64)
    expected = udct.forward_vect(x.float()).double()
    udct.to(torch.float64)
    torch.testing.assert_close(udct.forward_vect(x), expected, atol=1e-5, rtol=1e-4)
    assert udct.monogenic(x)[1].dtype == torch.float64


def _small() -> UDCT:
    return UDCT(
        shape=(16, 18), num_scales=2, transform_kind="monogenic", dtype=torch.float64
    )


def test_second_derivatives() -> None:
    """gradgradcheck holds for the flat forward and backward transforms."""
    udct = _small()
    x = torch.randn(16, 18, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(udct.forward_vect, (x,), fast_mode=True)
    v = udct.forward_vect(x.detach()).requires_grad_(True)
    assert torch.autograd.gradgradcheck(udct.backward_vect, (v,), fast_mode=True)


def test_vmap_and_func_grad() -> None:
    """torch.vmap matches batched calls; torch.func.grad matches autograd."""
    udct = _small()
    x = torch.randn(3, 16, 18, dtype=torch.float64)
    vec = udct.forward_vect(x)
    torch.testing.assert_close(torch.vmap(udct.forward_vect)(x), vec)
    torch.testing.assert_close(
        torch.vmap(udct.backward_vect)(vec), udct.backward_vect(vec).movedim(1, 0)
    )

    def loss(t: torch.Tensor) -> torch.Tensor:
        return (
            udct.forward_vect(t).pow(2).sum()
            + udct.backward_vect(udct.forward_vect(t)).abs().sum()
        )

    xg = x[0].clone().requires_grad_(True)
    loss(xg).backward()
    torch.testing.assert_close(torch.func.grad(loss)(x[0]), xg.grad)


def test_compile_matches_eager() -> None:
    """Each flat transform traces without graph breaks and matches eager,
    gradients included."""
    udct = _small()
    x = torch.randn(2, 16, 18, dtype=torch.float64)
    vec = udct.forward_vect(x)  # builds the plan, layout and Riesz filters
    recon = udct.backward_vect(vec)
    torch._dynamo.reset()  # pylint: disable=protected-access
    forward = torch.compile(udct.forward_vect, backend="aot_eager", fullgraph=True)
    backward = torch.compile(udct.backward_vect, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(forward(x), vec)
    torch.testing.assert_close(backward(vec), recon)

    for fn, compiled, inp in (
        (udct.forward_vect, forward, x),
        (udct.backward_vect, backward, vec),
    ):
        leaf_c = inp.clone().requires_grad_(True)
        compiled(leaf_c).pow(2).sum().backward()
        leaf_e = inp.clone().requires_grad_(True)
        fn(leaf_e).pow(2).sum().backward()
        torch.testing.assert_close(leaf_c.grad, leaf_e.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")  # type: ignore[untyped-decorator]
@pytest.mark.parametrize(("shape", "high"), CASES)  # type: ignore[untyped-decorator]
def test_cuda_matches_cpu(shape: tuple[int, ...], high: HighFrequencyMode) -> None:
    """The transform on CUDA matches the CPU, gradients included."""
    udct, _ = _pair(shape, high)
    udct_cuda, _ = _pair(shape, high)
    udct_cuda.cuda()
    x = torch.randn(2, *shape, dtype=torch.float64)
    xc = x.cuda().requires_grad_(True)
    xg = x.clone().requires_grad_(True)
    vec_cuda = udct_cuda.forward_vect(xc)
    vec = udct.forward_vect(xg)
    torch.testing.assert_close(vec_cuda.cpu(), vec)
    recon_cuda = udct_cuda.backward_vect(vec_cuda)
    recon = udct.backward_vect(vec)
    torch.testing.assert_close(recon_cuda.cpu(), recon)
    torch.testing.assert_close(udct_cuda.monogenic(xc).cpu(), udct.monogenic(xg))
    recon_cuda.pow(2).sum().backward()
    recon.pow(2).sum().backward()
    assert xc.grad is not None
    torch.testing.assert_close(xc.grad.cpu(), xg.grad)
