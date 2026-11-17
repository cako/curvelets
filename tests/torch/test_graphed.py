"""Tests for GraphedUDCT (CUDA-graph replay of the fused transform)."""

from __future__ import annotations

import importlib.util
from typing import Literal

import pytest
import torch

from curvelets.torch import UDCT, GraphedUDCT

Kind = Literal["real", "complex"]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _cuda_udct(kind: Kind, dtype: torch.dtype = torch.float32) -> UDCT:
    udct = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
        dtype=dtype,
    )
    udct.apply_to_tensors(lambda t: t.to("cuda"))
    return udct


def _image(kind: Kind, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if kind == "complex":
        x = torch.complex(x, torch.randn_like(x))
    return x


def test_requires_a_cuda_transform() -> None:
    """CPU transforms are rejected with a clear error."""
    udct = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    with pytest.raises(ValueError, match="CUDA"):
        GraphedUDCT(udct)


def test_rejects_monogenic() -> None:
    """The monogenic transform has no fused plan to capture."""
    udct = UDCT(
        shape=(32, 32), num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )
    with pytest.raises(NotImplementedError):
        GraphedUDCT(udct)


@requires_cuda  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 2)])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])  # type: ignore[untyped-decorator]
def test_matches_eager(
    kind: Kind, batch_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """Replayed graphs give the eager forward and backward, call after call."""
    udct = _cuda_udct(kind, dtype)
    graphed = GraphedUDCT(udct, batch_shape)
    assert graphed.input_shape == (*batch_shape, 64, 64)
    for _ in range(2):  # fresh inputs each time: the static buffers are refilled
        x = _image(kind, (*batch_shape, 64, 64), dtype)
        expected = udct._forward_vect(x)  # pylint: disable=protected-access
        coeffs = graphed.forward(x)
        assert coeffs.shape == graphed.coefficient_shape
        assert graphed.ncoef == expected.shape[-1]
        torch.testing.assert_close(coeffs, expected)
        torch.testing.assert_close(
            graphed.backward(coeffs),
            udct._backward_vect(coeffs),  # pylint: disable=protected-access
        )


@requires_cuda  # type: ignore[untyped-decorator]
def test_real_input_to_complex_transform() -> None:
    """The complex transform also takes real images of its precision."""
    udct = _cuda_udct("complex")
    graphed = GraphedUDCT(udct)
    x = torch.randn(64, 64, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(graphed.forward(x), udct._forward_vect(x))  # pylint: disable=protected-access


@requires_cuda  # type: ignore[untyped-decorator]
def test_outputs_are_copies_unless_asked() -> None:
    """By default results survive the next call; copy_outputs=False aliases them."""
    udct = _cuda_udct("real")
    x1, x2 = (torch.randn(64, 64, dtype=torch.float32, device="cuda") for _ in range(2))

    graphed = GraphedUDCT(udct)
    first = graphed.forward(x1)
    graphed.forward(x2)
    torch.testing.assert_close(first, udct._forward_vect(x1))  # pylint: disable=protected-access

    aliased = GraphedUDCT(udct, copy_outputs=False)
    first = aliased.forward(x1)
    second = aliased.forward(x2)
    assert first.data_ptr() == second.data_ptr()


@requires_cuda  # type: ignore[untyped-decorator]
def test_rejects_other_shapes_and_dtypes() -> None:
    """Inputs must match the captured shape and the transform's precision."""
    graphed = GraphedUDCT(_cuda_udct("real"), (2,))
    with pytest.raises(ValueError, match="shape"):
        graphed.forward(torch.randn(3, 64, 64, dtype=torch.float32, device="cuda"))
    with pytest.raises(TypeError):
        graphed.forward(torch.randn(2, 64, 64, device="cuda", dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        graphed.backward(torch.zeros(graphed.ncoef, device="cuda"))


@requires_cuda  # type: ignore[untyped-decorator]
@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="requires Triton"
)  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_compiled_graphs_match_eager(kind: Kind) -> None:
    """compile=True (Triton folds inside the graphs) gives the same results."""
    udct = _cuda_udct(kind)
    graphed = GraphedUDCT(udct, (2,), compile=True)
    x = _image(kind, (2, 64, 64), torch.float32)
    coeffs = graphed.forward(x)
    torch.testing.assert_close(coeffs, udct._forward_vect(x))  # pylint: disable=protected-access
    torch.testing.assert_close(
        graphed.backward(coeffs),
        udct._backward_vect(coeffs),  # pylint: disable=protected-access
    )
