"""Tests for UDCT(cuda_graphs=True) and the adjoint autograd Functions."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Literal

import pytest
import torch

import curvelets.torch._cuda_graphs as cuda_graphs_module
from curvelets.torch import UDCT, UDCTModule
from curvelets.torch._autograd import BackwardWithAdjoint, ForwardWithAdjoint

Kind = Literal["real", "complex"]
SHAPE = (32, 32)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


class _EagerGraph:
    """Stand-in for a captured graph: replay reruns ``fn`` into the static output."""

    def __init__(
        self, fn: Callable[[torch.Tensor], torch.Tensor], inp: torch.Tensor
    ) -> None:
        self.fn, self.inp = fn, inp
        self.out = fn(inp)

    def replay(self) -> None:
        self.out.copy_(self.fn(self.inp))


@pytest.fixture(  # type: ignore[untyped-decorator]
    params=[
        "cpu",
        pytest.param("cuda", marks=requires_cuda),
    ]
)
def device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """CUDA with real graphs, or CPU with an eager stand-in for the capture.

    The CPU variant runs the same dispatch, caching and autograd logic
    everywhere; only the stream capture itself needs a GPU.
    """
    if request.param == "cpu":

        def fake_capture(
            fn: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor
        ) -> tuple[torch.Tensor, _EagerGraph, torch.Tensor]:
            graph = _EagerGraph(fn, example.detach().clone())
            return graph.inp, graph, graph.out

        monkeypatch.setattr(
            cuda_graphs_module, "GRAPH_DEVICE_TYPES", frozenset({"cuda", "cpu"})
        )
        monkeypatch.setattr(cuda_graphs_module, "_capture", fake_capture)
    return str(request.param)


def _udct(
    kind: Kind = "real",
    dtype: torch.dtype = torch.float64,
    *,
    cuda_graphs: bool = False,
    device: str = "cpu",
) -> UDCT:
    udct = UDCT(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
        dtype=dtype,
        cuda_graphs=cuda_graphs,
    )
    if device != "cpu":
        udct.apply_to_tensors(lambda t: t.to(device))
    return udct


def _image(
    kind: Kind,
    batch: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((*batch, *SHAPE), generator=gen, dtype=dtype)
    if kind == "complex":
        x = torch.complex(x, torch.randn((*batch, *SHAPE), generator=gen, dtype=dtype))
    return x.to(device)


# ---------------------------------------------------------------- everywhere
def test_off_by_default_and_switchable() -> None:
    """The flag defaults to off, can be toggled, and UDCTModule forwards it."""
    udct = _udct()
    assert not udct.cuda_graphs
    udct.cuda_graphs = True
    assert udct.cuda_graphs
    module = UDCTModule(shape=SHAPE, num_scales=3, cuda_graphs=True)
    assert module.cuda_graphs
    module.cuda_graphs = False
    assert not module._udct.cuda_graphs  # pylint: disable=protected-access


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_cpu_tensors_run_as_usual(kind: Kind) -> None:
    """On CPU the flag changes nothing and captures nothing."""
    plain, graphed = _udct(kind), _udct(kind, cuda_graphs=True)
    x = _image(kind, (2,))
    coeffs = plain._forward_vect(x)  # pylint: disable=protected-access
    torch.testing.assert_close(graphed._forward_vect(x), coeffs)  # pylint: disable=protected-access
    torch.testing.assert_close(
        graphed._backward_vect(coeffs),  # pylint: disable=protected-access
        plain._backward_vect(coeffs),  # pylint: disable=protected-access
    )
    assert len(graphed._graphs) == 0  # pylint: disable=protected-access


def test_copies_keep_the_flag_and_start_without_graphs() -> None:
    """Pickled and deep-copied transforms keep the settings, not the graphs."""
    udct = _udct(cuda_graphs=True)
    udct.cuda_graph_cache_size = 3
    for clone in (copy.deepcopy(udct), pickle.loads(pickle.dumps(udct))):
        assert clone.cuda_graphs
        assert clone.cuda_graph_cache_size == 3
        assert len(clone._graphs) == 0  # pylint: disable=protected-access


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_adjoint_functions_gradcheck(kind: Kind) -> None:
    """Each direction's gradient is the other direction (tight frame)."""
    udct = UDCT(
        shape=(16, 16), num_scales=2, transform_kind=kind, dtype=torch.float64
    )
    x = torch.randn(16, 16, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn_like(x))
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: ForwardWithAdjoint.apply(t, udct), (x,))
    coeffs = udct._forward_vect(x.detach()).requires_grad_(True)  # pylint: disable=protected-access
    assert torch.autograd.gradcheck(
        lambda c: BackwardWithAdjoint.apply(c, udct), (coeffs,)
    )


# ------------------------------------------- replays (CPU stand-in, or CUDA)
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("batch", [(), (3,), (2, 2)])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])  # type: ignore[untyped-decorator]
def test_replay_matches_plain_calls(
    kind: Kind, batch: tuple[int, ...], dtype: torch.dtype, device: str
) -> None:
    """Flat, nested and filter calls match, one graph per direction and shape."""
    plain = _udct(kind, dtype, device=device)
    graphed = _udct(kind, dtype, cuda_graphs=True, device=device)
    for seed in range(2):  # fresh inputs: the static buffers are refilled
        x = _image(kind, batch, dtype, device, seed)
        coeffs = plain._forward_vect(x)  # pylint: disable=protected-access
        torch.testing.assert_close(graphed._forward_vect(x), coeffs)  # pylint: disable=protected-access
        torch.testing.assert_close(
            graphed._backward_vect(coeffs),  # pylint: disable=protected-access
            plain._backward_vect(coeffs),  # pylint: disable=protected-access
        )
        nested = graphed.forward(x)
        torch.testing.assert_close(graphed.vect(nested), coeffs)
        torch.testing.assert_close(graphed.backward(nested), plain.backward(nested))
        torch.testing.assert_close(
            graphed.filter(x, lambda c: c * 0.5), plain.filter(x, lambda c: c * 0.5)
        )
    assert len(graphed._graphs) == 2  # pylint: disable=protected-access


def test_results_survive_later_calls(device: str) -> None:
    """Replays return copies, not the graph's static output buffer."""
    udct = _udct(cuda_graphs=True, device=device)
    x1, x2 = _image("real", device=device, seed=1), _image("real", device=device, seed=2)
    first = udct._forward_vect(x1)  # pylint: disable=protected-access
    expected = first.clone()
    udct._forward_vect(x2)  # pylint: disable=protected-access
    torch.testing.assert_close(first, expected)


def test_cache_is_bounded_and_cleared(device: str) -> None:
    """Least recently used graphs are dropped; device moves clear the cache."""
    udct = _udct(cuda_graphs=True, device=device)
    udct.cuda_graph_cache_size = 2
    for batch in (1, 2, 3):
        udct._forward_vect(_image("real", (batch,), device=device))  # pylint: disable=protected-access
    assert len(udct._graphs) == 2  # pylint: disable=protected-access
    udct.apply_to_tensors(lambda t: t.to(device))
    assert len(udct._graphs) == 0  # pylint: disable=protected-access
    udct._forward_vect(_image("real", device=device))  # pylint: disable=protected-access
    udct.clear_cuda_graphs()
    assert len(udct._graphs) == 0  # pylint: disable=protected-access


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_gradients_match_plain_autograd(kind: Kind, device: str) -> None:
    """Gradients through graphed calls equal those of plain autograd."""
    plain = _udct(kind, device=device)
    graphed = _udct(kind, cuda_graphs=True, device=device)
    x = _image(kind, (2,), device=device)

    def image_grad(udct: UDCT) -> torch.Tensor:
        xg = x.clone().requires_grad_(True)
        loss = udct.vect(udct.forward(xg)).abs().pow(2).sum()
        (grad,) = torch.autograd.grad(loss, xg)
        return grad

    torch.testing.assert_close(image_grad(graphed), image_grad(plain))
    assert len(graphed._graphs) == 2  # forward, and backward as its gradient  # pylint: disable=protected-access

    coeffs = plain._forward_vect(x)  # pylint: disable=protected-access

    def coeff_grad(udct: UDCT) -> torch.Tensor:
        cg = coeffs.clone().requires_grad_(True)
        loss = udct._backward_vect(cg).abs().pow(2).sum()  # pylint: disable=protected-access
        (grad,) = torch.autograd.grad(loss, cg)
        return grad

    torch.testing.assert_close(coeff_grad(graphed), coeff_grad(plain))


def test_module_training_and_double_backward(device: str) -> None:
    """UDCTModule gradients and second derivatives match without graphs."""
    x = _image("real", (2,), device=device)

    def grads(cuda_graphs: bool) -> tuple[torch.Tensor, torch.Tensor]:
        module = UDCTModule(
            shape=SHAPE, num_scales=3, dtype=torch.float64, cuda_graphs=cuda_graphs
        ).to(device)
        xg = x.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(
            module(xg).abs().pow(2).sum(), xg, create_graph=True
        )
        (second,) = torch.autograd.grad(grad.pow(2).sum(), xg)
        return grad.detach(), second

    for got, expected in zip(grads(True), grads(False), strict=True):
        torch.testing.assert_close(got, expected)


def test_module_gradcheck_with_graphs(device: str) -> None:
    """gradcheck holds with graph replays on CUDA (float64)."""
    module = UDCTModule(
        shape=(16, 16), num_scales=2, dtype=torch.float64, cuda_graphs=True
    ).to(device)
    x = torch.randn(16, 16, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))


def test_vmap_runs_without_graphs(device: str) -> None:
    """Under torch.vmap calls fall back to plain execution and still match."""
    udct = _udct(cuda_graphs=True, device=device)
    x = _image("real", (3,), device=device)
    expected = _udct(device=device)._forward_vect(x)  # pylint: disable=protected-access
    batched = torch.vmap(udct._forward_vect)(x)  # pylint: disable=protected-access
    torch.testing.assert_close(batched, expected)
