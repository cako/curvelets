"""Copying, pickling and saving the torch UDCT (``copy``, ``pickle``, ``torch.save``)."""

from __future__ import annotations

import copy
import io
import pickle
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import pytest
import torch
from torch import nn

from curvelets.torch import UDCT, UDCTModule

SHAPE = (32, 32)
Kind = Literal["real", "complex"]
T = TypeVar("T")

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _udct(kind: Kind = "real") -> UDCT:
    return UDCT(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,
        dtype=torch.float64,
    )


def _image(
    kind: Kind = "real", batch: tuple[int, ...] = (), device: str = "cpu"
) -> torch.Tensor:
    dtype = torch.complex128 if kind == "complex" else torch.float64
    return torch.randn(*batch, *SHAPE, dtype=dtype, device=device)


def _saved(obj: T, **load_kwargs: Any) -> T:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    loaded: T = torch.load(buffer, weights_only=False, **load_kwargs)
    return loaded


COPIES: dict[str, Callable[[Any], Any]] = {
    "deepcopy": copy.deepcopy,
    "pickle": lambda obj: pickle.loads(pickle.dumps(obj)),
    "torch.save": _saved,
}


@pytest.mark.parametrize("how", list(COPIES))  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_copy_of_used_transform_matches(kind: Kind, how: str) -> None:
    """A transform that has run copies to one with the same results and gradients."""
    udct = _udct(kind)
    x = _image(kind, (2,))
    coeffs = udct.forward_vect(x)  # builds the plan before copying
    clone = COPIES[how](udct)
    assert isinstance(clone, UDCT)
    torch.testing.assert_close(clone.forward_vect(x), coeffs)
    torch.testing.assert_close(clone(x), udct(x))
    torch.testing.assert_close(clone.backward_vect(coeffs), udct.backward_vect(coeffs))

    grads = []
    for transform in (udct, clone):
        xg = x.clone().requires_grad_(True)
        transform.forward_vect(xg).abs().pow(2).sum().backward()
        grads.append(xg.grad)
    torch.testing.assert_close(grads[1], grads[0])


def test_copy_is_independent() -> None:
    """Recasting a copy leaves the original's precision alone."""
    udct = _udct()
    clone = copy.deepcopy(udct).float()
    assert clone.dtype == torch.float32
    assert udct.dtype == torch.float64
    assert udct.windows[0][0][0].values.dtype == torch.float64


class _Classifier(nn.Module):  # type: ignore[misc]
    """A network with the transform as a submodule (module level, so it pickles)."""

    def __init__(self) -> None:
        super().__init__()
        self.udct = _udct()
        self.head = nn.Linear(self.udct.vect(self.udct(_image())).shape[-1], 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.udct.forward_vect(x).abs())


def test_network_with_transform_saves_and_loads() -> None:
    """``torch.save`` of a whole network keeps the transform; its ``state_dict``
    holds only the learnable parameters and loads into a fresh network."""
    model = _Classifier()
    x = _image(batch=(4,))
    expected = model(x)

    torch.testing.assert_close(_saved(model)(x), expected)

    state = model.state_dict()
    assert set(state) == {"head.weight", "head.bias"}
    fresh = _Classifier()
    fresh.load_state_dict(_saved(state))
    torch.testing.assert_close(fresh(x), expected)


@pytest.mark.parametrize("how", list(COPIES))  # type: ignore[untyped-decorator]
def test_deprecated_udct_module_copies(how: str) -> None:
    """The deprecated UDCTModule keeps its class (flat output) and does not
    warn again when unpickled."""
    with pytest.deprecated_call():
        module = UDCTModule(
            shape=SHAPE, num_scales=3, wedges_per_direction=3, dtype=torch.float64
        )
    x = _image()
    clone = COPIES[how](module)  # warnings are errors in this suite
    assert type(clone) is UDCTModule
    torch.testing.assert_close(clone(x), module(x))


def test_plan_for_another_device_is_rebuilt() -> None:
    """A plan built for another device than the windows' (as after
    ``torch.load(map_location=...)``) is rebuilt before use, so no
    device-specific choice (fold strategy, Triton kernels) carries over."""
    udct = _udct()
    x = _image(batch=(2,))
    expected = udct.forward_vect(x)
    stale = udct._get_plan()  # pylint: disable=protected-access
    stale.device = torch.device("meta")  # as if built on another device
    torch.testing.assert_close(udct.forward_vect(x), expected)
    plan = udct._get_plan()  # pylint: disable=protected-access
    assert plan is not stale
    assert plan.device == torch.device("cpu")


@needs_cuda  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_cuda_transform_copies_and_loads_on_cpu(kind: Kind) -> None:
    """A transform used on CUDA copies to CUDA, and ``torch.load`` with
    ``map_location="cpu"`` gives a working CPU transform."""
    udct = _udct(kind).to("cuda")
    x = _image(kind, (2,), device="cuda")
    expected = udct.forward_vect(x)

    for how in ("deepcopy", "pickle", "torch.save"):
        clone = COPIES[how](udct)
        assert clone.windows[0][0][0].values.is_cuda
        torch.testing.assert_close(clone.forward_vect(x), expected)

    on_cpu = _saved(udct, map_location="cpu")
    assert not on_cpu.windows[0][0][0].values.is_cuda
    torch.testing.assert_close(on_cpu.forward_vect(x.cpu()), expected.cpu())
