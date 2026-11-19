"""Device-transfer regression tests for the PyTorch UDCT.

The CPU tests run everywhere. The CUDA tests are skipped without a GPU; run them on
a GPU machine (e.g. ``pytest tests/torch``).
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

import curvelets.torch._sparse_window as sparse_window_module
from curvelets.torch import UDCT, UDCTModule

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

TransformKind = Literal["real", "complex"]


def _make_udct(kind: TransformKind, shape: tuple[int, ...] = (32, 32)) -> UDCT:
    return UDCT(shape=shape, num_scales=3, wedges_per_direction=3, transform_kind=kind)


def _input(kind: TransformKind, shape: tuple[int, ...], device: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn(shape, generator=gen, dtype=torch.float64))
    return x.to(device)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_apply_to_tensors_keeps_periodized_maps(kind: TransformKind) -> None:
    """Moving the transform must not drop the precomputed index maps."""
    udct = _make_udct(kind)
    udct.apply_to_tensors(lambda t: t.clone())
    for scale in udct.windows:
        for direction in scale:
            for window in direction:
                assert window.folded_indices is not None
                assert window.out_shape is not None
                if kind == "complex":
                    assert window.flipped_indices is not None
                    assert window.flipped_folded_indices is not None


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_moved_transform_does_not_recompute_maps(
    kind: TransformKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward/backward after a move must use the attached maps, not recompute them."""
    udct = _make_udct(kind)
    udct.apply_to_tensors(lambda t: t.clone())
    x = _input(kind, udct.shape, "cpu")

    def _fail(*_args: object, **_kwargs: object) -> None:
        msg = "index maps were recomputed in the hot path"
        raise AssertionError(msg)

    monkeypatch.setattr(sparse_window_module, "compute_folded_indices", _fail)
    monkeypatch.setattr(sparse_window_module, "flip_fft_indices", _fail)
    recon = udct.backward(udct.forward(x))
    torch.testing.assert_close(recon, x)


@pytest.mark.parametrize("device", DEVICES)  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_struct_round_trip_after_move(device: str, kind: TransformKind) -> None:
    """``struct(vect(c))`` must work after the transform is moved to ``device``."""
    udct = _make_udct(kind)
    udct.apply_to_tensors(lambda t: t.to(device))
    x = _input(kind, udct.shape, device)
    coeffs = udct.forward(x)
    restored = udct.struct(udct.vect(coeffs))
    for scale_c, scale_r in zip(coeffs, restored, strict=True):
        for dir_c, dir_r in zip(scale_c, scale_r, strict=True):
            for wedge_c, wedge_r in zip(dir_c, dir_r, strict=True):
                torch.testing.assert_close(wedge_r, wedge_c)
    torch.testing.assert_close(udct.backward(restored), x)


@pytest.mark.parametrize("device", DEVICES)  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_udct_module_backprop_on_device(device: str, kind: TransformKind) -> None:
    """Backprop through ``UDCTModule`` works on ``device`` and matches the CPU."""
    shape = (32, 32)
    x_cpu = _input(kind, shape, "cpu")

    def grad_on(dev: str) -> torch.Tensor:
        module = UDCTModule(
            shape=shape, num_scales=3, wedges_per_direction=3, transform_type=kind
        ).to(dev)
        x = x_cpu.to(dev).requires_grad_(True)
        module(x).abs().pow(2).sum().backward()
        assert x.grad is not None
        return x.grad.cpu()

    torch.testing.assert_close(grad_on(device), grad_on("cpu"))


def test_public_decimation_ratios_follow_the_module() -> None:
    """The public ``decimation_ratios`` still move with ``.to()``."""
    module = UDCTModule(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    module = module.to(DEVICES[-1])
    assert all(d.device.type == DEVICES[-1] for d in module.decimation_ratios)
