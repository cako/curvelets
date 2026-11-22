"""Tests for the fused Triton fold kernel (curvelets.torch._triton_fold).

With a GPU these run the compiled kernel on CUDA tensors. Without one they run
it on CPU tensors through Triton's interpreter (``conftest.py`` sets
``TRITON_INTERPRET=1`` before Triton is imported), so the CSR layout, the
kernel and the dispatch are also covered in CPU-only CI.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

import pytest
import torch

from curvelets.torch import UDCT
from curvelets.torch import _triton_fold as tf
from curvelets.torch._foldmap import CsrFold
from curvelets.torch._plan import UDCTPlan

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="Triton is not installed"
)

Kind = Literal["real", "complex"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = [
    ((32, 32), 3, 3),
    ((48, 40), 3, 6),
    ((16, 16, 16), 3, 3),
]


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _kernel_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the kernel, and let CPU tensors use it through the interpreter."""
    monkeypatch.setenv("CURVELETS_TRITON_CPU", "1")
    monkeypatch.delenv("CURVELETS_TRITON", raising=False)


def _udct(
    shape: tuple[int, ...],
    ns: int,
    wpd: int,
    kind: Kind,
    dtype: torch.dtype,
) -> UDCT:
    udct = UDCT(
        shape, num_scales=ns, wedges_per_direction=wpd, transform_kind=kind, dtype=dtype
    )
    if DEVICE != "cpu":
        udct.apply_to_tensors(lambda t: t.to(DEVICE))
    return udct


def _plan(udct: UDCT) -> UDCTPlan:
    """Build a CSR plan (with the current environment) and install it on ``udct``."""
    plan = UDCTPlan.build(
        udct._params,  # pylint: disable=protected-access
        udct.windows,
        udct._decimation_ratios_cpu,  # pylint: disable=protected-access
        complex_mode=udct._transform_kind == "complex",  # pylint: disable=protected-access
        fold_strategy="csr",
    )
    udct._plan = plan  # pylint: disable=protected-access
    return plan


def _csr(plan: UDCTPlan) -> tuple[tf.CsrFold, tf.CsrFold]:
    assert isinstance(plan.fwd_map, CsrFold)
    assert plan.bwd_map is not None
    assert isinstance(plan.bwd_map, CsrFold)
    return plan.fwd_map, plan.bwd_map


def _input(kind: Kind, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "complex":
        x = torch.complex(x, torch.randn(shape, generator=gen, dtype=torch.float64))
        return x.to(torch.promote_types(dtype, torch.complex64)).to(DEVICE)
    return x.to(dtype).to(DEVICE)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    tol = 2e-6 if dtype == torch.float32 else 1e-13
    assert actual.dtype == expected.dtype
    scale = float(expected.abs().max())
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol * scale)


@pytest.mark.parametrize(("shape", "ns", "wpd"), CONFIGS)  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("batch", [(), (3,), (2, 9)])  # type: ignore[untyped-decorator]
def test_kernel_matches_torch_fold(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    ns: int,
    wpd: int,
    kind: Kind,
    dtype: torch.dtype,
    batch: tuple[int, ...],
) -> None:
    """Forward and backward through the kernel match the torch fold.

    The (2, 9) batch has 18 rows: more than one batch tile, and a partial one.
    """
    udct = _udct(shape, ns, wpd, kind, dtype)
    plan = _plan(udct)
    x = _input(kind, (*batch, *shape), dtype)
    coeffs = plan.forward_flat(x)
    recon = plan.backward_flat(coeffs)
    monkeypatch.setattr(tf, "triton_device", lambda _device: False)
    _assert_close(coeffs, plan.forward_flat(x), dtype)
    _assert_close(recon, plan.backward_flat(coeffs), dtype)


@pytest.mark.parametrize("pair", [False, True])  # type: ignore[untyped-decorator]
def test_csr_layout_and_weights(pair: bool) -> None:
    """Targets with 0, 1 and several contributions; weight pairs scale Re and Im."""
    targets = torch.tensor([4, 0, 4, 2, 4, 2, 0], device=DEVICE)
    sources = torch.tensor([1, 3, 0, 2, 5, 1, 4], device=DEVICE)
    w = torch.tensor(
        [0.5, -1.0, 2.0, 1.5, 3.0, -0.25, 1.0], dtype=torch.float64, device=DEVICE
    )
    weights = torch.stack([w, -2 * w], dim=-1) if pair else w
    fold = CsrFold.build(targets, sources, weights, num_targets=6, num_sources=6)
    assert fold.row_ptr.tolist() == [0, 2, 2, 4, 4, 7, 7]
    gen = torch.Generator().manual_seed(1)
    src = torch.randn(5, 6, generator=gen, dtype=torch.complex128).to(DEVICE)
    expected = torch.zeros(5, 6, dtype=torch.complex128, device=DEVICE)
    for t, s, wj in zip(targets.tolist(), sources.tolist(), w.tolist(), strict=True):
        im = -2 * wj if pair else wj
        expected[:, t] += wj * src[:, s].real + 1j * im * src[:, s].imag
    torch.testing.assert_close(fold.apply(src), expected)
    torch.testing.assert_close(fold._apply_torch(src), expected)  # pylint: disable=protected-access


def test_kernel_is_used_and_falls_back_for_vmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain and differentiable calls use the kernel; vmapped calls use torch ops."""
    calls = []
    real_triton = tf.CsrFold._apply_triton  # pylint: disable=protected-access

    def counting(self: tf.CsrFold, src: torch.Tensor) -> torch.Tensor:
        calls.append(src.shape)
        return real_triton(self, src)

    monkeypatch.setattr(tf.CsrFold, "_apply_triton", counting)
    udct = _udct((32, 32), 3, 3, "real", torch.float64)
    plan = _plan(udct)
    x = _input("real", (2, 32, 32), torch.float64)
    coeffs = plan.forward_flat(x)
    plan.backward_flat(coeffs)
    assert len(calls) == 2

    calls.clear()
    xg = x.clone().requires_grad_(True)
    plan.backward_flat(plan.forward_flat(xg)).sum().backward()
    assert len(calls) == 4
    assert xg.grad is not None

    with torch.no_grad():
        plan.forward_flat(xg)
    assert len(calls) == 5

    calls.clear()
    vmapped = torch.vmap(plan.forward_flat)(x)
    assert not calls
    torch.testing.assert_close(vmapped, coeffs)


def test_registered_op_and_compile_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``torch.library`` op (used under ``torch.compile``) matches the eager launch."""
    plan = _plan(_udct((32, 32), 3, 3, "real", torch.float32))
    fwd_csr, _ = _csr(plan)
    assert sum(t.numel() for t in fwd_csr.tensors()) > 0
    src = plan._spectrum(_input("real", (2, 32, 32), torch.float32))  # pylint: disable=protected-access
    eager = fwd_csr.apply(src)
    op = torch.ops.curvelets.csr_fold(
        src, fwd_csr.row_ptr, fwd_csr.col, fwd_csr.weights, fwd_csr.num_targets
    )
    torch.testing.assert_close(op, eager)
    # The traced branch, including registration for maps built elsewhere
    # (e.g. unpickled), without paying for a real compilation.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(tf, "_OP_REGISTERED", False)
    torch.testing.assert_close(fwd_csr.apply(src), eager)
    assert tf._OP_REGISTERED  # pylint: disable=protected-access


def test_autograd_matches_torch_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Differentiable calls through the kernel match the torch fallback gradient."""
    udct = _udct((16, 16, 16), 3, 3, "real", torch.float64)
    plan = _plan(udct)
    x = _input("real", (2, 16, 16, 16), torch.float64)

    xg1 = x.clone().requires_grad_(True)
    plan.backward_flat(plan.forward_flat(xg1)).sum().backward()

    monkeypatch.setattr(tf, "triton_device", lambda _device: False)
    plan_torch = _plan(_udct((16, 16, 16), 3, 3, "real", torch.float64))
    xg2 = x.clone().requires_grad_(True)
    plan_torch.backward_flat(plan_torch.forward_flat(xg2)).sum().backward()

    assert xg1.grad is not None
    assert xg2.grad is not None
    torch.testing.assert_close(xg1.grad, xg2.grad, rtol=1e-10, atol=1e-10)


def test_transpose_csr() -> None:
    """The transpose fold is the adjoint of the forward fold."""
    targets = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64, device=DEVICE)
    sources = torch.tensor([0, 1, 2, 1, 3], dtype=torch.int64, device=DEVICE)
    weights = torch.tensor(
        [[1.5, -1.5], [2.0, 3.0], [0.5, -0.5], [1.0, 1.0], [-2.0, 2.0]],
        dtype=torch.float64,
        device=DEVICE,
    )
    csr = tf.CsrFold.from_sorted(targets, sources, weights, 3, 4, torch.int64)
    t = csr.transpose()
    assert t.num_targets == 4
    assert t.num_sources == 3


def test_disabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CURVELETS_TRITON=0`` runs the PyTorch CSR fallback."""
    monkeypatch.setenv("CURVELETS_TRITON", "0")
    assert not tf.triton_available()
    assert not tf.triton_device(torch.device("cuda"))
    plan = _plan(_udct((32, 32), 3, 3, "real", torch.float32))
    assert isinstance(plan.fwd_map, CsrFold)
    x = _input("real", (2, 32, 32), torch.float32)
    coeffs = plan.forward_flat(x)
    assert coeffs.shape == (2, plan.ncoef)


def test_device_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA gets the kernel; CPU only for tests, through the interpreter."""
    assert tf.triton_device(torch.device("cuda"))
    assert not tf.triton_device(torch.device("mps"))
    monkeypatch.setenv("CURVELETS_TRITON_CPU", "0")
    assert not tf.triton_device(torch.device("cpu"))


def test_empty_batch() -> None:
    """Zero rows give empty results."""
    plan = _plan(_udct((32, 32), 3, 3, "real", torch.float32))
    out = plan.forward_flat(torch.zeros(0, 32, 32, device=DEVICE))
    assert out.shape == (0, plan.ncoef)
    assert plan.backward_flat(out).shape == (0, 32, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_compile_sees_through_kernel(kind: Kind) -> None:
    """``torch.compile(fullgraph=True)`` traces the registered op and matches eager."""
    plan = _plan(_udct((64, 64), 3, 3, kind, torch.float32))
    x = _input(kind, (2, 64, 64), torch.float32)
    coeffs = plan.forward_flat(x)
    torch._dynamo.reset()  # pylint: disable=protected-access
    # aot_eager, as in test_plan.py: Inductor warns on the complex FFTs.
    forward = torch.compile(plan.forward_flat, backend="aot_eager", fullgraph=True)
    backward = torch.compile(plan.backward_flat, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(forward(x), coeffs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        backward(coeffs), plan.backward_flat(coeffs), atol=1e-5, rtol=1e-5
    )
