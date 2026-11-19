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
from curvelets.torch._foldmap import GatherFold
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
    """Build a gather plan (with the current environment) and install it on ``udct``."""
    plan = UDCTPlan.build(
        udct._parameters,  # pylint: disable=protected-access
        udct.windows,
        udct._decimation_ratios_cpu,  # pylint: disable=protected-access
        complex_mode=udct._transform_kind == "complex",  # pylint: disable=protected-access
        fold_strategy="gather",
    )
    udct._plan = plan  # pylint: disable=protected-access
    return plan


def _csr(plan: UDCTPlan) -> tuple[tf.CsrFold | None, tf.CsrFold | None]:
    assert isinstance(plan.fwd_map, GatherFold)
    assert plan.bwd_map is not None
    return plan.fwd_map.csr, plan.bwd_map.csr


def _set_csr(plan: UDCTPlan, csr: tuple[tf.CsrFold | None, tf.CsrFold | None]) -> None:
    """Swap the CSR maps; the torch tables are built first (from the CSR maps)."""
    assert isinstance(plan.fwd_map, GatherFold)
    assert plan.bwd_map is not None
    plan.fwd_map.torch_tables()
    plan.bwd_map.torch_tables()
    plan.fwd_map.csr, plan.bwd_map.csr = csr


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
    csr = _csr(plan)
    assert csr[0] is not None
    assert csr[1] is not None
    x = _input(kind, (*batch, *shape), dtype)
    coeffs = plan.forward_flat(x)
    recon = plan.backward_flat(coeffs)
    _set_csr(plan, (None, None))
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
    fold = GatherFold.build(targets, sources, weights, num_targets=6, num_sources=6)
    assert fold.csr is not None
    assert fold.csr.row_ptr.tolist() == [0, 2, 2, 4, 4, 7, 7]
    gen = torch.Generator().manual_seed(1)
    src = torch.randn(5, 6, generator=gen, dtype=torch.complex128).to(DEVICE)
    expected = torch.zeros(5, 6, dtype=torch.complex128, device=DEVICE)
    for t, s, wj in zip(targets.tolist(), sources.tolist(), w.tolist(), strict=True):
        im = -2 * wj if pair else wj
        expected[:, t] += wj * src[:, s].real + 1j * im * src[:, s].imag
    torch.testing.assert_close(fold.apply(src), expected)


def test_kernel_is_used_and_falls_back_for_autograd_and_vmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain calls use the kernel; differentiable and vmapped calls use torch ops."""
    calls = []
    real_apply = tf.CsrFold.apply

    def counting(self: tf.CsrFold, src: torch.Tensor) -> torch.Tensor:
        calls.append(src.shape)
        return real_apply(self, src)

    monkeypatch.setattr(tf.CsrFold, "apply", counting)
    udct = _udct((32, 32), 3, 3, "real", torch.float64)
    plan = _plan(udct)
    x = _input("real", (2, 32, 32), torch.float64)
    coeffs = plan.forward_flat(x)
    plan.backward_flat(coeffs)
    assert len(calls) == 2

    calls.clear()
    xg = x.clone().requires_grad_(True)
    plan.backward_flat(plan.forward_flat(xg)).sum().backward()
    assert not calls
    assert xg.grad is not None

    with torch.no_grad():
        plan.forward_flat(xg)
    assert len(calls) == 1

    calls.clear()
    vmapped = torch.vmap(plan.forward_flat)(x)
    assert not calls
    torch.testing.assert_close(vmapped, coeffs)


def test_registered_op_and_compile_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``torch.library`` op (used under ``torch.compile``) matches the eager launch."""
    plan = _plan(_udct((32, 32), 3, 3, "real", torch.float32))
    fwd_csr, bwd_csr = _csr(plan)
    assert fwd_csr is not None
    assert bwd_csr is not None
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


def test_torch_tables_are_built_lazily_from_csr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel plans keep only CSR maps until a differentiable call needs the tables.

    The rebuilt tables equal the ones built directly from the contributions.
    """
    udct = _udct((16, 16, 16), 3, 3, "real", torch.float64)
    plan = _plan(udct)
    folds = [plan.fwd_map, plan.bwd_map]
    assert all(isinstance(f, GatherFold) and f.tables is None for f in folds)
    x = _input("real", (2, 16, 16, 16), torch.float64)
    plan.backward_flat(plan.forward_flat(x))
    assert all(f.tables is None for f in folds)  # type: ignore[union-attr]

    xg = x.clone().requires_grad_(True)
    plan.backward_flat(plan.forward_flat(xg)).sum().backward()
    assert all(f.tables is not None for f in folds)  # type: ignore[union-attr]

    monkeypatch.setenv("CURVELETS_TRITON", "0")
    eager = _plan(udct)
    for lazy, direct in zip(folds, [eager.fwd_map, eager.bwd_map], strict=True):
        assert isinstance(lazy, GatherFold)
        assert isinstance(direct, GatherFold)
        assert direct.csr is None
        assert direct.tables is not None
        torch.testing.assert_close(lazy.src0, direct.src0, rtol=0, atol=0)
        torch.testing.assert_close(lazy.w0, direct.w0, rtol=0, atol=0)
        assert len(lazy.levels) == len(direct.levels)
        for a, b in zip(lazy.levels, direct.levels, strict=True):
            assert a.width == b.width
            for name in ("targets", "sources", "weights"):
                torch.testing.assert_close(
                    getattr(a, name), getattr(b, name), rtol=0, atol=0
                )


def test_disabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CURVELETS_TRITON=0`` builds no CSR maps."""
    monkeypatch.setenv("CURVELETS_TRITON", "0")
    plan = _plan(_udct((32, 32), 3, 3, "real", torch.float32))
    assert _csr(plan) == (None, None)


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
