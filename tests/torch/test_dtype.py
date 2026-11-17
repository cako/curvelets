"""The transform has a fixed precision, set at construction (``dtype``)."""

from __future__ import annotations

import warnings

import pytest
import torch

from curvelets.torch import UDCT, UDCTModule
from curvelets.torch._foldmap import GatherFold, ScatterFold
from curvelets.torch._plan import UDCTPlan

SHAPE = (32, 32)


def _udct(dtype: torch.dtype | None = None, kind: str = "real") -> UDCT:
    return UDCT(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_kind=kind,  # type: ignore[arg-type]
        dtype=dtype,
    )


def _fold_weights(plan: UDCTPlan) -> list[torch.Tensor]:
    folds: list[GatherFold | ScatterFold] = [plan.fwd_map]
    folds += [f for f in (plan.bwd_map, plan.bwd_conj_map) if f is not None]
    folds += [f for chunk in plan.cpu_chunks or [] for f in chunk.bwd]
    weights: list[torch.Tensor] = []
    for fold in folds:
        if isinstance(fold, GatherFold):
            weights += [fold.w0, *(level.weights for level in fold.levels)]
        else:
            weights.append(fold.weights)
    return weights


def test_default_follows_torch_default_dtype() -> None:
    """Without ``dtype``, the transform uses :func:`torch.get_default_dtype`."""
    previous = torch.get_default_dtype()
    try:
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            udct = _udct()
            assert udct.dtype == default
            assert udct.coefficient_dtype == default.to_complex()
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_windows_and_folds_stored_in_transform_precision(
    dtype: torch.dtype, kind: str
) -> None:
    """Window values and all fold weights are stored in the transform's dtype."""
    udct = _udct(dtype, kind)
    values = [w.values for scale in udct.windows for d in scale for w in d]
    assert {v.dtype for v in values} == {dtype}
    plan = udct._get_plan()  # pylint: disable=protected-access
    assert {w.dtype for w in _fold_weights(plan)} == {dtype}


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_single_precision_matches_double(kind: str) -> None:
    """A float32 transform agrees with the float64 one to single precision."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(SHAPE, generator=gen, dtype=torch.float64)
    single, double = _udct(torch.float32, kind), _udct(torch.float64, kind)
    coeffs = single._forward_vect(x.float())  # pylint: disable=protected-access
    assert coeffs.dtype == torch.complex64
    reference = double._forward_vect(x)  # pylint: disable=protected-access
    torch.testing.assert_close(
        coeffs, reference.to(torch.complex64), atol=1e-5, rtol=1e-4
    )
    recon = single.backward(single.forward(x.float()))
    assert recon.dtype == (torch.float32 if kind == "real" else torch.complex64)
    torch.testing.assert_close(recon.real, x.float(), atol=1e-4, rtol=1e-4)


def test_mismatched_input_precision_raises() -> None:
    """Inputs of another precision are rejected with a hint, never cast silently."""
    udct = _udct(torch.float32)
    x = torch.randn(SHAPE, dtype=torch.float64)
    with pytest.raises(TypeError, match=r"dtype=torch\.float64"):
        udct.forward(x)
    with pytest.raises(TypeError, match="float32"):
        udct.filter(x, lambda c: c)
    with pytest.raises(TypeError, match="float32"):
        udct._forward_vect(x)  # pylint: disable=protected-access
    with pytest.raises(TypeError, match="integer|int64"):
        udct.forward(torch.ones(SHAPE, dtype=torch.int64))


def test_mismatched_coefficient_precision_raises() -> None:
    """Backward rejects coefficients of another precision."""
    single, double = _udct(torch.float32), _udct(torch.float64)
    coeffs = double.forward(torch.randn(SHAPE, dtype=torch.float64))
    with pytest.raises(TypeError, match="complex128"):
        single.backward(coeffs)
    with pytest.raises(TypeError, match="complex128"):
        single._backward_vect(double.vect(coeffs))  # pylint: disable=protected-access


def test_complex_transform_accepts_matching_complex_input() -> None:
    """The complex transform takes real or complex input of its precision."""
    udct = _udct(torch.float32, "complex")
    z = torch.randn(SHAPE, dtype=torch.complex64)
    torch.testing.assert_close(udct.backward(udct.forward(z)), z, atol=1e-4, rtol=1e-4)
    with pytest.raises(TypeError):
        udct.forward(z.to(torch.complex128))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32, torch.complex32])  # type: ignore[untyped-decorator]
def test_invalid_dtype_raises(dtype: torch.dtype) -> None:
    """Only float32 and float64 transforms can be built."""
    with pytest.raises(ValueError, match="dtype must be"):
        _udct(dtype)


def test_module_to_dtype_changes_precision() -> None:
    """``UDCTModule.to(dtype)`` / ``.double()`` switch the transform's precision."""
    module = UDCTModule(
        shape=SHAPE, num_scales=3, wedges_per_direction=3, dtype=torch.float32
    )
    x = torch.randn(SHAPE, dtype=torch.float64)
    with pytest.raises(TypeError):
        module(x)
    assert module.float()(x.float()).dtype == torch.complex64
    module = module.double()
    assert module.dtype == torch.float64
    out = module(x)
    assert out.dtype == torch.complex128
    reference = UDCTModule(
        shape=SHAPE, num_scales=3, wedges_per_direction=3, dtype=torch.float64
    )
    torch.testing.assert_close(out, reference(x))
    assert module.to(torch.float32).dtype == torch.float32


def test_module_gradient_has_input_precision() -> None:
    """Gradients flow back in the transform's precision."""
    module = UDCTModule(
        shape=SHAPE, num_scales=3, wedges_per_direction=3, dtype=torch.float32
    )
    x = torch.randn(SHAPE, requires_grad=True, dtype=torch.float32)
    module(x).abs().sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])  # type: ignore[untyped-decorator]
def test_complex_dtype_means_its_real_precision(dtype: torch.dtype) -> None:
    """``dtype=torch.complex64`` builds a float32 transform (PyLops-style)."""
    udct = _udct(dtype, "complex")
    assert udct.dtype == dtype.to_real()
    assert udct.coefficient_dtype == dtype
    z = torch.randn(SHAPE, dtype=dtype)
    coeffs = udct.forward(z)
    assert coeffs[1][0][0].dtype == dtype
    torch.testing.assert_close(udct.backward(coeffs), z, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["real", "complex"])  # type: ignore[untyped-decorator]
def test_module_to_complex_dtype_sets_real_precision(kind: str) -> None:
    """``UDCTModule.to(torch.complex128)`` keeps the windows real, in float64."""
    module = UDCTModule(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_type=kind,  # type: ignore[arg-type]
        dtype=torch.float32,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns that complex modules are new
        module = module.to(torch.complex128)
    assert module.dtype == torch.float64
    windows = module._udct.windows  # pylint: disable=protected-access
    assert all(not w.values.is_complex() for s in windows for d in s for w in d)
    x = torch.randn(SHAPE, dtype=torch.float64)
    assert module(x).dtype == torch.complex128


def test_complex_module_gradient_for_complex_input() -> None:
    """Complex inputs get complex gradients of the transform's precision."""
    module = UDCTModule(
        shape=SHAPE,
        num_scales=3,
        wedges_per_direction=3,
        transform_type="complex",
        dtype=torch.float32,
    )
    z = torch.randn(SHAPE, dtype=torch.complex64, requires_grad=True)
    module(z).abs().sum().backward()
    assert z.grad is not None
    assert z.grad.dtype == torch.complex64


def test_monogenic_dtypes_and_unimplemented_paths() -> None:
    """Monogenic coefficients are real; its transforms are not implemented yet."""
    udct = _udct(torch.float32, "monogenic")
    assert udct.dtype == torch.float32
    assert udct.coefficient_dtype == torch.float32
    x = torch.randn(SHAPE, dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        udct.forward(x)
    with pytest.raises(NotImplementedError):
        udct.backward([[[x]]])
    with pytest.raises(NotImplementedError):
        udct._forward_vect(x)  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="real-valued input"):
        udct.forward(x.to(torch.complex64))
