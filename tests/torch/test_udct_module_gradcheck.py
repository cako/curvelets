"""Gradcheck tests for UDCTModule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import curvelets.torch as torch_curvelets

from .conftest import get_test_configs, get_test_shapes


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("shape_idx", [0])
@pytest.mark.parametrize("cfg_idx", [0])
def test_udct_module_gradcheck(dim: int, shape_idx: int, cfg_idx: int) -> None:
    """
    Test that UDCTModule passes gradcheck for various dimensions.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    shape_idx : int
        Index into test shapes list.
    cfg_idx : int
        Index into test configs list.
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")
    if shape_idx >= len(shapes) or cfg_idx >= len(configs):
        pytest.skip(f"Index out of range for dimension {dim}")

    shape = shapes[shape_idx]
    config = configs[cfg_idx]

    # Create UDCTModule
    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    # Create test input with double precision and requires_grad
    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
    input_tensor = input_data.clone().requires_grad_(True)

    # Run gradcheck with relaxed tolerances
    # UDCT involves FFT operations which can have numerical precision issues
    result = torch.autograd.gradcheck(
        udct_module,
        input_tensor,
        atol=1e-5,
        rtol=1e-3,
        eps=1e-6,
    )

    assert result, "gradcheck failed for UDCTModule"


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_udct_module_gradcheck_multiple_configs(dim: int) -> None:
    """
    Test gradcheck with multiple configurations per dimension.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    # Test with first shape and all available configs
    shape = shapes[0]
    rng = np.random.default_rng(42)

    for cfg_idx, config in enumerate(configs):
        # Create UDCTModule
        udct_module = torch_curvelets.UDCTModule(
            shape=shape,
            angular_wedges_config=config,
            window_overlap=0.15,
            window_threshold=1e-5,
        )

        # Create test input
        input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
        input_tensor = input_data.clone().requires_grad_(True)

        # Run gradcheck
        result = torch.autograd.gradcheck(
            udct_module,
            input_tensor,
            atol=1e-5,
            rtol=1e-3,
            eps=1e-6,
        )

        assert result, f"gradcheck failed for UDCTModule with config {cfg_idx}"


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_udct_module_gradcheck_multiple_shapes(dim: int) -> None:
    """
    Test gradcheck with multiple shapes per dimension.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    """
    shapes = get_test_shapes(dim)
    configs = get_test_configs(dim)

    if not shapes or not configs:
        pytest.skip(f"No test shapes/configs defined for dimension {dim}")

    # Test with first config and all available shapes
    config = configs[0]
    rng = np.random.default_rng(42)

    for shape_idx, shape in enumerate(shapes):
        # Create UDCTModule
        udct_module = torch_curvelets.UDCTModule(
            shape=shape,
            angular_wedges_config=config,
            window_overlap=0.15,
            window_threshold=1e-5,
        )

        # Create test input
        input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
        input_tensor = input_data.clone().requires_grad_(True)

        # Run gradcheck
        result = torch.autograd.gradcheck(
            udct_module,
            input_tensor,
            atol=1e-5,
            rtol=1e-3,
            eps=1e-6,
        )

        assert result, f"gradcheck failed for UDCTModule with shape {shape_idx}"


def test_udct_module_forward_backward_consistency() -> None:
    """Test that forward_nested and backward are consistent with forward."""
    shape = (64, 64)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))

    # Get flattened coefficients from forward()
    flattened = udct_module(input_data)

    # Get nested coefficients from forward_nested()
    nested = udct_module.forward_nested(input_data)

    # Verify they match when flattened
    flattened_from_nested = udct_module.vect(nested)

    assert torch.allclose(flattened, flattened_from_nested, atol=1e-10)

    # Verify backward reconstruction works
    reconstructed = udct_module.backward(nested)
    assert reconstructed.shape == shape


def test_udct_module_gradcheck_small_shape() -> None:
    """Test gradcheck with a small shape to ensure it works for minimal cases."""
    shape = (32, 32)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
    input_tensor = input_data.clone().requires_grad_(True)

    result = torch.autograd.gradcheck(
        udct_module,
        input_tensor,
        atol=1e-5,
        rtol=1e-3,
        eps=1e-6,
    )

    assert result, "gradcheck failed for small shape"


@pytest.mark.parametrize("transform_type", ["real", "complex", "monogenic"])
def test_udct_module_gradcheck_transform_types(transform_type: str) -> None:
    """
    Test gradcheck for all transform types.

    Parameters
    ----------
    transform_type : str
        Transform type to test ("real", "complex", or "monogenic").
    """
    shape = (64, 64)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
        transform_type=transform_type,  # type: ignore[arg-type]
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
    input_tensor = input_data.clone().requires_grad_(True)

    result = torch.autograd.gradcheck(
        udct_module,
        input_tensor,
        atol=1e-5,
        rtol=1e-3,
        eps=1e-6,
    )

    assert result, f"gradcheck failed for transform_type={transform_type}"


@pytest.mark.parametrize("transform_type", ["real", "complex", "monogenic"])
def test_udct_module_forward_backward_consistency_transform_types(
    transform_type: str,
) -> None:
    """
    Test forward/backward consistency for all transform types.

    Parameters
    ----------
    transform_type : str
        Transform type to test ("real", "complex", or "monogenic").
    """
    shape = (64, 64)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
        transform_type=transform_type,  # type: ignore[arg-type]
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))

    # Get flattened coefficients from forward()
    flattened = udct_module(input_data)

    # Get nested coefficients from forward_nested()
    nested = udct_module.forward_nested(input_data)

    # Verify they match when flattened
    flattened_from_nested = udct_module.vect(nested)

    assert torch.allclose(flattened, flattened_from_nested, atol=1e-10)

    # Verify backward reconstruction works
    reconstructed = udct_module.backward(nested)
    if transform_type == "monogenic":
        # Monogenic returns tuple of (scalar, riesz_1, ..., riesz_ndim)
        assert isinstance(reconstructed, tuple)
        assert len(reconstructed) == len(shape) + 1  # ndim + 1 components
        assert all(comp.shape == shape for comp in reconstructed)
    else:
        # Real/complex return single tensor
        assert isinstance(reconstructed, torch.Tensor)
        assert reconstructed.shape == shape


def test_udct_module_complex_transform() -> None:
    """Test complex transform with complex input."""
    shape = (64, 64)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
        transform_type="complex",
    )

    rng = np.random.default_rng(42)
    # Create complex input
    real_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
    imag_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
    input_data = torch.complex(real_part, imag_part)
    input_tensor = input_data.clone().requires_grad_(True)

    # Test forward
    output = udct_module(input_tensor)
    assert output.is_complex()

    # Test gradcheck
    result = torch.autograd.gradcheck(
        udct_module,
        input_tensor,
        atol=1e-5,
        rtol=1e-3,
        eps=1e-6,
    )
    assert result, "gradcheck failed for complex transform"


def test_udct_module_monogenic_transform() -> None:
    """Test monogenic transform produces correct structure."""
    shape = (64, 64)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
        transform_type="monogenic",
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))

    # Test forward_nested returns MUDCTCoefficients
    nested = udct_module.forward_nested(input_data)
    # MUDCTCoefficients structure: list[list[list[list[Tensor]]]]
    # Each inner list has ndim+1 components
    assert isinstance(nested, list)
    assert len(nested) > 0
    # Check first scale has correct structure
    first_scale = nested[0]
    assert isinstance(first_scale, list)
    if len(first_scale) > 0 and len(first_scale[0]) > 0:
        first_wedge = first_scale[0][0]
        assert isinstance(first_wedge, list)
        # Should have ndim+1 components (scalar + ndim Riesz)
        assert len(first_wedge) == len(shape) + 1

    # Test backward returns tuple
    reconstructed = udct_module.backward(nested)
    assert isinstance(reconstructed, tuple)
    assert len(reconstructed) == len(shape) + 1
