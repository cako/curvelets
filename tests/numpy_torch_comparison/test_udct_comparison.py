"""NumPy vs PyTorch comparison tests for UDCT class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import curvelets.numpy as np_curvelets
import curvelets.torch as torch_curvelets

from .conftest import TEST_SHAPES


@pytest.mark.parametrize("ndim", [2, 3])
def test_udct_forward_matches_numpy(ndim):
    """Test UDCT.forward() produces same results as NumPy."""
    shape = TEST_SHAPES[ndim]
    rng = np.random.default_rng(42)
    image_np = rng.normal(size=shape)
    image_torch = torch.from_numpy(image_np)

    angular_config_np = np.array([[3, 3]]) if ndim == 2 else np.array([[3, 3, 3]])
    angular_config_torch = torch.from_numpy(angular_config_np)

    np_udct = np_curvelets.UDCT(
        shape=shape,
        angular_wedges_config=angular_config_np,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    torch_udct = torch_curvelets.UDCT(
        shape=shape,
        angular_wedges_config=angular_config_torch,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    np_coeffs = np_udct.forward(image_np)
    torch_coeffs = torch_udct.forward(image_torch)

    # Check same structure
    assert len(np_coeffs) == len(torch_coeffs)

    for scale_idx in range(len(np_coeffs)):
        assert len(np_coeffs[scale_idx]) == len(torch_coeffs[scale_idx])
        for dir_idx in range(len(np_coeffs[scale_idx])):
            assert len(np_coeffs[scale_idx][dir_idx]) == len(
                torch_coeffs[scale_idx][dir_idx]
            )
            for wedge_idx in range(len(np_coeffs[scale_idx][dir_idx])):
                np_coeff = np_coeffs[scale_idx][dir_idx][wedge_idx]
                torch_coeff = torch_coeffs[scale_idx][dir_idx][wedge_idx]
                np.testing.assert_allclose(
                    np_coeff, torch_coeff.numpy(), atol=1e-5, rtol=1e-4
                )


@pytest.mark.parametrize("ndim", [2, 3])
def test_udct_roundtrip_matches_numpy(ndim):
    """Test UDCT forward+backward roundtrip produces same results as NumPy."""
    shape = TEST_SHAPES[ndim]
    rng = np.random.default_rng(42)
    image_np = rng.normal(size=shape)
    image_torch = torch.from_numpy(image_np)

    angular_config_np = np.array([[3, 3]]) if ndim == 2 else np.array([[3, 3, 3]])
    angular_config_torch = torch.from_numpy(angular_config_np)

    np_udct = np_curvelets.UDCT(
        shape=shape,
        angular_wedges_config=angular_config_np,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    torch_udct = torch_curvelets.UDCT(
        shape=shape,
        angular_wedges_config=angular_config_torch,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    # NumPy roundtrip
    np_coeffs = np_udct.forward(image_np)
    np_recon = np_udct.backward(np_coeffs)

    # PyTorch roundtrip
    torch_coeffs = torch_udct.forward(image_torch)
    torch_recon = torch_udct.backward(torch_coeffs)

    # Both reconstructions should match between NumPy and PyTorch
    # Note: The roundtrip may not be perfect due to thresholding/truncation,
    # but the implementations should produce matching results
    np.testing.assert_allclose(np_recon, torch_recon.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("ndim", [2, 3])
def test_udct_vect_struct_roundtrip(ndim):
    """Test vect/struct roundtrip produces same coefficients."""
    shape = TEST_SHAPES[ndim]
    rng = np.random.default_rng(42)
    image_torch = torch.from_numpy(rng.normal(size=shape))

    angular_config = torch.tensor([[3, 3]]) if ndim == 2 else torch.tensor([[3, 3, 3]])

    torch_udct = torch_curvelets.UDCT(
        shape=shape,
        angular_wedges_config=angular_config,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    coeffs = torch_udct.forward(image_torch)
    vec = torch_udct.vect(coeffs)
    coeffs_reconstructed = torch_udct.struct(vec)

    # Check reconstruction matches original
    for scale_idx in range(len(coeffs)):
        for dir_idx in range(len(coeffs[scale_idx])):
            for wedge_idx in range(len(coeffs[scale_idx][dir_idx])):
                orig = coeffs[scale_idx][dir_idx][wedge_idx]
                recon = coeffs_reconstructed[scale_idx][dir_idx][wedge_idx]
                np.testing.assert_allclose(orig.numpy(), recon.numpy(), atol=1e-10)
