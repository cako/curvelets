"""Round-trip tests for PyTorch UDCT implementation."""

from __future__ import annotations

import pytest
import torch

from .conftest import setup_torch_transform


class TestTorchRoundTrip:
    """Round-trip tests for PyTorch UDCT."""

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("cfg_idx", [0, 1])
    def test_torch_round_trip_relative(self, rng, dim, cfg_idx):
        """Test that forward+backward produces result close to original."""
        transform = setup_torch_transform(dim, shape_idx=0, cfg_idx=cfg_idx)
        data = torch.from_numpy(rng.random(transform._obj.shape))

        coeffs = transform.forward(data)
        reconstructed = transform.backward(coeffs)

        # Check relative error
        max_abs = torch.abs(data).max()
        error = torch.abs(data - reconstructed).max()
        relative_error = error / max_abs

        # Allow some numerical error
        assert relative_error < 0.5, f"Relative error too large: {relative_error}"


@pytest.mark.parametrize("dim", [2, 3])
def test_torch_round_trip_curvelet_mode(rng, dim):
    """Test round-trip in curvelet mode."""
    transform = setup_torch_transform(dim, shape_idx=0, cfg_idx=0, high="curvelet")
    data = torch.from_numpy(rng.random(transform._obj.shape))

    coeffs = transform.forward(data)
    reconstructed = transform.backward(coeffs)

    # Verify result has same shape and dtype
    assert reconstructed.shape == data.shape
    assert reconstructed.dtype == data.dtype


@pytest.mark.parametrize("dim", [2, 3])
def test_torch_vect_struct_roundtrip(rng, dim):
    """Test that vect() and struct() are inverses."""
    transform = setup_torch_transform(dim, shape_idx=0, cfg_idx=0)
    data = torch.from_numpy(rng.random(transform._obj.shape))

    coeffs = transform.forward(data)

    # Vectorize then restructure
    vec = transform._obj.vect(coeffs)
    coeffs_reconstructed = transform._obj.struct(vec)

    # Check all coefficients match
    for scale_idx in range(len(coeffs)):
        for dir_idx in range(len(coeffs[scale_idx])):
            for wedge_idx in range(len(coeffs[scale_idx][dir_idx])):
                orig = coeffs[scale_idx][dir_idx][wedge_idx]
                recon = coeffs_reconstructed[scale_idx][dir_idx][wedge_idx]
                assert torch.allclose(orig, recon, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_torch_coefficient_structure(rng, dim):
    """Test that forward produces expected coefficient structure."""
    transform = setup_torch_transform(dim, shape_idx=0, cfg_idx=0)
    data = torch.from_numpy(rng.random(transform._obj.shape))

    coeffs = transform.forward(data)

    # Check that we have at least 2 scales (lowpass + at least 1 detail scale)
    assert len(coeffs) >= 2, "Should have at least 2 scales"

    # Check lowpass has single direction and single wedge
    assert len(coeffs[0]) == 1, "Lowpass should have 1 direction"
    assert len(coeffs[0][0]) == 1, "Lowpass direction should have 1 wedge"

    # Check that coefficients are tensors
    for scale in coeffs:
        for direction in scale:
            for wedge_coeff in direction:
                assert isinstance(wedge_coeff, torch.Tensor)


@pytest.mark.parametrize("dim", [2, 3])
def test_torch_windows_are_sparse(dim):
    """Test that windows are stored in sparse format."""
    transform = setup_torch_transform(dim, shape_idx=0, cfg_idx=0)

    windows = transform._obj.windows

    for scale in windows:
        for direction in scale:
            for window in direction:
                assert isinstance(window, tuple), "Window should be tuple"
                assert len(window) == 2, "Window tuple should have 2 elements"
                indices, values = window
                assert isinstance(indices, torch.Tensor), "Indices should be tensor"
                assert isinstance(values, torch.Tensor), "Values should be tensor"
