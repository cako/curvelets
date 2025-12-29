"""Integration tests for PyTorch UDCT public methods with complex transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvelets.torch import UDCT


class TestComplexTransformIntegration:
    """Test suite for complex transform edge cases and end-to-end workflows."""

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_round_trip(self, rng, device):
        """
        Test complex transform round-trip with PyTorch UDCT.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        device : str
            Device to run tests on.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]], device=device),
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64)).to(device)
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_wavelet_mode(self, rng, device):
        """
        Test complex transform in wavelet mode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        device : str
            Device to run tests on.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]], device=device),
            high_frequency_mode="wavelet",
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64)).to(device)
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify structure: should have 3 scales (1 lowpass + 2 from angular_wedges_config)
        assert len(coeffs) == 3

        # Verify highest scale has 2*ndim directions (for complex transform)
        highest_scale_idx = 2
        assert len(coeffs[highest_scale_idx]) == 2 * transform.ndim

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_scales", [2, 3, 4, 5])
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_different_scales(self, rng, num_scales, device):
        """
        Test complex transform with different numbers of scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        num_scales : int
            Number of scales to test.
        device : str
            Device to run tests on.
        """
        # Create angular_wedges_config for num_scales
        # Shape is (num_scales - 1, 2) for 2D
        angular_config = torch.tensor([[3, 3]] * (num_scales - 1), device=device)
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=angular_config,
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64)).to(device)
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify structure matches number of scales (1 lowpass + num_scales-1 from config)
        assert len(coeffs) == num_scales

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_vect_struct(self, rng, device):
        """
        Test vect() and struct() with complex transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        device : str
            Device to run tests on.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]], device=device),
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64)).to(device)
        coeffs_orig = transform.forward(data)
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify structure matches
        assert len(coeffs_recon) == len(coeffs_orig)
        for scale_idx in range(len(coeffs_orig)):
            assert len(coeffs_recon[scale_idx]) == len(coeffs_orig[scale_idx])
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                assert len(coeffs_recon[scale_idx][direction_idx]) == len(
                    coeffs_orig[scale_idx][direction_idx]
                )
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]

                    assert isinstance(recon_wedge, torch.Tensor)
                    assert recon_wedge.shape == orig_wedge.shape
                    torch.testing.assert_close(
                        recon_wedge, orig_wedge, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_multidim(self, rng, dim, device):
        """
        Test complex transform with different dimensions.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        dim : int
            Dimension (2, 3, or 4).
        device : str
            Device to run tests on.
        """
        shapes = {
            2: (64, 64),
            3: (32, 32, 32),
            4: (16, 16, 16, 16),
        }
        shape = shapes[dim]

        # Create angular_wedges_config for 3 scales
        ndim = len(shape)
        angular_config = torch.tensor([[3] * ndim, [6] * ndim], device=device)
        transform = UDCT(
            shape=shape,
            angular_wedges_config=angular_config,
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=shape).astype(np.float64)).to(device)
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        # 4D has lower precision due to numerical errors
        atol = 5e-4 if dim == 4 else 1e-4
        rtol = 5e-4 if dim == 4 else 1e-4
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_complex_transform_complex_input(self, rng, device):
        """
        Test complex transform with complex-valued input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        device : str
            Device to run tests on.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]], device=device),
            transform_kind="complex",
        )
        data = torch.from_numpy(
            (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
                np.complex128
            )
        ).to(device)
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon, data, atol=1e-4, rtol=1e-4)
