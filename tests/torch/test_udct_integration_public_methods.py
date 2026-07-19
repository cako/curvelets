"""Integration tests for PyTorch UDCT public methods with complex transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvelets.torch import UDCT


class TestComplexTransformIntegration:
    """Test suite for complex transform edge cases and end-to-end workflows."""

    def test_complex_transform_round_trip(self, rng):
        """
        Test complex transform round-trip with PyTorch UDCT.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]]),
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64))
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=1e-4, rtol=1e-4)

    def test_complex_transform_wavelet_mode(self, rng):
        """
        Test complex transform in wavelet mode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]]),
            high_frequency_mode="wavelet",
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64))
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
    def test_complex_transform_different_scales(self, rng, num_scales):
        """
        Test complex transform with different numbers of scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        num_scales : int
            Number of scales to test.
        """
        # Create angular_wedges_config for num_scales
        # Shape is (num_scales - 1, 2) for 2D
        angular_config = torch.tensor([[3, 3]] * (num_scales - 1))
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=angular_config,
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64))
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify structure matches number of scales (1 lowpass + num_scales-1 from config)
        assert len(coeffs) == num_scales

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=1e-4, rtol=1e-4)

    def test_complex_transform_vect_struct(self, rng):
        """
        Test vect() and struct() with complex transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]]),
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=(64, 64)).astype(np.float64))
        coeffs_orig = transform.forward(data)
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Type narrowing: transform returns UDCTCoefficients
        assert isinstance(coeffs_recon, list)  # Both are lists, but helps mypy
        assert isinstance(coeffs_orig, list)  # Both are lists, but helps mypy

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
                    assert isinstance(orig_wedge, torch.Tensor)
                    # Type narrowing: we know these are UDCTCoefficients (Tensor)
                    assert recon_wedge.shape == orig_wedge.shape
                    torch.testing.assert_close(
                        recon_wedge, orig_wedge, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_complex_transform_multidim(self, rng, dim):
        """
        Test complex transform with different dimensions.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        dim : int
            Dimension (2, 3, or 4).
        """
        shapes = {
            2: (64, 64),
            3: (32, 32, 32),
            4: (16, 16, 16, 16),
        }
        shape = shapes[dim]

        # Create angular_wedges_config for 3 scales
        ndim = len(shape)
        angular_config = torch.tensor([[3] * ndim, [6] * ndim])
        transform = UDCT(
            shape=shape,
            angular_wedges_config=angular_config,
            transform_kind="complex",
        )
        data = torch.from_numpy(rng.normal(size=shape).astype(np.float64))
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        # 4D has lower precision due to numerical errors
        atol = 5e-4 if dim == 4 else 1e-4
        rtol = 5e-4 if dim == 4 else 1e-4
        assert recon.shape == data.shape
        torch.testing.assert_close(recon.real, data, atol=atol, rtol=rtol)

    def test_complex_transform_complex_input(self, rng):
        """
        Test complex transform with complex-valued input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            angular_wedges_config=torch.tensor([[3, 3], [6, 6]]),
            transform_kind="complex",
        )
        data = torch.from_numpy(
            (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
                np.complex128
            )
        )
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)

        # Verify reconstruction accuracy
        assert recon.shape == data.shape
        torch.testing.assert_close(recon, data, atol=1e-4, rtol=1e-4)


class TestCoefficientShapes:
    """Test suite for PyTorch UDCT coefficient_shapes() method across modes and dimensions."""

    @pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])
    @pytest.mark.parametrize("high_freq_mode", ["curvelet", "wavelet"])
    def test_coefficient_shapes_2d(self, transform_kind, high_freq_mode):
        """
        Test that coefficient_shapes() exactly matches actual forward transform shapes in 2D.

        Parameters
        ----------
        transform_kind : str
            Transform kind ("real", "complex", "monogenic").
        high_freq_mode : str
            High frequency mode ("curvelet" or "wavelet").
        """
        shape = (64, 64)
        transform = UDCT(
            shape=shape,
            num_scales=3,
            wedges_per_direction=3,
            high_frequency_mode=high_freq_mode,
            transform_kind=transform_kind,
        )
        predicted_shapes = transform.coefficient_shapes()
        if transform_kind == "monogenic":
            total_size = sum(
                int(np.prod(w)) for s in predicted_shapes for d in s for w in d
            )
            vec = torch.zeros(total_size, dtype=torch.float64)
            actual_coeffs = transform.struct(vec)
        else:
            actual_coeffs = transform.forward(torch.zeros(shape, dtype=torch.float64))
        actual_shapes = [
            [[tuple(wedge.shape) for wedge in direction] for direction in scale]
            for scale in actual_coeffs
        ]
        assert predicted_shapes == actual_shapes

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_coefficient_shapes_multidim(self, dim):
        """
        Test that coefficient_shapes() matches actual forward transform shapes for 2D/3D/4D.

        Parameters
        ----------
        dim : int
            Dimension (2, 3, or 4).
        """
        shapes_map = {
            2: (64, 64),
            3: (32, 32, 32),
            4: (16, 16, 16, 16),
        }
        shape = shapes_map[dim]
        transform = UDCT(shape=shape, num_scales=2, wedges_per_direction=3)
        predicted_shapes = transform.coefficient_shapes()
        actual_coeffs = transform.forward(torch.zeros(shape, dtype=torch.float64))
        actual_shapes = [
            [[tuple(wedge.shape) for wedge in direction] for direction in scale]
            for scale in actual_coeffs
        ]
        assert predicted_shapes == actual_shapes

    @pytest.mark.parametrize("transform_type", ["real", "complex"])
    def test_coefficient_shapes_udct_module(self, transform_type):
        """
        Test that UDCTModule.coefficient_shapes() delegates and matches actual forward output.

        Parameters
        ----------
        transform_type : str
            Transform type ("real" or "complex").
        """
        from curvelets.torch import UDCTModule

        shape = (64, 64)
        module = UDCTModule(
            shape=shape,
            num_scales=3,
            wedges_per_direction=3,
            transform_type=transform_type,
        )
        predicted_shapes = module.coefficient_shapes()
        actual_vec = module.forward(torch.zeros(shape, dtype=torch.float64))
        actual_coeffs = module.struct(actual_vec)
        actual_shapes = [
            [[tuple(wedge.shape) for wedge in direction] for direction in scale]
            for scale in actual_coeffs
        ]
        assert predicted_shapes == actual_shapes

