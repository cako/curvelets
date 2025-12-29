"""Tests for UDCT._compute_optimal_window_overlap method."""

from __future__ import annotations

import numpy as np

from curvelets.numpy import UDCT


class TestComputeOptimalWindowOverlap:
    """Test suite for UDCT._compute_optimal_window_overlap static method."""

    def test_basic_computation_wpd3(self):
        """
        Test optimal window_overlap computation for wedges_per_direction=3.

        The computed value should be approximately 0.037 (10% of theoretical max).
        """
        num_scales = 3
        wedges_per_scale = (3 * 2 ** np.arange(num_scales - 1)).astype(int)
        alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)

        # Expected value around 0.037 based on formula
        assert 0.03 < alpha < 0.05

    def test_basic_computation_wpd6(self):
        """
        Test optimal window_overlap computation for wedges_per_direction=6.

        The computed value should be approximately 0.09 (10% of theoretical max).
        """
        num_scales = 3
        wedges_per_scale = (6 * 2 ** np.arange(num_scales - 1)).astype(int)
        alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)

        # Expected value around 0.09 based on formula
        assert 0.08 < alpha < 0.10

    def test_basic_computation_wpd12(self):
        """
        Test optimal window_overlap computation for wedges_per_direction=12.

        The computed value should be approximately 0.16 (10% of theoretical max).
        """
        num_scales = 3
        wedges_per_scale = (12 * 2 ** np.arange(num_scales - 1)).astype(int)
        alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)

        # Expected value around 0.16 based on formula
        assert 0.15 < alpha < 0.18

    def test_satisfies_constraint(self):
        """
        Test that computed values satisfy the Nguyen & Chauris constraint.

        The constraint is: (2^(s/N))(1+2a)(1+a) < N for all scales.
        """
        for wpd in [3, 6, 9, 12]:
            num_scales = 3
            wedges_per_scale = (wpd * 2 ** np.arange(num_scales - 1)).astype(int)
            alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)

            # Verify constraint is satisfied for all scales
            for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
                k = 2 ** (scale_idx / num_wedges)
                constraint_value = k * (1 + 2 * alpha) * (1 + alpha)
                assert constraint_value < num_wedges, (
                    f"Constraint violated for wpd={wpd}, scale_idx={scale_idx}: "
                    f"{constraint_value:.4f} >= {num_wedges}"
                )

    def test_alpha_increases_with_wpd(self):
        """
        Test that computed alpha increases with wedges_per_direction.

        More wedges means looser constraint, allowing larger alpha.
        """
        num_scales = 3
        alphas = []
        for wpd in [3, 6, 9, 12]:
            wedges_per_scale = (wpd * 2 ** np.arange(num_scales - 1)).astype(int)
            alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
            alphas.append(alpha)

        # Verify alpha is monotonically increasing
        for i in range(len(alphas) - 1):
            assert alphas[i] < alphas[i + 1], (
                f"Alpha should increase: {alphas[i]:.4f} >= {alphas[i + 1]:.4f}"
            )

    def test_alpha_positive(self):
        """Test that computed alpha is always positive."""
        for wpd in [3, 6, 9, 12, 15, 18]:
            for num_scales in [2, 3, 4, 5]:
                wedges_per_scale = (wpd * 2 ** np.arange(num_scales - 1)).astype(int)
                alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
                assert alpha > 0, f"Alpha should be positive: {alpha}"

    def test_different_num_scales(self):
        """
        Test that alpha is consistent across different num_scales.

        The first scale typically dominates, so alpha should be similar.
        """
        wpd = 6
        alphas = []
        for num_scales in [2, 3, 4, 5]:
            wedges_per_scale = (wpd * 2 ** np.arange(num_scales - 1)).astype(int)
            alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
            alphas.append(alpha)

        # All alphas should be similar (first scale dominates)
        for alpha in alphas:
            assert abs(alpha - alphas[0]) < 0.01, (
                f"Alpha should be consistent across num_scales: {alphas}"
            )


class TestAutoWindowOverlapIntegration:
    """Test suite for auto window_overlap in UDCT initialization."""

    def test_auto_overlap_with_num_scales(self, rng):
        """
        Test that UDCT uses auto-computed window_overlap when None is provided.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        # Create transform without specifying window_overlap
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)

        # Verify window_overlap was computed
        assert transform.parameters.window_overlap is not None
        assert transform.parameters.window_overlap > 0

        # Verify reconstruction works
        data = rng.standard_normal((64, 64))
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)
        error = np.max(np.abs(data - recon))
        assert error < 1e-3, f"Reconstruction error too high: {error}"

    def test_auto_overlap_with_angular_config(self, rng):
        """
        Test auto window_overlap with angular_wedges_config interface.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        angular_wedges_config = np.array([[3, 3], [6, 6]])
        transform = UDCT(shape=(64, 64), angular_wedges_config=angular_wedges_config)

        # Verify window_overlap was computed
        assert transform.parameters.window_overlap is not None
        assert transform.parameters.window_overlap > 0

        # Verify reconstruction works
        data = rng.standard_normal((64, 64))
        coeffs = transform.forward(data)
        recon = transform.backward(coeffs)
        error = np.max(np.abs(data - recon))
        assert error < 1e-3, f"Reconstruction error too high: {error}"

    def test_explicit_overlap_overrides_auto(self):
        """
        Test that explicit window_overlap overrides auto computation.
        """
        explicit_overlap = 0.2
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            window_overlap=explicit_overlap,
        )

        # Verify explicit value was used
        assert transform.parameters.window_overlap == explicit_overlap

    def test_reconstruction_quality_various_wpd(self, rng):
        """
        Test reconstruction quality with auto-computed window_overlap.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        for wpd in [3, 6, 12]:
            transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=wpd)
            data = rng.standard_normal((64, 64))
            coeffs = transform.forward(data)
            recon = transform.backward(coeffs)
            error = np.max(np.abs(data - recon))

            # Error threshold depends on wpd
            threshold = 0.1 if wpd <= 6 else 0.15
            assert error < threshold, (
                f"Reconstruction error too high for wpd={wpd}: {error}"
            )
