"""Tests for error paths in numpy module."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from curvelets.numpy import UDCT, MeyerWavelet


class TestUDCTWindowOverlapWarnings:
    """Test suite for window overlap validation warnings."""

    def test_window_overlap_warning_triggered(self, caplog):
        """
        Test that window overlap validation warnings are logged.

        Parameters
        ----------
        caplog : pytest.LogCaptureFixture
            Pytest fixture for capturing log messages.
        """
        # Create a scenario where const >= num_wedges to trigger warning
        # This happens when window_overlap is too large relative to num_wedges
        # We need: (2^(scale_idx/num_wedges)) * (1+2a) * (1+a) >= num_wedges
        # For scale_idx=1, num_wedges=3, we need a large window_overlap
        with caplog.at_level(logging.WARNING):
            # Use a large window_overlap that will trigger the warning
            # For scale 1 with 3 wedges: const = 2^(1/3) * (1+2a) * (1+a)
            # With a=0.5: const ≈ 1.26 * 2 * 1.5 = 3.78, which is >= 3
            _ = UDCT(
                shape=(64, 64),
                num_scales=3,
                wedges_per_direction=3,
                window_overlap=0.5,  # Large overlap to trigger warning
            )

        # Check that warning was logged
        assert len(caplog.records) > 0
        warning_messages = [record.message for record in caplog.records]
        assert any("window_overlap" in msg for msg in warning_messages)

    def test_window_overlap_warning_not_triggered(self, caplog):
        """
        Test that warnings are not triggered with valid window_overlap.

        Parameters
        ----------
        caplog : pytest.LogCaptureFixture
            Pytest fixture for capturing log messages.
        """
        with caplog.at_level(logging.WARNING):
            # Use a small window_overlap that should not trigger warning
            _ = UDCT(
                shape=(64, 64),
                num_scales=3,
                wedges_per_direction=3,
                window_overlap=0.15,  # Small overlap, should not trigger warning
            )

        # Check that no warnings were logged (or only unrelated warnings)
        warning_messages = [record.message for record in caplog.records]
        assert not any("window_overlap" in msg for msg in warning_messages)


class TestMeyerWaveletErrors:
    """Test suite for MeyerWavelet error cases."""

    def test_backward_before_forward(self, rng):
        """
        Test that calling backward() with invalid structure raises ValueError.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> invalid_coeffs = [[np.random.randn(32, 32)]]  # Missing highpass bands
        >>> try:
        ...     wavelet.backward(invalid_coeffs)
        ... except ValueError as e:
        ...     print("Error caught:", str(e))
        Error caught: coefficients must have 2 subband groups...
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        # Invalid structure: only 1 subband group instead of 2
        invalid_coeffs = [[rng.normal(size=(32, 32)).astype(np.float64)]]

        with pytest.raises(ValueError, match="coefficients must have 2 subband groups"):
            wavelet.backward(invalid_coeffs)

    def test_forward_shape_mismatch(self, rng):
        """
        Test that calling forward() with wrong shape raises ValueError.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> wrong_shape_signal = np.random.randn(32, 32)
        >>> try:
        ...     wavelet.forward(wrong_shape_signal)
        ... except ValueError as e:
        ...     print("Error caught:", str(e))
        Error caught: Signal shape (32, 32) does not match expected shape (64, 64)
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        wrong_shape_signal = rng.normal(size=(32, 32)).astype(np.float64)

        with pytest.raises(
            ValueError, match=r"Signal shape.*does not match expected shape"
        ):
            wavelet.forward(wrong_shape_signal)

    def test_forward_shape_mismatch_3d(self, rng):
        """
        Test shape mismatch error with 3D data.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(32, 32, 32))
        wrong_shape_signal = rng.normal(size=(16, 16, 16)).astype(np.float64)

        with pytest.raises(
            ValueError, match=r"Signal shape.*does not match expected shape"
        ):
            wavelet.forward(wrong_shape_signal)


class TestUDCTErrors:
    """Test suite for UDCT error cases."""

    def test_forward_shape_mismatch(self, rng):
        """
        Test that calling forward() with shape mismatch raises AssertionError.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> wrong_shape_data = np.random.randn(32, 32)
        >>> try:
        ...     transform.forward(wrong_shape_data)
        ... except AssertionError:
        ...     print("Error caught: shape mismatch")
        Error caught: shape mismatch
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        wrong_shape_data = rng.normal(size=(32, 32)).astype(np.float64)

        with pytest.raises(AssertionError):
            transform.forward(wrong_shape_data)

    def test_forward_shape_mismatch_3d(self, rng):
        """
        Test shape mismatch error with 3D data.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(32, 32, 32), num_scales=2, wedges_per_direction=3)
        wrong_shape_data = rng.normal(size=(16, 16, 16)).astype(np.float64)

        with pytest.raises(AssertionError):
            transform.forward(wrong_shape_data)

    def test_wavelet_mode_meyer_wavelet_none_edge_case(self, rng):
        """
        Test RuntimeError when _meyer_wavelet is None in wavelet mode.

        This is an edge case that could occur if the wavelet object is somehow
        not initialized. We test this by manually setting it to None.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3,
        ...                  high_frequency_mode="meyer")
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> # Manually break the wavelet object (edge case)
        >>> transform._meyer_wavelet = None
        >>> try:
        ...     transform.backward(coeffs)
        ... except RuntimeError as e:
        ...     print("Error caught:", str(e))
        Error caught: MeyerWavelet not initialized
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            high_frequency_mode="meyer",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs = transform.forward(data)

        # Manually set _meyer_wavelet to None to simulate edge case
        transform._meyer_wavelet = None

        with pytest.raises(RuntimeError, match="MeyerWavelet not initialized"):
            transform.backward(coeffs)

    def test_wavelet_mode_forward_meyer_wavelet_none(self, rng):
        """
        Test RuntimeError when _meyer_wavelet is None in forward().

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            high_frequency_mode="meyer",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)

        # Manually set _meyer_wavelet to None to simulate edge case
        transform._meyer_wavelet = None

        with pytest.raises(RuntimeError, match="MeyerWavelet not initialized"):
            transform.forward(data)

    def test_meyer_multiple_forward_calls(self, rng):
        """
        Test that multiple forward() calls work correctly without data corruption.

        This test verifies the fix for the bug where calling forward() on multiple
        images before calling backward() would cause silent data corruption because
        highpass bands were stored as instance state and got overwritten.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=2, wedges_per_direction=3,
        ...                  high_frequency_mode="meyer")
        >>> image_A = np.random.randn(64, 64)
        >>> image_B = np.random.randn(64, 64) + 100
        >>> coeffs_A = transform.forward(image_A)
        >>> coeffs_B = transform.forward(image_B)
        >>> recon_A = transform.backward(coeffs_A)
        >>> recon_B = transform.backward(coeffs_B)
        >>> np.allclose(image_A, recon_A, atol=1e-4)
        True
        >>> np.allclose(image_B, recon_B, atol=1e-4)
        True
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=2,
            wedges_per_direction=3,
            high_frequency_mode="meyer",
        )

        # Create two different images
        image_A = rng.normal(size=(64, 64)).astype(np.float64)
        image_B = rng.normal(size=(64, 64)).astype(np.float64) + 100.0

        # Call forward on both images (this was the bug scenario)
        coeffs_A = transform.forward(image_A)
        coeffs_B = transform.forward(image_B)

        # Verify that each set of coefficients has its own highpass bands
        assert hasattr(coeffs_A, "_meyer_highpass_bands")
        assert hasattr(coeffs_B, "_meyer_highpass_bands")
        assert coeffs_A._meyer_highpass_bands is not None
        assert coeffs_B._meyer_highpass_bands is not None

        # Now backward should use the correct highpass bands for each
        recon_A = transform.backward(coeffs_A)
        recon_B = transform.backward(coeffs_B)

        # Verify reconstruction accuracy
        assert np.allclose(image_A, recon_A, atol=1e-4), (
            "Image A should be correctly reconstructed even after forward(B)"
        )
        assert np.allclose(image_B, recon_B, atol=1e-4), (
            "Image B should be correctly reconstructed"
        )

        # Verify that reconstructions are different (not corrupted)
        assert not np.allclose(recon_A, recon_B), (
            "Reconstructions should be different for different inputs"
        )

    def test_meyer_backward_missing_highpass_bands_attribute(self, rng):
        """
        Test RuntimeError when coefficients are missing _meyer_highpass_bands attribute.

        This can occur if coefficients were created with an older version or were
        modified incorrectly.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=2, wedges_per_direction=3,
        ...                  high_frequency_mode="meyer")
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> delattr(coeffs, '_meyer_highpass_bands')
        >>> try:
        ...     transform.backward(coeffs)
        ... except RuntimeError as e:
        ...     print("Error caught:", str(e))
        Error caught: Coefficients missing highpass bands attribute...
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=2,
            wedges_per_direction=3,
            high_frequency_mode="meyer",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs = transform.forward(data)

        # Remove the attribute to simulate old version or manual modification
        if hasattr(coeffs, "_meyer_highpass_bands"):
            delattr(coeffs, "_meyer_highpass_bands")

        with pytest.raises(
            RuntimeError, match="Coefficients missing highpass bands attribute"
        ):
            transform.backward(coeffs)

    def test_meyer_backward_none_highpass_bands(self, rng):
        """
        Test RuntimeError when _meyer_highpass_bands attribute is None.

        This can occur if coefficients were modified incorrectly.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=2, wedges_per_direction=3,
        ...                  high_frequency_mode="meyer")
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> coeffs._meyer_highpass_bands = None
        >>> try:
        ...     transform.backward(coeffs)
        ... except RuntimeError as e:
        ...     print("Error caught:", str(e))
        Error caught: Highpass bands are not available in coefficients...
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=2,
            wedges_per_direction=3,
            high_frequency_mode="meyer",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs = transform.forward(data)

        # Set attribute to None to simulate incomplete coefficients
        coeffs._meyer_highpass_bands = None

        with pytest.raises(
            RuntimeError, match="Highpass bands are not available in coefficients"
        ):
            transform.backward(coeffs)
