"""Comprehensive tests for MeyerWavelet class."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import MeyerWavelet


class TestMeyerWavelet1DTransform:
    """Test suite for 1D forward and inverse transforms."""

    def test_forward_transform_1d_real(self, rng):
        """
        Test 1D forward transform with real input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> lowpass.shape
        (32, 64)
        >>> highpass.shape
        (32, 64)
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)

        # Verify shapes are halved along the transform axis
        assert lowpass.shape == (32, 64)
        assert highpass.shape == (32, 64)

        # Verify output is real
        assert not np.iscomplexobj(lowpass)
        assert not np.iscomplexobj(highpass)

        # Verify dtype preservation
        assert lowpass.dtype == signal.dtype
        assert highpass.dtype == signal.dtype

    def test_forward_transform_1d_complex(self, rng):
        """
        Test 1D forward transform with complex input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> np.iscomplexobj(lowpass)
        True
        >>> np.iscomplexobj(highpass)
        True
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)

        # Verify shapes are halved along the transform axis
        assert lowpass.shape == (32, 64)
        assert highpass.shape == (32, 64)

        # Verify output is complex
        assert np.iscomplexobj(lowpass)
        assert np.iscomplexobj(highpass)

        # Verify dtype preservation
        assert lowpass.dtype == signal.dtype
        assert highpass.dtype == signal.dtype

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_forward_transform_1d_dtype_preservation(self, rng, dtype):
        """
        Test dtype preservation in 1D forward transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        dtype : numpy.dtype
            Data type to test.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        if np.iscomplexobj(np.array(0, dtype=dtype)):
            signal = (
                rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
            ).astype(dtype)
        else:
            signal = rng.normal(size=(64, 64)).astype(dtype)

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)

        # Verify dtype preservation
        assert lowpass.dtype == dtype
        assert highpass.dtype == dtype

    def test_inverse_transform_1d_real(self, rng):
        """
        Test 1D inverse transform with real input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)
        >>> np.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)

        # Verify shape is restored
        assert reconstructed.shape == signal.shape

        # Verify output is real
        assert not np.iscomplexobj(reconstructed)

        # Verify reconstruction accuracy
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_inverse_transform_1d_complex(self, rng):
        """
        Test 1D inverse transform with complex input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)

        # Verify shape is restored
        assert reconstructed.shape == signal.shape

        # Verify output is complex
        assert np.iscomplexobj(reconstructed)

        # Verify reconstruction accuracy
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_inverse_transform_1d_mixed_complex(self, rng):
        """
        Test inverse transform with one complex and one real subband.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        lowpass, highpass = wavelet._forward_transform_1d(signal, 0)

        # Make highpass complex (simulating mixed scenario)
        highpass_complex = highpass.astype(np.complex128)

        reconstructed = wavelet._inverse_transform_1d(lowpass, highpass_complex, 0)

        # Should handle mixed types gracefully
        assert reconstructed.shape == signal.shape


class TestMeyerWaveletMultiDimensional:
    """Test suite for multi-dimensional forward and backward transforms."""

    def test_forward_2d(self, rng):
        """
        Test full 2D forward transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> coefficients = wavelet.forward(signal)
        >>> len(coefficients)  # 2 subband groups
        2
        >>> coefficients[0][0].shape  # Lowpass subband
        (32, 32)
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        coefficients = wavelet.forward(signal)

        # Verify structure: 2 subband groups
        assert len(coefficients) == 2

        # First group: lowpass (1 subband)
        assert len(coefficients[0]) == 1
        assert coefficients[0][0].shape == (32, 32)

        # Second group: highpass (2^2 - 1 = 3 subbands)
        assert len(coefficients[1]) == 3
        for subband in coefficients[1]:
            assert subband.shape == (32, 32)

    def test_forward_3d(self, rng):
        """
        Test full 3D forward transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(32, 32, 32))
        signal = rng.normal(size=(32, 32, 32)).astype(np.float64)

        coefficients = wavelet.forward(signal)

        # Verify structure: 2 subband groups
        assert len(coefficients) == 2

        # First group: lowpass (1 subband)
        assert len(coefficients[0]) == 1
        assert coefficients[0][0].shape == (16, 16, 16)

        # Second group: highpass (2^3 - 1 = 7 subbands)
        assert len(coefficients[1]) == 7
        for subband in coefficients[1]:
            assert subband.shape == (16, 16, 16)

    def test_forward_complex_input(self, rng):
        """
        Test forward transform with complex input arrays.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )

        coefficients = wavelet.forward(signal)

        # Verify structure
        assert len(coefficients) == 2
        assert len(coefficients[0]) == 1
        assert len(coefficients[1]) == 3

        # Verify all subbands are complex
        assert np.iscomplexobj(coefficients[0][0])
        for subband in coefficients[1]:
            assert np.iscomplexobj(subband)

    def test_backward_2d(self, rng):
        """
        Test full 2D backward transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        # Verify shape is restored
        assert reconstructed.shape == signal.shape

        # Verify reconstruction accuracy
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_backward_3d(self, rng):
        """
        Test full 3D backward transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(32, 32, 32))
        signal = rng.normal(size=(32, 32, 32)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        # Verify shape is restored
        assert reconstructed.shape == signal.shape

        # Verify reconstruction accuracy
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_backward_complex_input(self, rng):
        """
        Test backward transform with complex coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        # Verify shape is restored
        assert reconstructed.shape == signal.shape

        # Verify output is complex
        assert np.iscomplexobj(reconstructed)

        # Verify reconstruction accuracy
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)


class TestMeyerWaveletRoundTrip:
    """Test suite for round-trip accuracy."""

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_round_trip_different_dtypes(self, rng, dtype):
        """
        Test round-trip with different dtypes.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        dtype : numpy.dtype
            Data type to test.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        if np.iscomplexobj(np.array(0, dtype=dtype)):
            signal = (
                rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
            ).astype(dtype)
        else:
            signal = rng.normal(size=(64, 64)).astype(dtype)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        # Verify shape
        assert reconstructed.shape == signal.shape

        # Verify dtype preservation
        assert reconstructed.dtype == dtype

        # Verify reconstruction accuracy (tolerance depends on dtype)
        atol = 1e-5 if dtype in (np.float32, np.complex64) else 1e-10
        np.testing.assert_allclose(signal, reconstructed, atol=atol)

    def test_round_trip_2d(self, rng):
        """
        Round-trip test for 2D signals.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_round_trip_3d(self, rng):
        """
        Round-trip test for 3D signals.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(32, 32, 32))
        signal = rng.normal(size=(32, 32, 32)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_round_trip_4d(self, rng):
        """
        Round-trip test for 4D signals.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(16, 16, 16, 16))
        signal = rng.normal(size=(16, 16, 16, 16)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        # Verify structure: 2 subband groups
        assert len(coefficients) == 2

        # First group: lowpass (1 subband)
        assert len(coefficients[0]) == 1
        assert coefficients[0][0].shape == (8, 8, 8, 8)

        # Second group: highpass (2^4 - 1 = 15 subbands)
        assert len(coefficients[1]) == 15

        # Verify reconstruction
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_round_trip_real(self, rng):
        """
        Test forward+backward round-trip with real input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = rng.normal(size=(64, 64)).astype(np.float64)

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_round_trip_complex(self, rng):
        """
        Test forward+backward round-trip with complex input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )

        coefficients = wavelet.forward(signal)
        reconstructed = wavelet.backward(coefficients)

        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)


class TestMeyerWaveletInternal:
    """Test suite for internal methods."""

    def test_filter_computation(self):
        """
        Test _compute_single_filter() directly.
        """
        wavelet = MeyerWavelet(shape=(64, 64))
        signal_length = 64

        lowpass_filter, highpass_filter = wavelet._compute_single_filter(signal_length)

        # Verify filters are 1D arrays
        assert lowpass_filter.ndim == 1
        assert highpass_filter.ndim == 1

        # Verify filter lengths
        assert len(lowpass_filter) == signal_length
        assert len(highpass_filter) == signal_length

        # Verify perfect reconstruction property: |lowpass|² + |highpass|² = 1
        reconstruction_sum = np.abs(lowpass_filter) ** 2 + np.abs(highpass_filter) ** 2
        np.testing.assert_allclose(reconstruction_sum, 1.0, atol=1e-10)

    def test_filter_initialization(self):
        """
        Test _initialize_filters() with different shapes.
        """
        # Test with uniform shape (one filter size)
        wavelet1 = MeyerWavelet(shape=(64, 64))
        assert len(wavelet1._filters) == 1
        assert 64 in wavelet1._filters

        # Test with non-uniform shape (multiple filter sizes)
        wavelet2 = MeyerWavelet(shape=(64, 128))
        assert len(wavelet2._filters) == 2
        assert 64 in wavelet2._filters
        assert 128 in wavelet2._filters

        # Test with 3D non-uniform shape
        wavelet3 = MeyerWavelet(shape=(32, 64, 128))
        assert len(wavelet3._filters) == 3
        assert 32 in wavelet3._filters
        assert 64 in wavelet3._filters
        assert 128 in wavelet3._filters
