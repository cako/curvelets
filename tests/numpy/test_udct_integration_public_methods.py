"""Integration tests for NumPy UDCT public methods with monogenic and complex transforms."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT


class TestMonogenicVectStruct:
    """Test suite for vect() and struct() with monogenic transform coefficients."""

    def test_vect_monogenic_coefficients(self, rng):
        """
        Test vect() with monogenic transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), transform_kind="monogenic")
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> vec.ndim == 1
        True
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="monogenic",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs = transform.forward(data)

        vec = transform.vect(coeffs)

        # Verify vector is 1D
        assert vec.ndim == 1

        # Verify vector dtype is complex
        assert np.iscomplexobj(vec)

        # Verify vector length matches sum of all coefficient sizes
        # For monogenic, each wedge has ndim+1 components
        total_size = 0
        for scale_coeffs in coeffs:
            for direction_coeffs in scale_coeffs:
                for wedge_coeffs in direction_coeffs:
                    # wedge_coeffs is a list of ndim+1 arrays
                    for component in wedge_coeffs:
                        total_size += component.size

        assert vec.size == total_size

    def test_struct_monogenic_vector(self, rng):
        """
        Test struct() with monogenic transform vector.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), transform_kind="monogenic")
        >>> data = np.random.randn(64, 64)
        >>> coeffs_orig = transform.forward(data)
        >>> vec = transform.vect(coeffs_orig)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs_orig)
        True
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="monogenic",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
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

                    # For monogenic, each wedge is a list of ndim+1 components
                    assert isinstance(orig_wedge, list)
                    assert isinstance(recon_wedge, list)
                    assert len(recon_wedge) == len(orig_wedge)
                    assert len(recon_wedge) == 3  # ndim+1 = 2+1 = 3

                    # Verify each component matches
                    for comp_idx in range(len(orig_wedge)):
                        orig_comp = orig_wedge[comp_idx]
                        recon_comp = recon_wedge[comp_idx]
                        assert isinstance(recon_comp, np.ndarray)
                        assert recon_comp.shape == orig_comp.shape
                        np.testing.assert_array_equal(recon_comp, orig_comp)

    def test_vect_struct_round_trip_monogenic(self, rng):
        """
        Test round-trip vect() and struct() with monogenic transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="monogenic",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify structure and values match exactly
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

                    for comp_idx in range(len(orig_wedge)):
                        orig_comp = orig_wedge[comp_idx]
                        recon_comp = recon_wedge[comp_idx]
                        np.testing.assert_array_equal(recon_comp, orig_comp)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_vect_struct_monogenic_multidim(self, rng, dim):
        """
        Test vect() and struct() with monogenic transform for different dimensions.

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

        transform = UDCT(
            shape=shape,
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="monogenic",
        )
        data = rng.normal(size=shape).astype(np.float64)
        coeffs_orig = transform.forward(data)

        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify structure matches
        assert len(coeffs_recon) == len(coeffs_orig)
        for scale_idx in range(len(coeffs_orig)):
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]

                    # For monogenic, each wedge has ndim+1 components
                    assert len(recon_wedge) == len(orig_wedge)
                    assert len(recon_wedge) == dim + 1

                    for comp_idx in range(len(orig_wedge)):
                        orig_comp = orig_wedge[comp_idx]
                        recon_comp = recon_wedge[comp_idx]
                        np.testing.assert_array_equal(recon_comp, orig_comp)


class TestComplexEdgeCases:
    """Test suite for struct() with complex transform edge cases."""

    def test_struct_complex_wavelet_mode(self, rng):
        """
        Test struct() with complex transform in wavelet mode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(
        ...     shape=(64, 64),
        ...     num_scales=3,
        ...     high_frequency_mode="wavelet",
        ...     transform_kind="complex"
        ... )
        >>> data = np.random.randn(64, 64)
        >>> coeffs_orig = transform.forward(data)
        >>> vec = transform.vect(coeffs_orig)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs_orig)
        True
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            high_frequency_mode="wavelet",
            transform_kind="complex",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
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

                    assert isinstance(recon_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    @pytest.mark.parametrize("num_scales", [2, 3, 4, 5])
    def test_struct_complex_different_scales(self, rng, num_scales):
        """
        Test struct() with complex transform for different numbers of scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        num_scales : int
            Number of scales to test.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=num_scales,
            wedges_per_direction=3,
            transform_kind="complex",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs_orig = transform.forward(data)
        vec = transform.vect(coeffs_orig)

        coeffs_recon = transform.struct(vec)

        # Verify structure matches
        assert len(coeffs_recon) == len(coeffs_orig)
        assert len(coeffs_recon) == num_scales

        for scale_idx in range(len(coeffs_orig)):
            assert len(coeffs_recon[scale_idx]) == len(coeffs_orig[scale_idx])
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                assert len(coeffs_recon[scale_idx][direction_idx]) == len(
                    coeffs_orig[scale_idx][direction_idx]
                )
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]

                    assert isinstance(recon_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_struct_complex_wavelet_mode_different_scales(self, rng):
        """
        Test struct() with complex transform in wavelet mode with different scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=4,
            wedges_per_direction=3,
            high_frequency_mode="wavelet",
            transform_kind="complex",
        )
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs_orig = transform.forward(data)
        vec = transform.vect(coeffs_orig)

        coeffs_recon = transform.struct(vec)

        # Verify structure matches
        assert len(coeffs_recon) == len(coeffs_orig)
        assert len(coeffs_recon) == 4

        # Verify highest scale has 2*ndim directions (for complex transform)
        highest_scale_idx = 3
        assert len(coeffs_recon[highest_scale_idx]) == 2 * transform.parameters.ndim

        for scale_idx in range(len(coeffs_orig)):
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]

                    assert isinstance(recon_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)
