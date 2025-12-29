"""Tests for UDCT vect(), struct(), and _from_sparse() methods."""

from __future__ import annotations

import numpy as np

from curvelets.numpy import UDCT


class TestVectMethod:
    """Test suite for UDCT.vect() method."""

    def test_vect_basic_2d(self, rng):
        """
        Test vect() with basic 2D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs = transform.forward(data)

        vec = transform.vect(coeffs)

        # Verify vector is 1D
        assert vec.ndim == 1

        # Verify vector length matches sum of all coefficient sizes
        total_size = sum(
            wedge.size
            for scale_coeffs in coeffs
            for direction_coeffs in scale_coeffs
            for wedge in direction_coeffs
        )
        assert vec.size == total_size

        # Verify vector dtype is complex (even for real coefficients)
        assert np.iscomplexobj(vec)

    def test_vect_complex_coefficients(self, rng):
        """
        Test vect() with complex transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="complex",
        )
        data = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )
        coeffs = transform.forward(data)

        vec = transform.vect(coeffs)

        # Verify vector is complex
        assert np.iscomplexobj(vec)

        # Verify vector length
        total_size = sum(
            wedge.size
            for scale_coeffs in coeffs
            for direction_coeffs in scale_coeffs
            for wedge in direction_coeffs
        )
        assert vec.size == total_size

    def test_vect_different_shapes(self, rng):
        """
        Test vect() with different input shapes.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        shapes = [(32, 32), (64, 64), (128, 128)]
        for shape in shapes:
            transform = UDCT(shape=shape, num_scales=2, wedges_per_direction=3)
            data = rng.normal(size=shape).astype(np.float64)
            coeffs = transform.forward(data)

            vec = transform.vect(coeffs)

            # Verify vector length matches
            total_size = sum(
                wedge.size
                for scale_coeffs in coeffs
                for direction_coeffs in scale_coeffs
                for wedge in direction_coeffs
            )
            assert vec.size == total_size

    def test_vect_different_scales(self, rng):
        """
        Test vect() with different numbers of scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        for num_scales in [2, 3, 4]:
            transform = UDCT(
                shape=(64, 64), num_scales=num_scales, wedges_per_direction=3
            )
            data = rng.normal(size=(64, 64)).astype(np.float64)
            coeffs = transform.forward(data)

            vec = transform.vect(coeffs)

            # Verify vector length
            total_size = sum(
                wedge.size
                for scale_coeffs in coeffs
                for direction_coeffs in scale_coeffs
                for wedge in direction_coeffs
            )
            assert vec.size == total_size

    def test_vect_3d(self, rng):
        """
        Test vect() with 3D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(32, 32, 32), num_scales=2, wedges_per_direction=3)
        data = rng.normal(size=(32, 32, 32)).astype(np.float64)
        coeffs = transform.forward(data)

        vec = transform.vect(coeffs)

        # Verify vector properties
        assert vec.ndim == 1
        total_size = sum(
            wedge.size
            for scale_coeffs in coeffs
            for direction_coeffs in scale_coeffs
            for wedge in direction_coeffs
        )
        assert vec.size == total_size

    def test_vect_4d(self, rng):
        """
        Test vect() with 4D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(16, 16, 16, 16), num_scales=2, wedges_per_direction=3)
        data = rng.normal(size=(16, 16, 16, 16)).astype(np.float64)
        coeffs = transform.forward(data)

        vec = transform.vect(coeffs)

        # Verify vector properties
        assert vec.ndim == 1
        total_size = sum(
            wedge.size
            for scale_coeffs in coeffs
            for direction_coeffs in scale_coeffs
            for wedge in direction_coeffs
        )
        assert vec.size == total_size


class TestStructMethod:
    """Test suite for UDCT.struct() method."""

    def test_struct_basic_2d(self, rng):
        """
        Test struct() with basic 2D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
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
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    assert isinstance(recon_wedge, np.ndarray)
                    assert isinstance(orig_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape

    def test_struct_complex_coefficients(self, rng):
        """
        Test struct() with complex transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="complex",
        )
        data = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )
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

    def test_struct_different_shapes(self, rng):
        """
        Test struct() with different input shapes.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        shapes = [(32, 32), (64, 64), (128, 128)]
        for shape in shapes:
            transform = UDCT(shape=shape, num_scales=2, wedges_per_direction=3)
            data = rng.normal(size=shape).astype(np.float64)
            coeffs_orig = transform.forward(data)
            vec = transform.vect(coeffs_orig)

            coeffs_recon = transform.struct(vec)

            # Verify structure matches
            assert len(coeffs_recon) == len(coeffs_orig)


class TestVectStructRoundTrip:
    """Test suite for vect() and struct() round-trip."""

    def test_round_trip_basic_2d(self, rng):
        """
        Test round-trip: coeffs → vect() → struct() → coeffs'.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
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

                    # Verify shapes match
                    assert isinstance(recon_wedge, np.ndarray)
                    assert isinstance(orig_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape

                    # Verify values match (should be exact for round-trip)
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_complex_coefficients(self, rng):
        """
        Test round-trip with complex transform coefficients.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(64, 64),
            num_scales=3,
            wedges_per_direction=3,
            transform_kind="complex",
        )
        data = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(
            np.complex128
        )
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify structure and values match
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

                    # Verify shapes match
                    assert isinstance(recon_wedge, np.ndarray)
                    assert isinstance(orig_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape

                    # Verify values match (should be exact for round-trip)
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_different_shapes(self, rng):
        """
        Test round-trip with different input shapes.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        shapes = [(32, 32), (64, 64), (128, 128)]
        for shape in shapes:
            transform = UDCT(shape=shape, num_scales=2, wedges_per_direction=3)
            data = rng.normal(size=shape).astype(np.float64)
            coeffs_orig = transform.forward(data)

            # Round-trip conversion
            vec = transform.vect(coeffs_orig)
            coeffs_recon = transform.struct(vec)

            # Verify values match
            for scale_idx in range(len(coeffs_orig)):
                for direction_idx in range(len(coeffs_orig[scale_idx])):
                    for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                        orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                        recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]
                        np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_different_scales(self, rng):
        """
        Test round-trip with different numbers of scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        for num_scales in [2, 3, 4]:
            transform = UDCT(
                shape=(64, 64), num_scales=num_scales, wedges_per_direction=3
            )
            data = rng.normal(size=(64, 64)).astype(np.float64)
            coeffs_orig = transform.forward(data)

            # Round-trip conversion
            vec = transform.vect(coeffs_orig)
            coeffs_recon = transform.struct(vec)

            # Verify values match
            for scale_idx in range(len(coeffs_orig)):
                for direction_idx in range(len(coeffs_orig[scale_idx])):
                    for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                        orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                        recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]
                        np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_angular_wedges_config(self, rng):
        """
        Test round-trip with angular_wedges_config interface.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        angular_wedges_config = np.array([[3, 3], [6, 6], [12, 12]])
        transform = UDCT(shape=(64, 64), angular_wedges_config=angular_wedges_config)
        data = rng.normal(size=(64, 64)).astype(np.float64)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify values match
        for scale_idx in range(len(coeffs_orig)):
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_3d(self, rng):
        """
        Test round-trip with 3D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(32, 32, 32), num_scales=2, wedges_per_direction=3)
        data = rng.normal(size=(32, 32, 32)).astype(np.float64)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
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
                    assert isinstance(orig_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_3d_complex(self, rng):
        """
        Test round-trip with 3D transform in complex mode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(
            shape=(32, 32, 32),
            num_scales=2,
            wedges_per_direction=3,
            transform_kind="complex",
        )
        data = (
            rng.normal(size=(32, 32, 32)) + 1j * rng.normal(size=(32, 32, 32))
        ).astype(np.complex128)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
        vec = transform.vect(coeffs_orig)
        coeffs_recon = transform.struct(vec)

        # Verify values match
        for scale_idx in range(len(coeffs_orig)):
            for direction_idx in range(len(coeffs_orig[scale_idx])):
                for wedge_idx in range(len(coeffs_orig[scale_idx][direction_idx])):
                    orig_wedge = coeffs_orig[scale_idx][direction_idx][wedge_idx]
                    recon_wedge = coeffs_recon[scale_idx][direction_idx][wedge_idx]
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)

    def test_round_trip_4d(self, rng):
        """
        Test round-trip with 4D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(16, 16, 16, 16), num_scales=2, wedges_per_direction=3)
        data = rng.normal(size=(16, 16, 16, 16)).astype(np.float64)
        coeffs_orig = transform.forward(data)

        # Round-trip conversion
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
                    assert isinstance(orig_wedge, np.ndarray)
                    assert recon_wedge.shape == orig_wedge.shape
                    np.testing.assert_array_equal(recon_wedge, orig_wedge)
