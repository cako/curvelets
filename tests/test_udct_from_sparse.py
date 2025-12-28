"""Tests for UDCT.from_sparse() method."""

from __future__ import annotations

import numpy as np
import pytest

from curvelets.numpy import UDCT


class TestFromSparseMethod:
    """Test suite for UDCT.from_sparse() method."""

    def test_from_sparse(self, rng):
        """
        Test conversion from sparse to dense window representation.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        >>> sparse_window = transform.windows[0][0][0]
        >>> dense_window = transform.from_sparse(sparse_window)
        >>> dense_window.shape
        (64, 64)
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)

        # Get a sparse window from the transform
        sparse_window = transform.windows[0][0][0]

        # Convert to dense
        dense_window = transform.from_sparse(sparse_window)

        # Verify shape matches transform parameters
        assert dense_window.shape == transform.parameters.shape

        # Verify dtype matches sparse window values dtype
        _, val = sparse_window
        assert dense_window.dtype == val.dtype

        # Verify that non-zero values match
        idx, val = sparse_window
        np.testing.assert_array_equal(dense_window.flat[idx], val)

        # Verify that zero positions are actually zero
        mask = np.ones(transform.parameters.shape, dtype=bool)
        mask.flat[idx] = False
        assert np.all(dense_window[mask] == 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_from_sparse_different_dtypes(self, rng, dtype):
        """
        Test from_sparse() with different dtypes.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        dtype : numpy.dtype
            Data type to test.
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)

        # Get a sparse window
        sparse_window = transform.windows[0][0][0]

        # Convert to dense
        dense_window = transform.from_sparse(sparse_window)

        # Verify dtype (should match the sparse window values dtype)
        _, val = sparse_window
        assert dense_window.dtype == val.dtype

    def test_from_sparse_multiple_scales(self, rng):
        """
        Test from_sparse() with windows from different scales.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)

        # Test windows from different scales
        for scale_idx in range(transform.parameters.num_scales):
            for direction_idx in range(len(transform.windows[scale_idx])):
                for wedge_idx in range(len(transform.windows[scale_idx][direction_idx])):
                    sparse_window = transform.windows[scale_idx][direction_idx][wedge_idx]

                    # Convert to dense
                    dense_window = transform.from_sparse(sparse_window)

                    # Verify shape
                    assert dense_window.shape == transform.parameters.shape

                    # Verify values match
                    idx, val = sparse_window
                    np.testing.assert_array_equal(dense_window.flat[idx], val)

    def test_from_sparse_3d(self, rng):
        """
        Test from_sparse() with 3D transform.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator fixture.
        """
        transform = UDCT(shape=(32, 32, 32), num_scales=2, wedges_per_direction=3)

        # Get a sparse window
        sparse_window = transform.windows[0][0][0]

        # Convert to dense
        dense_window = transform.from_sparse(sparse_window)

        # Verify shape
        assert dense_window.shape == transform.parameters.shape
        assert dense_window.shape == (32, 32, 32)

        # Verify values match
        idx, val = sparse_window
        np.testing.assert_array_equal(dense_window.flat[idx], val)
