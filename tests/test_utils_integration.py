"""Integration test for normal_vector_field utility function.

This test covers the currently uncovered normal_vector_field function
(lines 217-233 in src/curvelets/utils/_utils.py).
"""

from __future__ import annotations

import numpy as np

from curvelets.utils import normal_vector_field


def test_normal_vector_field(rng):
    """Test normal_vector_field with simple 2D array and default parameters."""
    # Create a simple 2D test array
    data = rng.normal(size=(32, 32))

    # Test with default parameters
    rows, cols = 3, 4
    vectors = normal_vector_field(data, rows=rows, cols=cols)

    # Verify output shape
    assert vectors.shape == (rows, cols, 2)

    # Verify vectors are normalized (magnitude should be ~1.0)
    magnitudes = np.linalg.norm(vectors, axis=2)
    np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-10)
