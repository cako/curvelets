"""NumPy vs PyTorch comparison tests for Riesz filters."""

from __future__ import annotations

import numpy as np
import pytest

import curvelets.numpy._riesz as np_riesz
import curvelets.torch._riesz as torch_riesz

from .conftest import TEST_SHAPES


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_riesz_filters_matches_numpy(ndim):
    """Test riesz_filters produces same results as NumPy."""
    shape = TEST_SHAPES[ndim]

    result_np = np_riesz.riesz_filters(shape)
    result_torch = torch_riesz.riesz_filters(shape)

    # Check same number of filters
    assert len(result_np) == len(result_torch) == ndim

    # Check each filter matches
    # Use atol=1e-6 due to floating point precision differences between NumPy and PyTorch
    for i in range(ndim):
        np.testing.assert_allclose(result_np[i], result_torch[i].numpy(), atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_riesz_filters_dc_component_zero(ndim):
    """Test that DC component is zero for all filters."""
    shape = TEST_SHAPES[ndim]

    result_torch = torch_riesz.riesz_filters(shape)

    # DC component is at index (0, 0, ...) for all dimensions
    dc_index = tuple(0 for _ in shape)
    for r_filter in result_torch:
        assert r_filter[dc_index] == 0
