"""Tests for SparseWindow edge cases and utility methods."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from curvelets.numpy import SparseWindow


def test_sparse_window_missing_decimation_exceptions():
    # Create window without attaching decimation
    dense = np.ones((4, 4))
    window = SparseWindow.from_dense(dense, threshold=0.5)

    with pytest.raises(
        ValueError, match="decimation required when folded_indices are not attached"
    ):
        window.resolve_indices()

    with pytest.raises(
        ValueError,
        match="decimation required when flipped_folded_indices are not attached",
    ):
        window.resolve_indices(flip=True)

    with pytest.raises(
        ValueError, match="decimation required when out_shape is not attached"
    ):
        window._resolved_out_shape()

    # If decimation is passed directly, it shouldn't raise
    idx, _ = window.resolve_indices(decimation=[2, 2])
    assert len(idx) == 16
    assert window._resolved_out_shape(decimation=[2, 2]) == (2, 2)


def test_sparse_window_multiply_extract():
    dense = np.array([0.0, 0.5, 1.0])
    window = SparseWindow.from_dense(dense, threshold=0.3)
    source = np.array([1.0, 2.0, 3.0])

    # Without out parameter
    res = window.multiply_extract(source)
    assert res[1] == 1.0
    assert res[2] == 3.0
    assert res[0] == 0.0

    # With out parameter
    out_buf = np.ones(3)
    res2 = window.multiply_extract(source, out=out_buf)
    assert res2 is out_buf
    assert res2[1] == 1.0
    assert res2[2] == 3.0
    assert res2[0] == 0.0


def test_sparse_window_multiply_at_indices():
    dense = np.array([0.0, 0.5, 1.0])
    window = SparseWindow.from_dense(dense, threshold=0.3)
    source = np.array([1.0, 2.0, 3.0])
    filter_arr = np.array([0.5, 1.5, 2.5])

    # Without out parameter
    res = window.multiply_at_indices(source, filter_arr)
    assert (
        res[1] == 3.0
    )  # 2.0 * 1.5 (source * filter, window value doesn't multiply here, it's just indices)
    assert res[2] == 7.5  # 3.0 * 2.5
    assert res[0] == 0.0

    # With out parameter
    out_buf = np.ones(3)
    res2 = window.multiply_at_indices(source, filter_arr, out=out_buf)
    assert res2 is out_buf
    assert res2[1] == 3.0
    assert res2[2] == 7.5
    assert res2[0] == 0.0


def test_sparse_window_fold_product_out():
    dense = np.ones((4, 4))
    window = SparseWindow.from_dense(dense, threshold=0.5)
    window.attach_periodized([2, 2])
    image_freq: npt.NDArray[np.complex128] = np.ones((4, 4), dtype=np.complex128)

    out_buf: npt.NDArray[np.complex128] = np.ones((2, 2), dtype=np.complex128)
    res = window.fold_product(image_freq, out=out_buf)

    assert res is out_buf
    assert res.shape == (2, 2)
    # The output buffer should be cleared and then folded values accumulated
    assert np.all(res == 4.0)
