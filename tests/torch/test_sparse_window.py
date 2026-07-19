"""Tests for PyTorch SparseWindow edge cases and utility methods."""

from __future__ import annotations

import pytest
import torch

from curvelets.torch._sparse_window import SparseWindow


def test_torch_sparse_window_multiply_extract():
    dense = torch.tensor([0.0, 0.5, 1.0])
    window = SparseWindow.from_dense(dense, threshold=0.3)
    source = torch.tensor([1.0, 2.0, 3.0])

    # Without out parameter
    res = window.multiply_extract(source)
    assert res[1].item() == 1.0
    assert res[2].item() == 3.0
    assert res[0].item() == 0.0

    # With out parameter
    out_buf = torch.ones(3)
    res2 = window.multiply_extract(source, out=out_buf)
    assert res2 is out_buf
    assert res2[1].item() == 1.0
    assert res2[2].item() == 3.0
    assert res2[0].item() == 0.0


def test_torch_sparse_window_multiply_at_indices():
    dense = torch.tensor([0.0, 0.5, 1.0])
    window = SparseWindow.from_dense(dense, threshold=0.3)
    source = torch.tensor([1.0, 2.0, 3.0])
    filter_arr = torch.tensor([0.5, 1.5, 2.5])

    # Without out parameter
    res = window.multiply_at_indices(source, filter_arr)
    assert res[1].item() == 3.0  # 2.0 * 1.5
    assert res[2].item() == 7.5  # 3.0 * 2.5
    assert res[0].item() == 0.0

    # With out parameter
    out_buf = torch.ones(3)
    res2 = window.multiply_at_indices(source, filter_arr, out=out_buf)
    assert res2 is out_buf
    assert res2[1].item() == 3.0
    assert res2[2].item() == 7.5
    assert res2[0].item() == 0.0


def test_torch_sparse_window_to_device():
    dense = torch.tensor([0.1, 0.9])
    window = SparseWindow.from_dense(dense, threshold=0.5)

    # Test moving to CPU explicitly (since CUDA might not be available)
    window_cpu = window.to(torch.device("cpu"))
    assert window_cpu.device.type == "cpu"
    assert torch.equal(window_cpu.values, window.values)


def test_torch_sparse_window_periodized_exceptions():
    dense = torch.tensor([0.1, 0.9, 0.5, 0.2]).view(2, 2)
    window = SparseWindow.from_dense(dense, threshold=0.3)

    # Calling resolve_indices or _resolved_out_shape without attaching periodized or passing decimation
    with pytest.raises(ValueError, match="decimation required when folded_indices are not attached"):
        window.resolve_indices(flip=False)

    with pytest.raises(ValueError, match="decimation required when flipped_folded_indices are not attached"):
        window.resolve_indices(flip=True)

    with pytest.raises(ValueError, match="decimation required when out_shape is not attached"):
        window._resolved_out_shape()


def test_torch_sparse_window_fold_product_and_scatter_tiled_fallback():
    dense = torch.ones((4, 4), dtype=torch.float64)
    window = SparseWindow.from_dense(dense, threshold=0.5)

    # Test fallback decimation when maps are not attached
    image_freq = torch.ones((4, 4), dtype=torch.complex128)
    folded = window.fold_product(image_freq, decimation=[2, 2])
    assert folded.shape == (2, 2)
    assert torch.all(folded != 0)

    # Test scatter_tiled fallback
    target = torch.zeros((4, 4), dtype=torch.complex128)
    small_fft = torch.ones((2, 2), dtype=torch.complex128)
    window.scatter_tiled(small_fft, target, scale=1.5, decimation=[2, 2])
    assert torch.any(target != 0)

