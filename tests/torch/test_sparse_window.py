"""Tests for PyTorch SparseWindow edge cases and utility methods."""

from __future__ import annotations

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
