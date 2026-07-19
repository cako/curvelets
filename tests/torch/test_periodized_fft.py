"""Tests for PyTorch periodized sparse FFT helpers and SparseWindow methods."""

from __future__ import annotations

import math

import pytest
import torch

from curvelets.torch import UDCT, SparseWindow
from curvelets.torch._periodized_fft import (
    compute_folded_indices,
    decimated_shape,
    flip_fft_indices,
)
from curvelets.torch._utils import downsample, flip_fft_all_axes, upsample


def test_torch_flip_fft_indices_matches_dense_flip():
    shape = (32, 32)
    dense = torch.zeros(shape, dtype=torch.float64)
    idx = torch.randperm(math.prod(shape))[:200]
    dense.view(-1)[idx] = torch.rand(200, dtype=torch.float64)
    window = SparseWindow.from_dense(dense, threshold=1e-12)

    dense_flip = flip_fft_all_axes(window.to_dense())
    flipped_idx = flip_fft_indices(window.indices, shape)

    assert torch.max(torch.abs(dense_flip.view(-1)[flipped_idx] - window.values)) < 1e-14
    flipped_window = SparseWindow.from_dense(dense_flip, threshold=1e-12)
    assert set(flipped_idx.tolist()) == set(flipped_window.indices.tolist())


def test_torch_window_analyze_matches_ifft_downsample():
    shape = (64, 64)
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=3)
    data = torch.randn(shape, dtype=torch.float64)
    image_freq = torch.fft.fftn(data)

    window = udct._windows[1][0][0]
    dec = udct._decimation_ratios[1][0]
    assert window.folded_indices is not None
    assert window.out_shape is not None

    freq_band = torch.zeros(shape, dtype=image_freq.dtype, device=image_freq.device)
    window.multiply_extract(image_freq, out=freq_band)
    coeff_ref = downsample(torch.fft.ifftn(freq_band), dec)
    prod_d = float(torch.prod(dec.float()).item())
    coeff_ref *= math.sqrt(2.0 * prod_d)

    scale = float(math.sqrt(2.0 * prod_d) / prod_d)
    coeff_opt = window.analyze(image_freq, scale)

    rel = torch.max(torch.abs(coeff_ref - coeff_opt)) / torch.max(torch.abs(coeff_ref))
    assert rel < 1e-12


def test_torch_window_synthesize_matches_upsample_fft():
    shape = (64, 64)
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=3)
    window = udct._windows[1][0][0]
    dec = udct._decimation_ratios[1][0]
    assert window.out_shape is not None

    out_shape = window.out_shape
    coeff = torch.randn(out_shape, dtype=torch.float64) + 1j * torch.randn(
        out_shape, dtype=torch.float64
    )

    up = upsample(coeff, dec)
    prod_d = float(torch.prod(dec.float()).item())
    up /= math.sqrt(2.0 * prod_d)
    bf = prod_d * torch.fft.fftn(up)
    target_ref = torch.zeros(shape, dtype=torch.complex128, device=coeff.device)
    window.scatter_add(target_ref, bf)

    target_opt = torch.zeros(shape, dtype=torch.complex128, device=coeff.device)
    scale = float(math.sqrt(prod_d / 2.0))
    window.synthesize(coeff, target_opt, scale)

    rel = torch.max(torch.abs(target_ref - target_opt)) / torch.max(torch.abs(target_ref))
    assert rel < 1e-12


def test_torch_attach_periodized_sets_maps():
    window = SparseWindow.from_dense(torch.ones((8, 8)), threshold=0.5)
    window.attach_periodized([2, 2], with_flip=True)
    assert window.decimation is not None
    assert window.out_shape == (4, 4)
    assert window.folded_indices is not None
    assert window.flipped_indices is not None
    assert window.flipped_folded_indices is not None
    idx, folded = window.resolve_indices()
    assert len(idx) == len(folded) == window.size


@pytest.mark.parametrize(
    ("shape", "num_scales"),
    [
        ((32, 32), 3),
        ((64, 64), 3),
        ((16, 16, 16), 3),
    ],
)
def test_torch_real_round_trip_periodized(shape, num_scales):
    udct = UDCT(shape=shape, num_scales=num_scales, wedges_per_direction=3)
    data = torch.randn(shape, dtype=torch.float64)
    recon = udct.backward(udct.forward(data))
    atol = 1e-4 if len(shape) == 2 else 1e-3
    torch.testing.assert_close(data, recon, atol=atol, rtol=1e-4)


@pytest.mark.parametrize("flip", [False, True])
def test_torch_complex_wedge_flip_equivalence(flip):
    shape = (32, 32)
    udct = UDCT(
        shape=shape, num_scales=3, wedges_per_direction=3, transform_kind="complex"
    )
    data = (torch.randn(shape, dtype=torch.float64) + 1j * torch.randn(shape, dtype=torch.float64))
    coeffs = udct.forward(data)
    recon = udct.backward(coeffs)
    torch.testing.assert_close(data, recon, atol=1e-4, rtol=1e-4)

    window = udct._windows[1][0][0]
    assert window.flipped_indices is not None
    assert window.flipped_folded_indices is not None
    if flip:
        folded = compute_folded_indices(
            window.flipped_indices, shape, udct._decimation_ratios[1][0]
        )
        assert torch.equal(folded, window.flipped_folded_indices)


def test_torch_decimated_shape_exception():
    with pytest.raises(ValueError, match="shape and decimation must have equal length"):
        decimated_shape((64, 64), [2])
