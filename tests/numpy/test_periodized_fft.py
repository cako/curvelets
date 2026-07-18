"""Tests for periodized sparse FFT helpers and SparseWindow methods."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from curvelets.numpy import UDCT, SparseWindow
from curvelets.numpy._periodized_fft import compute_folded_indices, flip_fft_indices
from curvelets.numpy._utils import downsample, flip_fft_all_axes, upsample


def test_flip_fft_indices_matches_dense_flip():
    rng = np.random.default_rng(0)
    shape = (32, 32)
    dense = np.zeros(shape)
    idx = rng.choice(np.prod(shape), size=200, replace=False)
    dense.flat[idx] = rng.random(200)
    window = SparseWindow.from_dense(dense, threshold=1e-12)

    dense_flip = flip_fft_all_axes(window.to_dense())
    flipped_idx = flip_fft_indices(window.indices, shape)

    assert np.max(np.abs(dense_flip.flat[flipped_idx] - window.values)) < 1e-14
    flipped_window = SparseWindow.from_dense(dense_flip, threshold=1e-12)
    assert set(flipped_idx.tolist()) == set(flipped_window.indices.tolist())


def test_window_analyze_matches_ifft_downsample():
    rng = np.random.default_rng(1)
    shape = (128, 128)
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=3)
    data = rng.standard_normal(shape)
    image_freq = np.fft.fftn(data)

    window = udct.windows[1][0][0]
    dec = udct.decimation_ratios[1][0]
    assert window.folded_indices is not None
    assert window.out_shape is not None

    freq_band = np.zeros(shape, dtype=image_freq.dtype)
    window.multiply_extract(image_freq, out=freq_band, dtype=image_freq.dtype)
    coeff_ref = downsample(np.fft.ifftn(freq_band), dec)
    coeff_ref *= np.sqrt(2 * np.prod(dec))

    scale = float(np.sqrt(2 * np.prod(dec)) / np.prod(dec))
    coeff_opt = window.analyze(image_freq, scale)

    rel = np.max(np.abs(coeff_ref - coeff_opt)) / np.max(np.abs(coeff_ref))
    assert rel < 1e-12


def test_window_synthesize_matches_upsample_fft():
    rng = np.random.default_rng(2)
    shape = (128, 128)
    udct = UDCT(shape=shape, num_scales=3, wedges_per_direction=3)
    window = udct.windows[1][0][0]
    dec = udct.decimation_ratios[1][0]
    assert window.out_shape is not None

    out_shape = window.out_shape
    coeff = rng.standard_normal(out_shape) + 1j * rng.standard_normal(out_shape)

    up = upsample(coeff, dec)
    up /= np.sqrt(2 * np.prod(dec))
    bf = np.prod(dec) * np.fft.fftn(up)
    target_ref: npt.NDArray[np.complex128] = np.zeros(shape, dtype=np.complex128)
    window.scatter_add(target_ref, bf, dtype=np.complex128)

    target_opt: npt.NDArray[np.complex128] = np.zeros(shape, dtype=np.complex128)
    scale = float(np.sqrt(np.prod(dec) / 2.0))
    window.synthesize(coeff, target_opt, scale)

    rel = np.max(np.abs(target_ref - target_opt)) / np.max(np.abs(target_ref))
    assert rel < 1e-12


def test_attach_periodized_sets_maps():
    window = SparseWindow.from_dense(np.ones((8, 8)), threshold=0.5)
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
        ((64, 64), 3),
        ((128, 128), 3),
        ((32, 32, 32), 3),
    ],
)
def test_real_round_trip_periodized(shape, num_scales):
    rng = np.random.default_rng(3)
    udct = UDCT(shape=shape, num_scales=num_scales, wedges_per_direction=3)
    data = rng.standard_normal(shape)
    recon = udct.backward(udct.forward(data))
    atol = 1e-4 if len(shape) == 2 else 1e-3
    np.testing.assert_allclose(data, recon, atol=atol)


@pytest.mark.parametrize("flip", [False, True])
def test_complex_wedge_flip_equivalence(flip):
    rng = np.random.default_rng(4)
    shape = (64, 64)
    udct = UDCT(
        shape=shape, num_scales=3, wedges_per_direction=3, transform_kind="complex"
    )
    data = rng.standard_normal(shape).astype(np.complex128)
    coeffs = udct.forward(data)
    recon = udct.backward(coeffs)
    np.testing.assert_allclose(data, recon, atol=1e-4)

    window = udct.windows[1][0][0]
    assert window.flipped_indices is not None
    assert window.flipped_folded_indices is not None
    if flip:
        folded = compute_folded_indices(
            window.flipped_indices, shape, udct.decimation_ratios[1][0]
        )
        np.testing.assert_array_equal(folded, window.flipped_folded_indices)


def test_monogenic_round_trip_periodized():
    rng = np.random.default_rng(5)
    shape = (64, 64)
    udct = UDCT(
        shape=shape, num_scales=3, wedges_per_direction=3, transform_kind="monogenic"
    )
    data = rng.standard_normal(shape)
    components = udct.backward(udct.forward(data))
    np.testing.assert_allclose(data, components[0], atol=1e-4)
    assert udct._riesz_filters is not None
