"""Tests for the NumPy FFT engines and the half-spectrum fold paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from curvelets.numpy import (
    UDCT,
    SparseWindow,
    _backward_transform,
    _fft,
    _forward_transform,
    _udct,
)
from curvelets.numpy import _backward_transform_monogenic as _backward_monogenic
from curvelets.numpy._fft import available_fft_engines, resolve_fft_engine
from curvelets.numpy._periodized_fft import (
    compute_half_spectrum_maps,
    half_spectrum_shape,
    half_spectrum_source,
    hermitian_from_half,
)


def _flat(coefficients: Any) -> list[np.ndarray]:
    return [band for scale in coefficients for direction in scale for band in direction]


@pytest.mark.parametrize("engine", available_fft_engines())
@pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_engines_agree_with_numpy(engine, transform_kind, dtype):
    shape = (48, 40)
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape).astype(dtype)
    if transform_kind == "complex":
        data = data + 1j * rng.standard_normal(shape).astype(dtype)
    reference = UDCT(
        shape=shape, num_scales=3, transform_kind=transform_kind, fft_engine="numpy"
    )
    udct = UDCT(
        shape=shape, num_scales=3, transform_kind=transform_kind, fft_engine=engine
    )
    assert udct.fft_engine == engine

    ref_coeffs = reference.forward(data)
    ref_recon = reference.backward(ref_coeffs)
    coeffs = udct.forward(data)
    recon = udct.backward(coeffs)

    rtol = 1e-4 if dtype == np.float32 else 1e-10
    for got, ref in zip(_flat(coeffs), _flat(ref_coeffs), strict=True):
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
    recon_t = recon if isinstance(recon, tuple) else (recon,)
    ref_t = ref_recon if isinstance(ref_recon, tuple) else (ref_recon,)
    for got, ref in zip(recon_t, ref_t, strict=True):
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def test_unknown_engine_raises():
    with pytest.raises(ValueError, match="fft_engine must be one of"):
        UDCT(shape=(32, 32), num_scales=2, fft_engine="cufft")


def test_missing_engine_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        _fft,
        "_load",
        lambda name, _workers=1: None if name == "fftw" else _fft._numpy_backend(),
    )
    with pytest.warns(UserWarning, match="'fftw' is not available, reverting to numpy"):
        udct = UDCT(shape=(32, 32), num_scales=2, fft_engine="fftw")
    assert udct.fft_engine == "numpy"


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_workers_raise(workers):
    with pytest.raises(ValueError, match="fft_workers must be a positive integer"):
        UDCT(shape=(32, 32), num_scales=2, fft_workers=workers)


@pytest.mark.parametrize("engine", [e for e in available_fft_engines() if e != "numpy"])
def test_fft_workers_is_passed_to_engine(engine):
    data = np.random.default_rng(0).standard_normal((40, 48))
    one = UDCT(shape=(40, 48), num_scales=3, fft_engine=engine, fft_workers=1)
    two = UDCT(shape=(40, 48), num_scales=3, fft_engine=engine, fft_workers=2)
    if engine != "mkl_fft":  # mkl_fft without mkl-service has no workers
        assert one._fft.fftn.keywords["workers"] == 1
        assert two._fft.fftn.keywords["workers"] == 2
    np.testing.assert_allclose(
        two.backward(two.forward(data)), one.backward(one.forward(data)), atol=1e-12
    )


def test_engine_is_resolved_once_at_construction(monkeypatch):
    udct = UDCT(shape=(32, 32), num_scales=2, fft_engine="numpy")

    def fail(*_args: Any) -> None:
        msg = "engine resolved after construction"
        raise AssertionError(msg)

    for module in (_forward_transform, _backward_transform, _backward_monogenic, _udct):
        monkeypatch.setattr(module, "resolve_fft_engine", fail)
    data = np.random.default_rng(0).standard_normal((32, 32))
    np.testing.assert_allclose(udct.backward(udct.forward(data)), data, atol=1e-10)


def test_auto_prefers_mkl_fft_then_benchmarks(monkeypatch):
    fake_mkl = _fft.FFTBackend(
        "mkl_fft", np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn
    )
    _fft._auto_for_shape.cache_clear()
    monkeypatch.setattr(
        _fft, "_load", lambda name, _workers=1: fake_mkl if name == "mkl_fft" else None
    )
    assert resolve_fft_engine("auto", (16, 16)) is fake_mkl

    _fft._auto_for_shape.cache_clear()
    monkeypatch.setattr(_fft, "_load", lambda _name, _workers=1: None)
    assert resolve_fft_engine("auto", (16, 16)).name == "numpy"
    _fft._auto_for_shape.cache_clear()


def test_proxy_shape_respects_budget():
    assert _fft._proxy_shape((64, 64)) == (32, 32)
    proxy = _fft._proxy_shape((1024, 1024, 1024))
    assert len(proxy) == 3
    assert np.prod(proxy) <= _fft._BENCHMARK_MAX_ELEMENTS


@pytest.mark.parametrize("shape", [(8, 8), (8, 7), (6, 5, 4)])
def test_half_spectrum_gather_matches_full_spectrum(shape):
    rng = np.random.default_rng(1)
    data = rng.standard_normal(shape)
    indices = rng.choice(np.prod(shape), size=np.prod(shape) // 2, replace=False)
    maps = compute_half_spectrum_maps(
        indices.astype(np.intp), np.zeros_like(indices), np.ones(indices.size), shape
    )
    source = half_spectrum_source(np.fft.rfftn(data))
    np.testing.assert_allclose(
        source[maps.source], np.fft.fftn(data).reshape(-1)[indices], atol=1e-12
    )


@pytest.mark.parametrize("shape", [(8, 8), (8, 7), (6, 5, 4)])
def test_half_spectrum_scatter_matches_real_part(shape):
    rng = np.random.default_rng(2)
    dense = np.zeros(shape)
    support = rng.choice(np.prod(shape), size=np.prod(shape) // 2, replace=False)
    dense.flat[support] = rng.random(support.size) + 0.1
    window = SparseWindow.from_dense(dense, threshold=0.05)
    window.attach_periodized(
        [2] * len(shape) if shape[-1] % 2 == 0 else [1] * len(shape)
    )
    small = rng.standard_normal(window.out_shape) + 1j * rng.standard_normal(
        window.out_shape
    )

    full = np.zeros(shape, dtype=np.complex128)
    window.scatter_tiled(small, full, 1.5)

    own: npt.NDArray[np.complex128] = np.zeros(
        half_spectrum_shape(shape), dtype=np.complex128
    )
    mirror = np.zeros_like(own)
    window.scatter_half(small, own, mirror, 1.5)
    recon = np.fft.irfftn(
        hermitian_from_half(own, mirror), s=shape, axes=tuple(range(len(shape)))
    )

    np.testing.assert_allclose(recon, np.fft.ifftn(full).real, atol=1e-12)


def test_fold_with_collisions_accumulates():
    # Odd length folded by 2 has colliding bins; the fold must sum them.
    window = SparseWindow.from_dense(np.ones(5), threshold=0.5)
    window.attach_periodized([2])
    samples: npt.NDArray[np.complex128] = np.arange(1, 6, dtype=np.complex128)
    folded = window.fold(samples)
    expected = np.zeros(window.out_shape, dtype=np.complex128)
    np.add.at(expected, window.folded_indices, samples)
    np.testing.assert_array_equal(folded, expected)


def test_scatter_tiled_non_contiguous_target():
    window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
    window.attach_periodized([2, 2])
    small: npt.NDArray[np.complex128] = np.arange(4, dtype=np.complex128).reshape(2, 2)
    contiguous: npt.NDArray[np.complex128] = np.zeros((4, 4), dtype=np.complex128)
    window.scatter_tiled(small, contiguous, 2.0)
    fortran: npt.NDArray[np.complex128] = np.zeros(
        (4, 4), dtype=np.complex128, order="F"
    )
    window.scatter_tiled(small, fortran, 2.0)
    np.testing.assert_array_equal(fortran, contiguous)
