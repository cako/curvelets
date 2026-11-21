"""Tests for the NumPy FFT engines and the half-spectrum fold paths."""

from __future__ import annotations

import ctypes
import functools
import os
import sys
import threading
import types
from collections.abc import Callable
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
        UDCT(shape=(32, 32), num_scales=2, fft_engine="cufft")  # type: ignore[arg-type]


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
        for udct, workers in ((one, 1), (two, 2)):
            fftn = udct._fft.fftn
            if engine == "fftw":  # bound method of the engine's plan cache
                assert isinstance(fftn.__self__, _fft._FFTWPlans)  # type: ignore[attr-defined]
                assert fftn.__self__._workers == workers  # type: ignore[attr-defined]
            else:
                assert isinstance(fftn, functools.partial)
                assert fftn.keywords["workers"] == workers
    np.testing.assert_allclose(
        two.backward(two.forward(data)), one.backward(one.forward(data)), atol=1e-12
    )


def test_default_workers_respect_cpu_affinity(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 32, raising=False)
    assert _fft._resolve_workers(None) == 32
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False
    )
    assert _fft._resolve_workers(None) == 8


def test_engine_is_resolved_once_at_construction(monkeypatch):
    udct = UDCT(shape=(32, 32), num_scales=2, fft_engine="numpy")

    def fail(*_args: Any) -> None:
        msg = "engine resolved after construction"
        raise AssertionError(msg)

    for module in (_forward_transform, _backward_transform, _backward_monogenic, _udct):
        monkeypatch.setattr(module, "resolve_fft_engine", fail)
    data = np.random.default_rng(0).standard_normal((32, 32))
    np.testing.assert_allclose(udct.backward(udct.forward(data)), data, atol=1e-10)


@pytest.mark.parametrize("fastest", ["numpy", "scipy", "mkl_fft"])
def test_auto_picks_fastest_installed_engine(monkeypatch, fastest):
    fakes = {
        name: _fft.FFTBackend(
            name, np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn
        )
        for name in ("scipy", "mkl_fft")
    }
    # The stock FFTW plans slowly (FFTW_MEASURE): not an auto candidate.
    fakes["fftw"] = _fft.FFTBackend(
        "fftw",
        np.fft.fftn,
        np.fft.ifftn,
        np.fft.rfftn,
        np.fft.irfftn,
        cheap_planning=False,
    )
    monkeypatch.setattr(_fft, "_load", lambda name, _workers=1: fakes.get(name))
    timed: list[str] = []

    def fake_time(backend, _data):
        timed.append(backend.name)
        return 0.0 if backend.name == fastest else 1.0

    monkeypatch.setattr(_fft, "_time_backend", fake_time)
    _fft._auto_for_shape.cache_clear()
    try:
        assert resolve_fft_engine("auto", (16, 16)).name == fastest
    finally:
        _fft._auto_for_shape.cache_clear()
    # fftw is never a candidate: its first call per shape pays for planning.
    assert sorted(timed) == ["mkl_fft", "numpy", "scipy"]


def test_auto_without_optional_engines_is_numpy(monkeypatch):
    monkeypatch.setattr(_fft, "_load", lambda _name, _workers=1: None)

    def fail(*_args):
        msg = "nothing to benchmark"
        raise AssertionError(msg)

    monkeypatch.setattr(_fft, "_time_backend", fail)
    _fft._auto_for_shape.cache_clear()
    try:
        assert resolve_fft_engine("auto", (16, 16)).name == "numpy"
    finally:
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
    expected: npt.NDArray[np.complex128] = np.zeros(
        window.out_shape, dtype=np.complex128
    )
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


def _fake_scipy_fft_module(
    name: str,
) -> tuple[types.ModuleType, list[tuple[str, int | None]]]:
    """A stand-in SciPy-style FFT interface, and the log of its calls' ``workers``."""
    module = types.ModuleType(name)
    calls: list[tuple[str, int | None]] = []
    for fn_name in ("fftn", "ifftn", "rfftn", "irfftn"):
        numpy_fn = getattr(np.fft, fn_name)

        def fn(
            x: Any,
            *args: Any,
            _name: str = fn_name,
            _np: Any = numpy_fn,
            workers: int | None = None,
            **kwargs: Any,
        ) -> Any:
            calls.append((_name, workers))
            return _np(x, *args, **kwargs)

        setattr(module, fn_name, fn)
    return module, calls


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch, modules: dict[str, types.ModuleType]
) -> None:
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    _fft._load.cache_clear()


@pytest.fixture
def clear_engine_cache():
    _fft._load.cache_clear()
    yield
    _fft._load.cache_clear()


def _fake_pyfftw() -> tuple[types.ModuleType, list[dict[str, Any]]]:
    """
    A stand-in pyFFTW: ``builders.<func>`` returns a plan that, like a real one,
    writes into its own output buffer and returns that buffer on every call.
    """
    plans: list[dict[str, Any]] = []

    def builder(func: str) -> Callable[..., Callable[[np.ndarray], np.ndarray]]:
        def build(
            scratch: np.ndarray,
            *,
            s: Any,
            axes: Any,
            threads: int,
            planner_effort: str,
            avoid_copy: bool,
        ) -> Callable[[np.ndarray], np.ndarray]:
            info = {
                "func": func,
                "shape": scratch.shape,
                "threads": threads,
                "effort": planner_effort,
                "avoid_copy": avoid_copy,
                "thread": threading.get_ident(),
                "runs": 0,
            }
            plans.append(info)
            buffer: dict[str, np.ndarray] = {}

            def run(x: np.ndarray) -> np.ndarray:
                info["runs"] += 1
                result = getattr(np.fft, func)(x, s=s, axes=axes)
                out = buffer.setdefault("out", np.empty_like(result))
                out[...] = result
                return out

            return run

        return build

    builders = types.ModuleType("pyfftw.builders")
    for func in ("fftn", "ifftn", "rfftn", "irfftn"):
        setattr(builders, func, builder(func))
    pyfftw = types.ModuleType("pyfftw")
    setattr(pyfftw, "builders", builders)  # noqa: B010
    setattr(pyfftw, "empty_aligned", np.empty)  # noqa: B010
    return pyfftw, plans


@pytest.mark.usefixtures("clear_engine_cache")
def test_fftw_engine_plans_once_and_copies_out(monkeypatch):
    pyfftw, plans = _fake_pyfftw()
    _install_fake_modules(
        monkeypatch, {"pyfftw": pyfftw, "pyfftw.builders": pyfftw.builders}
    )
    udct = UDCT(shape=(32, 32), num_scales=3, fft_engine="fftw", fft_workers=3)
    assert udct.fft_engine == "fftw"
    data = np.random.default_rng(0).standard_normal((32, 32))

    coeffs = udct.forward(data)
    first = [c.copy() for s in coeffs for d in s for c in d]
    np.testing.assert_allclose(udct.backward(coeffs), data, atol=1e-10)
    planned = len(plans)
    assert planned > 0
    assert {p["threads"] for p in plans} == {3}
    assert {p["effort"] for p in plans} == {"FFTW_MEASURE"}
    assert not any(p["avoid_copy"] for p in plans)

    # Later transforms reuse the plans; outputs are copies, not plan buffers.
    coeffs2 = udct.forward(data)
    np.testing.assert_allclose(udct.backward(coeffs2), data, atol=1e-10)
    assert len(plans) == planned
    again = [c for s in coeffs2 for d in s for c in d]
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    flat = [c for s in coeffs for d in s for c in d]
    assert len({id(c) for c in flat}) == len(flat)

    # Another thread gets its own plans (FFTW plans are not thread-safe).
    worker = threading.Thread(target=lambda: udct.backward(udct.forward(data)))
    worker.start()
    worker.join()
    assert len(plans) == 2 * planned
    assert len({p["thread"] for p in plans}) == 2


@pytest.mark.usefixtures("clear_engine_cache")
@pytest.mark.parametrize("with_mkl_service", [True, False])
def test_mkl_fft_engine_wiring_with_fake_mkl_fft(monkeypatch, with_mkl_service):
    scipy_fft, scipy_calls = _fake_scipy_fft_module("mkl_fft.interfaces.scipy_fft")
    numpy_fft, numpy_calls = _fake_scipy_fft_module("mkl_fft.interfaces.numpy_fft")
    interfaces = types.ModuleType("mkl_fft.interfaces")
    setattr(interfaces, "numpy_fft", numpy_fft)  # noqa: B010
    package = types.ModuleType("mkl_fft")
    setattr(package, "interfaces", interfaces)  # noqa: B010
    modules = {
        "mkl_fft": package,
        "mkl_fft.interfaces": interfaces,
        "mkl_fft.interfaces.numpy_fft": numpy_fft,
    }
    if with_mkl_service:
        setattr(interfaces, "scipy_fft", scipy_fft)  # noqa: B010
        modules["mkl_fft.interfaces.scipy_fft"] = scipy_fft
    else:
        # The SciPy interface imports mkl-service; simulate it being missing.
        def missing(name: str) -> types.ModuleType:
            if name == "scipy_fft":
                msg = "No module named 'mkl'"
                raise ImportError(msg)
            raise AttributeError(name)

        setattr(interfaces, "__getattr__", missing)  # noqa: B010
    _install_fake_modules(monkeypatch, modules)

    udct = UDCT(shape=(32, 32), num_scales=2, fft_engine="mkl_fft", fft_workers=2)
    assert udct.fft_engine == "mkl_fft"
    data = np.random.default_rng(0).standard_normal((32, 32))
    np.testing.assert_allclose(udct.backward(udct.forward(data)), data, atol=1e-10)
    used = scipy_calls if with_mkl_service else numpy_calls
    unused = numpy_calls if with_mkl_service else scipy_calls
    assert used
    assert not unused
    assert {w for _, w in used} == ({2} if with_mkl_service else {None})


def test_default_workers_without_affinity_uses_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert _fft._resolve_workers(None) == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _fft._resolve_workers(None) == 1


@pytest.mark.parametrize("transform_kind", ["real", "complex", "monogenic"])
def test_transform_functions_use_the_default_engine(transform_kind):
    """The private transform functions use the default engine when none is given."""
    from curvelets.numpy._backward_transform import _apply_backward_transform
    from curvelets.numpy._backward_transform_monogenic import (
        _apply_backward_transform_monogenic,
    )
    from curvelets.numpy._forward_transform import (
        _apply_forward_transform_complex,
        _apply_forward_transform_monogenic,
        _apply_forward_transform_real,
    )

    shape = (32, 32)
    udct = UDCT(shape=shape, num_scales=2, transform_kind=transform_kind)
    args = (udct.parameters, udct.windows, udct.decimation_ratios)
    data = np.random.default_rng(0).standard_normal(shape)
    if transform_kind == "real":
        coeffs = _apply_forward_transform_real(data, *args)
        recon = _apply_backward_transform(coeffs, *args)
        np.testing.assert_allclose(recon, data, atol=1e-10)
    elif transform_kind == "complex":
        cdata = data + 1j * data
        coeffs = _apply_forward_transform_complex(cdata, *args)
        recon = _apply_backward_transform(coeffs, *args, use_complex_transform=True)
        np.testing.assert_allclose(recon, cdata, atol=1e-10)
    else:
        coeffs = _apply_forward_transform_monogenic(data, *args)
        components = _apply_backward_transform_monogenic(coeffs, *args)
        np.testing.assert_allclose(components[0], data, atol=1e-10)


def test_gather_rejects_flip_with_half_spectrum():
    window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
    window.attach_periodized([2, 2])
    source = half_spectrum_source(np.fft.rfftn(np.ones((4, 4))))
    with pytest.raises(ValueError, match="flip is not supported"):
        window.gather(source, flip=True, half=True)


def test_fold_product_applies_extra_weights():
    window = SparseWindow.from_dense(np.ones((4, 4)), threshold=0.5)
    window.attach_periodized([2, 2])
    spectrum: npt.NDArray[np.complex128] = np.ones((4, 4), dtype=np.complex128)
    extra = np.full(window.size, 3.0)
    np.testing.assert_allclose(
        window.fold_product(spectrum, extra_at_indices=extra),
        3.0 * window.fold_product(spectrum),
    )


def test_auto_times_fftw_when_it_plans_cheaply(monkeypatch):
    fftw = _fft.FFTBackend(
        "fftw",
        np.fft.fftn,
        np.fft.ifftn,
        np.fft.rfftn,
        np.fft.irfftn,
        cheap_planning=True,
    )
    monkeypatch.setattr(
        _fft, "_load", lambda name, _workers=1: fftw if name == "fftw" else None
    )
    monkeypatch.setattr(
        _fft, "_time_backend", lambda backend, _data: 0.0 if backend is fftw else 1.0
    )
    _fft._auto_for_shape.cache_clear()
    try:
        assert resolve_fft_engine("auto", (16, 16)) is fftw
    finally:
        _fft._auto_for_shape.cache_clear()


@pytest.mark.usefixtures("clear_engine_cache")
@pytest.mark.parametrize("amd", [False, True])
def test_fftw_planner_effort_depends_on_fftw_build(monkeypatch, amd):
    pyfftw, plans = _fake_pyfftw()
    _install_fake_modules(
        monkeypatch, {"pyfftw": pyfftw, "pyfftw.builders": pyfftw.builders}
    )
    monkeypatch.setattr(_fft, "_fftw_is_amd", lambda: amd)
    udct = UDCT(shape=(32, 32), num_scales=2, fft_engine="fftw", fft_workers=2)
    assert udct._fft.cheap_planning is amd
    data = np.random.default_rng(0).standard_normal((32, 32))
    np.testing.assert_allclose(udct.backward(udct.forward(data)), data, atol=1e-10)
    expected = "FFTW_ESTIMATE" if amd else "FFTW_MEASURE"
    assert {p["effort"] for p in plans} == {expected}


def test_fftw_is_amd_checks_loaded_fftw_libraries(monkeypatch, tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/libm.so.6\n"
        "7f02-7f03 r-xp 00000000 08:01 2 /opt/x/pyfftw.libs/libfftw3-ab12.so.3.6.10\n"
        "7f04-7f05 r-xp 00000000 08:01 3 /opt/x/libfftw3_threads.so.3\n"
        "7f06-7f07 rw-p 00000000 00:00 0\n",
        encoding="utf-8",
    )
    opened: list[str] = []

    class Library:
        def __init__(self, path: str, symbols: tuple[str, ...]) -> None:
            opened.append(path)
            for symbol in symbols:
                setattr(self, symbol, object())

    monkeypatch.setattr(ctypes, "CDLL", lambda path: Library(path, ()))
    assert not _fft._fftw_is_amd(str(maps))
    # Only the FFTW libraries themselves are opened (not libm, not _threads).
    assert opened == ["/opt/x/pyfftw.libs/libfftw3-ab12.so.3.6.10"]

    monkeypatch.setattr(
        ctypes, "CDLL", lambda path: Library(path, ("fftw_aoclversion",))
    )
    assert _fft._fftw_is_amd(str(maps))

    def unloadable(path: str) -> None:
        raise OSError(path)

    monkeypatch.setattr(ctypes, "CDLL", unloadable)
    assert not _fft._fftw_is_amd(str(maps))
    assert not _fft._fftw_is_amd(str(tmp_path / "missing"))


def test_udct_default_engine_without_timing(monkeypatch):
    def fail(*_args: Any) -> None:
        msg = "the default must not time engines"
        raise AssertionError(msg)

    monkeypatch.setattr(_fft, "_time_backend", fail)
    # SciPy is optional (e.g. absent on PyPy); without it the default is NumPy.
    expected = "scipy" if "scipy" in available_fft_engines() else "numpy"
    assert UDCT(shape=(32, 32), num_scales=2).fft_engine == expected


def test_default_engine_falls_back_to_numpy_silently(monkeypatch, recwarn):
    monkeypatch.setattr(_fft, "_load", lambda _name, _workers=1: None)
    udct = UDCT(shape=(32, 32), num_scales=2)
    assert udct.fft_engine == "numpy"
    assert not [w for w in recwarn if "FFT engine" in str(w.message)]
