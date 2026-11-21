"""
FFT engines for the NumPy UDCT.

The transform is FFT-bound, and ``numpy.fft`` runs on a single thread. As in
PyLops, the FFT implementation is chosen per transform with an ``engine``:

- ``"numpy"``: :mod:`numpy.fft` (always available);
- ``"scipy"``: :mod:`scipy.fft`, multithreaded with ``workers``;
- ``"fftw"``: FFTW through ``pyFFTW`` (:mod:`pyfftw.interfaces.scipy_fft`,
  with its plan cache enabled), multithreaded;
- ``"mkl_fft"``: Intel MKL through ``mkl_fft``, multithreaded
  (:mod:`mkl_fft.interfaces.scipy_fft`, which needs ``mkl-service``; without
  it, :mod:`mkl_fft.interfaces.numpy_fft` with MKL's default thread count);
- ``"auto"`` (default): ``numpy`` and whichever of ``scipy`` and ``mkl_fft``
  are importable are timed once per transform shape on a wedge-sized proxy
  problem, and the fastest is kept. ``fftw`` is never chosen automatically: its
  first call for each shape pays for planning.

``workers`` sets the thread count of the multithreaded engines (default: all
CPUs). A requested engine whose package is not installed falls back to
``numpy`` with a warning. Every engine returns the same values up to floating-point
rounding.
"""

from __future__ import annotations

import functools
import math
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

FFTEngine = Literal["auto", "numpy", "scipy", "fftw", "mkl_fft"]

FFT_ENGINES: tuple[str, ...] = ("auto", "numpy", "scipy", "fftw", "mkl_fft")

# Proxy problems larger than this many elements are shrunk before timing, so the
# one-off benchmark stays in the milliseconds even for very large transforms.
_BENCHMARK_MAX_ELEMENTS = 2**18
_BENCHMARK_REPEATS = 3

_INSTALL_HINTS = {
    "scipy": 'pip install "curvelets[scipy]"',
    "fftw": 'pip install "curvelets[fftw]"',
    "mkl_fft": 'pip install "curvelets[mkl]" (Linux x86_64 / Windows x64)',
}


@dataclass(frozen=True)
class FFTBackend:
    """N-dimensional FFT functions with the ``numpy.fft`` signatures."""

    name: str
    fftn: Callable[..., Any]
    ifftn: Callable[..., Any]
    rfftn: Callable[..., Any]
    irfftn: Callable[..., Any]


def _from_scipy_like(name: str, module: Any, **kwargs: Any) -> FFTBackend:
    return FFTBackend(
        name,
        functools.partial(module.fftn, **kwargs),
        functools.partial(module.ifftn, **kwargs),
        functools.partial(module.rfftn, **kwargs),
        functools.partial(module.irfftn, **kwargs),
    )


def _numpy_backend(_workers: int = 1) -> FFTBackend:
    return FFTBackend("numpy", np.fft.fftn, np.fft.ifftn, np.fft.rfftn, np.fft.irfftn)


def _scipy_backend(workers: int) -> FFTBackend:
    import scipy.fft  # pylint: disable=import-outside-toplevel

    return _from_scipy_like("scipy", scipy.fft, workers=workers)


def _fftw_backend(workers: int) -> FFTBackend:
    # pylint: disable-next=import-outside-toplevel,import-error
    import pyfftw.interfaces.cache

    # pylint: disable-next=import-outside-toplevel,import-error
    from pyfftw.interfaces import scipy_fft

    # Keep FFTW plans alive between calls (the UDCT reuses the same shapes).
    pyfftw.interfaces.cache.enable()
    return _from_scipy_like("fftw", scipy_fft, workers=workers)


def _mkl_fft_backend(workers: int) -> FFTBackend:
    try:
        # pylint: disable-next=import-outside-toplevel,import-error
        from mkl_fft.interfaces import scipy_fft
    except ImportError:
        # The SciPy interface needs mkl-service; the NumPy one does not, but it
        # has no ``workers`` and runs with MKL's default thread count.
        # pylint: disable-next=import-outside-toplevel,import-error
        from mkl_fft.interfaces import numpy_fft

        return _from_scipy_like("mkl_fft", numpy_fft)
    return _from_scipy_like("mkl_fft", scipy_fft, workers=workers)


_FACTORIES: dict[str, Callable[[int], FFTBackend]] = {
    "numpy": _numpy_backend,
    "scipy": _scipy_backend,
    "fftw": _fftw_backend,
    "mkl_fft": _mkl_fft_backend,
}


@functools.cache
def _load(name: str, workers: int = 1) -> FFTBackend | None:
    """Instantiate an engine, or ``None`` when its package cannot be imported."""
    try:
        return _FACTORIES[name](workers)
    except ImportError:
        return None


def _usable_cpu_count() -> int:
    """
    CPUs this process may run on.

    ``os.cpu_count()`` counts the whole machine, which over-subscribes threads
    in containers and under CPU affinity (e.g. 128 reported, 32 usable).
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python >= 3.13
    if process_cpu_count is not None:
        count = process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return count or 1


def _resolve_workers(workers: object) -> int:
    if workers is None:
        return _usable_cpu_count()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        msg = f"fft_workers must be a positive integer or None, got {workers!r}"
        raise ValueError(msg)
    return workers


def available_fft_engines() -> list[str]:
    """
    Names of the FFT engines importable in this environment.

    Returns
    -------
    list of str
        Subset of ``["numpy", "scipy", "fftw", "mkl_fft"]``; always contains
        ``"numpy"``.

    Examples
    --------
    >>> from curvelets.numpy._fft import available_fft_engines
    >>> "numpy" in available_fft_engines()
    True
    """
    return [name for name in _FACTORIES if _load(name) is not None]


def _proxy_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Wedge-sized proxy (half per axis), shrunk to the benchmark budget."""
    proxy = [max(1, s // 2) for s in shape]
    while math.prod(proxy) > _BENCHMARK_MAX_ELEMENTS:
        largest = max(range(len(proxy)), key=proxy.__getitem__)
        proxy[largest] = max(1, proxy[largest] // 2)
    return tuple(proxy)


def _time_backend(backend: FFTBackend, data: np.ndarray) -> float:
    backend.ifftn(data)  # warm-up (plan creation, thread pool start-up)
    best = math.inf
    for _ in range(_BENCHMARK_REPEATS):
        start = time.perf_counter()
        backend.fftn(backend.ifftn(data))
        best = min(best, time.perf_counter() - start)
    return best


@functools.cache
def _auto_for_shape(shape: tuple[int, ...], workers: int) -> FFTBackend:
    candidates = [_numpy_backend()] + [
        backend
        for backend in (_load(name, workers) for name in ("scipy", "mkl_fft"))
        if backend is not None
    ]
    if len(candidates) == 1:
        return candidates[0]
    rng = np.random.default_rng(0)
    proxy = _proxy_shape(shape)
    data = rng.standard_normal(proxy) + 1j * rng.standard_normal(proxy)
    timings = {backend: _time_backend(backend, data) for backend in candidates}
    return min(timings, key=timings.__getitem__)


def resolve_fft_engine(
    engine: str, shape: tuple[int, ...], workers: int | None = None
) -> FFTBackend:
    """
    FFT backend for an engine name and transform shape.

    Parameters
    ----------
    engine : {"auto", "numpy", "scipy", "fftw", "mkl_fft"}
        Requested engine. ``"auto"`` picks the fastest of ``numpy`` and the
        installed ``scipy`` / ``mkl_fft`` for this shape (timed once and
        cached).
    shape : tuple of int
        Transform (image) shape.
    workers : int, optional
        Threads for the multithreaded engines (``None``: all CPUs this
        process may use). Ignored by
        ``numpy``.

    Returns
    -------
    FFTBackend
        The engine's FFT functions; ``numpy`` (with a warning) when the
        requested package is not installed.

    Raises
    ------
    ValueError
        If ``engine`` is not a known engine name, or ``workers`` is not a
        positive integer or ``None``.

    Examples
    --------
    >>> from curvelets.numpy._fft import resolve_fft_engine
    >>> resolve_fft_engine("numpy", (64, 64)).name
    'numpy'
    >>> resolve_fft_engine("auto", (64, 64)).name in ("numpy", "scipy", "mkl_fft")
    True
    """
    if engine not in FFT_ENGINES:
        msg = f"fft_engine must be one of {list(FFT_ENGINES)}, got {engine!r}"
        raise ValueError(msg)
    num_workers = _resolve_workers(workers)
    if engine == "auto":
        return _auto_for_shape(tuple(int(s) for s in shape), num_workers)
    backend = _load(engine, num_workers)
    if backend is None:
        msg = (
            f"FFT engine {engine!r} is not available, reverting to numpy. "
            f"To use it, run: {_INSTALL_HINTS[engine]}."
        )
        warnings.warn(msg, stacklevel=3)
        return _numpy_backend()
    return backend
