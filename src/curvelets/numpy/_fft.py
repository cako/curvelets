"""
FFT engines for the NumPy UDCT.

The transform is FFT-bound, and ``numpy.fft`` runs on a single thread. As in
PyLops, the FFT implementation is chosen per transform with an ``engine``:

- ``"numpy"``: :mod:`numpy.fft` (always available);
- ``"scipy"``: :mod:`scipy.fft`, multithreaded with ``workers``;
- ``"fftw"``: FFTW through ``pyFFTW``, multithreaded. Plans are made on the
  first transform of each shape and kept for the life of the engine (see
  :class:`_FFTWPlans`). With the stock FFTW they use ``FFTW_MEASURE``
  (seconds; up to about a minute for large 3D transforms): its
  ``FFTW_ESTIMATE`` plans split small FFTs over every thread and ran up to
  100x slower on a 32-vCPU EPYC. With AMD's FFTW (amd-fftw / AOCL, detected
  at run time, see :func:`_fftw_is_amd`) they use ``FFTW_ESTIMATE``, which
  plans almost instantly and performed well;
- ``"mkl_fft"``: Intel MKL through ``mkl_fft``, multithreaded
  (:mod:`mkl_fft.interfaces.scipy_fft`, which needs ``mkl-service``; without
  it, :mod:`mkl_fft.interfaces.numpy_fft` with MKL's default thread count);
- ``"auto"`` (default): ``numpy`` and the importable engines that plan
  cheaply (``scipy``, ``mkl_fft``, and ``fftw`` when pyFFTW runs on AMD's
  FFTW) are timed once per transform shape on a wedge-sized proxy problem, and
  the fastest is kept. ``fftw`` on the stock FFTW is never chosen
  automatically: its planning cost would fall on every user with pyFFTW
  installed.

``workers`` sets the thread count of the multithreaded engines (default: all
CPUs). A requested engine whose package is not installed falls back to
``numpy`` with a warning. Every engine returns the same values up to floating-point
rounding.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
import re
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    #: Whether a first transform costs about as much as later ones (no slow
    #: planning), which makes the engine a candidate for ``"auto"``.
    cheap_planning: bool = True


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


class _FFTWPlans:
    """
    FFTW plans for one engine, made once per transform and kept.

    ``pyfftw.interfaces`` caches plans for only 0.1 s by default and keys them on
    the input's memory alignment, so a UDCT re-planned many of its FFTs on
    every call (seconds per call for large 3D transforms with many threads).
    Here a plan is keyed on what defines the transform (function, shape, dtype,
    ``s``, ``axes``) and kept; it owns aligned input/output buffers, so the
    caller's alignment does not matter. Plans are per thread, as executing one
    FFTW plan from two threads at once is not safe.

    ``FFTW_MEASURE`` makes the first transform of each shape slow (FFTW times
    candidate algorithms, including how many of the ``workers`` threads to
    use), and every later one fast. Plans hold their buffers (about the size of
    their input and output) for the life of the engine.
    """

    def __init__(self, pyfftw: Any, workers: int, effort: str = "FFTW_MEASURE") -> None:
        self._pyfftw = pyfftw
        self._workers = workers
        self._effort = effort
        self._local = threading.local()

    def _plan(
        self, func: str, x: np.ndarray, s: Any, axes: Any
    ) -> Callable[[np.ndarray], np.ndarray]:
        plans: dict[tuple[Any, ...], Any] = self._local.__dict__.setdefault("plans", {})
        s_key = None if s is None else tuple(s)
        axes_key = None if axes is None else tuple(axes)
        key = (func, x.shape, x.dtype.str, s_key, axes_key)
        plan = plans.get(key)
        if plan is None:
            # Plan on a scratch array: FFTW_MEASURE overwrites its input.
            scratch = self._pyfftw.empty_aligned(x.shape, dtype=x.dtype)
            plan = getattr(self._pyfftw.builders, func)(
                scratch,
                s=s_key,
                axes=axes_key,
                threads=self._workers,
                planner_effort=self._effort,
                avoid_copy=False,
            )
            plans[key] = plan
        return plan  # type: ignore[no-any-return]

    def _run(self, func: str, x: Any, s: Any = None, axes: Any = None) -> np.ndarray:
        x = np.asarray(x)
        # The plan copies x into its own buffer (the caller's array is never
        # overwritten, even by c2r transforms), normalizes inverse transforms
        # like numpy.fft, and returns its own output buffer: copy it out.
        return self._plan(func, x, s, axes)(x).copy()

    def fftn(self, x: Any, s: Any = None, axes: Any = None) -> np.ndarray:
        return self._run("fftn", x, s, axes)

    def ifftn(self, x: Any, s: Any = None, axes: Any = None) -> np.ndarray:
        return self._run("ifftn", x, s, axes)

    def rfftn(self, x: Any, s: Any = None, axes: Any = None) -> np.ndarray:
        return self._run("rfftn", x, s, axes)

    def irfftn(self, x: Any, s: Any = None, axes: Any = None) -> np.ndarray:
        return self._run("irfftn", x, s, axes)


_FFTW_LIBRARY = re.compile(r"libfftw3[fl]?([-.]|$)")
_AOCL_VERSION_SYMBOLS = ("fftw_aoclversion", "fftwf_aoclversion", "fftwl_aoclversion")


def _fftw_is_amd(maps_path: str = "/proc/self/maps") -> bool:
    """
    Whether the FFTW that pyFFTW loaded is AMD's (amd-fftw, part of AOCL).

    AMD's FFTW exports ``fftw_aoclversion``; the stock FFTW does not. The
    libraries actually mapped into the process are checked (pyFFTW wheels
    bundle their own FFTW, so a system-wide AMD FFTW must not count). Linux
    only: elsewhere, and on any error, the stock FFTW is assumed.

    Parameters
    ----------
    maps_path : str, optional
        The process's memory map; a parameter for testing.

    Returns
    -------
    bool
        True when a loaded FFTW library exports an AOCL version symbol.

    Examples
    --------
    >>> from curvelets.numpy._fft import _fftw_is_amd
    >>> _fftw_is_amd("/nonexistent")
    False
    """
    try:
        with Path(maps_path).open(encoding="utf-8") as maps:
            paths = {
                line.split(maxsplit=5)[-1].strip()
                for line in maps
                if "libfftw3" in line and len(line.split()) == 6
            }
    except OSError:
        return False
    for path in sorted(paths):
        if not _FFTW_LIBRARY.match(Path(path).name):
            continue
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        if any(hasattr(library, symbol) for symbol in _AOCL_VERSION_SYMBOLS):
            return True
    return False


def _fftw_backend(workers: int) -> FFTBackend:
    import pyfftw.builders  # pylint: disable=import-outside-toplevel,import-error

    amd = _fftw_is_amd()
    plans = _FFTWPlans(
        pyfftw, workers, effort="FFTW_ESTIMATE" if amd else "FFTW_MEASURE"
    )
    return FFTBackend(
        "fftw",
        plans.fftn,
        plans.ifftn,
        plans.rfftn,
        plans.irfftn,
        cheap_planning=amd,
    )


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
        for backend in (_load(name, workers) for name in ("scipy", "mkl_fft", "fftw"))
        if backend is not None and backend.cheap_planning
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
        installed engines that plan cheaply (``scipy``, ``mkl_fft``, and
        ``fftw`` on AMD's FFTW) for this shape (timed once and cached).
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
