.. _cpu-performance:

Speed on CPUs
#############

The NumPy transform (:class:`curvelets.numpy.UDCT`) spends most of its time in FFTs: one full-size FFT of the input, and one small FFT per wedge. :mod:`numpy.fft` runs on a single thread, so the main lever on a CPU is a faster, multithreaded FFT library. This page explains how to choose one, and what else affects speed.

For GPUs, see the :ref:`PyTorch backend <torch-backend>`.

Choosing an FFT engine
======================

The FFT library is chosen when the transform is created, with ``fft_engine``, and its thread count with ``fft_workers``:

.. code-block:: python

    from curvelets.numpy import UDCT

    udct = UDCT(shape=(512, 512), fft_engine="auto", fft_workers=None)
    udct.fft_engine  # the engine actually in use, e.g. "scipy"

.. list-table::
   :header-rows: 1
   :widths: 12 28 60

   * - Engine
     - Install
     - Notes
   * - ``"numpy"``
     - included
     - Single-threaded. Always available.
   * - ``"scipy"``
     - ``pip install curvelets[scipy]``
     - Multithreaded, no planning. **The recommended default**: fast on every CPU we measured.
   * - ``"mkl_fft"``
     - ``pip install curvelets[mkl]``
     - Intel MKL, multithreaded. Wheels exist for Linux x86_64 and Windows x64 only. Fast on some shapes, but can be very slow on AMD CPUs (see below).
   * - ``"fftw"``
     - ``pip install curvelets[fftw]``
     - FFTW via pyFFTW, multithreaded. With the stock FFTW (the pyFFTW wheel), it plans each transform shape on first use, which takes seconds (up to about a minute for large 3D transforms), then is fast; ``"auto"`` never picks it. With AMD's FFTW (see below), planning is nearly instant and ``"auto"`` considers it.
   * - ``"auto"``
     - (default)
     - Times ``numpy`` and the installed engines that plan cheaply (``scipy``, ``mkl_fft``, and ``fftw`` when pyFFTW runs on AMD's FFTW), once per transform shape, on a small proxy problem, and uses the fastest. It skips ``fftw`` on the stock FFTW, whose planning cost would fall on every user with pyFFTW installed.

If a requested engine is not installed, the transform falls back to ``numpy`` with a warning. All engines give the same result up to floating-point rounding.

In short: install SciPy (``pip install curvelets[scipy]``) and keep ``fft_engine="auto"``. Consider ``"fftw"`` when you will run many transforms of the same shape and can afford the one-off planning time.

Threads
=======

``fft_workers`` sets the number of threads of the multithreaded engines. The default (``None``) uses every CPU the process may run on, respecting CPU affinity and container limits (in a container that sees a 128-core host but may use 32 cores, it uses 32). ``numpy`` ignores it.

If you already run several transforms in parallel (for example one per process, or with a thread pool), give each a smaller ``fft_workers`` (often ``1``) so that the threads do not compete for the same cores.

Other things that help
======================

* **Reuse the** :class:`~curvelets.numpy.UDCT` **object.** Creating it computes the windows and resolves the FFT engine (and, for ``"auto"``, times the candidates); ``forward`` and ``backward`` then reuse all of that, and the ``"fftw"`` engine reuses its plans.
* **Use float32 when it is precise enough.** A float32 input gives complex64 coefficients and is usually faster than float64, as well as halving memory.
* **Prefer the real transform** (``transform_kind="real"``, the default) for real data. It works on half of the spectrum (``rfftn``), and the complex transform has twice as many coefficients.

AMD CPUs
========

* **SciPy is a good choice.** On a 32-vCPU AMD EPYC 9575F it was within about 1.5 times of the fastest engine in every case we measured, with no planning.
* **Avoid MKL in 3D.** MKL is tuned for Intel CPUs. On the EPYC, ``"mkl_fft"`` was about 25 times slower than SciPy for a 128×128×128 float32 transform (competitive in 2D). ``"auto"`` measures this and picks SciPy instead.
* **For the best FFTW, build pyFFTW against AMD's FFTW.** AMD maintains an optimized fork of FFTW, `amd-fftw <https://github.com/amd/amd-fftw>`_ (part of AOCL). pyFFTW has no wheel for it, but the ``"fftw"`` engine uses it unchanged once pyFFTW is built against it. On the EPYC it usually planned 2–5 times faster than the stock FFTW, and was the fastest engine for 256×256×256 transforms.

  .. code-block:: sh

      # Build and install amd-fftw in double, single and long double precision
      git clone --depth 1 https://github.com/amd/amd-fftw.git && cd amd-fftw
      for opts in "" "--enable-single" "--enable-long-double"; do
        simd="--enable-sse2 --enable-avx --enable-avx2 --enable-avx512"
        [ "$opts" = "--enable-long-double" ] && simd=""
        ./configure --prefix=$PREFIX --enable-shared --enable-threads --enable-amd-opt $simd $opts CFLAGS="-O3 -march=native"
        make -j MAKEINFO=true && make install MAKEINFO=true && make distclean
      done
      cd ..

      # Build pyFFTW against it, and load it at run time
      PYFFTW_INCLUDE=$PREFIX/include PYFFTW_LIB_DIR=$PREFIX/lib pip install --no-binary pyfftw pyfftw
      export LD_LIBRARY_PATH=$PREFIX/lib

  ``-march=native`` and ``--enable-avx512`` target the CPU the library is built on; drop them for a build that must run on other machines.

  The transform detects AMD's FFTW at run time (on Linux, from the library pyFFTW actually loaded), plans with ``FFTW_ESTIMATE``, which is nearly instant and performed well with it, and lets ``"auto"`` time it against the other engines.

Intel CPUs
==========

MKL is Intel's own FFT library, so ``"mkl_fft"`` (``pip install curvelets[mkl]``) is worth trying; ``"auto"`` compares it with SciPy for you. We have not benchmarked Intel CPUs ourselves.

Reference speedups
==================

Measured on a 32-vCPU AMD EPYC 9575F (Zen 5) with ``fft_workers=32``, comparing the previous NumPy implementation (single-threaded :mod:`numpy.fft`) with the current one using ``fft_engine="auto"``, which chose SciPy or MKL. Forward / backward:

.. list-table::
   :header-rows: 1

   * - Transform
     - Previous (ms)
     - Current, ``"auto"`` (ms)
     - Speedup
   * - real 512×512, float64
     - 7.8 / 11.1
     - 5.8 / 6.2
     - 1.3× / 1.8×
   * - real 2048×2048, float32
     - 189 / 166
     - 52 / 51
     - 3.7× / 3.3×
   * - real 256×256×256, float64
     - 2086 / 2228
     - 470 / 519
     - 4.4× / 4.3×
   * - real 256×256×256, float32
     - 1330 / 3654
     - 333 / 333
     - 4.0× / 11×
   * - complex 1024×1024, float64
     - 69 / 83
     - 24 / 29
     - 2.9× / 2.9×

With only NumPy installed, the current implementation is 1.3–2.4 times faster than the previous one. The full results and the benchmark script are in the repository, in ``benchmarks/numpy_engines_bench.py`` and ``benchmarks/data/``. Timings on a shared machine vary by up to about 2× between runs; treat these as indicative.
