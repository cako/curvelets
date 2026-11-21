.. _cpu-performance:

Speed on CPUs
#############

The NumPy transform (:class:`curvelets.numpy.UDCT`) spends most of its time in FFTs. It uses SciPy's multithreaded FFT when SciPy is installed, and :mod:`numpy.fft`, which runs on one CPU core, otherwise. Other FFT libraries can be selected too. For GPUs, see the :ref:`PyTorch backend <torch-backend>`.

.. versionadded:: 1.3
   Selectable FFT libraries (``fft_engine``, ``fft_workers``), with SciPy as the default when it is installed.

Installation
============

Install **Curvelets** with SciPy. This is the recommended setup on every CPU, and nothing else is needed: when SciPy is installed, the transform uses it by default.

.. code-block:: sh

    pip install curvelets[scipy]

Optionally, add another FFT library; ``fft_engine="auto"`` (below) compares the installed ones and uses the fastest:

.. code-block:: sh

    pip install curvelets[mkl]    # Intel MKL (Linux x86_64 and Windows x64)
    pip install curvelets[fftw]   # FFTW, through pyFFTW

Usage
=====

By default the transform uses SciPy when it is installed, and NumPy otherwise:

.. code-block:: python

    from curvelets.numpy import UDCT

    udct = UDCT(shape=(512, 512))
    udct.fft_engine  # "scipy" if SciPy is installed, otherwise "numpy"
    coeffs = udct.forward(image)

To choose the library yourself, pass ``fft_engine``, for example ``UDCT(shape=(512, 512), fft_engine="mkl_fft")``. Or let **Curvelets** pick the fastest installed one with ``fft_engine="auto"``. It times the candidates when the first transform of a shape is created (a fraction of a second for most shapes), and reuses that choice afterwards:

.. code-block:: python

    udct = UDCT(shape=(512, 512), fft_engine="auto")
    udct.fft_engine  # the library it chose, e.g. "scipy"

* ``fft_engine``: ``"scipy"``, ``"numpy"``, ``"mkl_fft"``, ``"fftw"`` or ``"auto"``. The default is ``"scipy"`` when SciPy is installed and ``"numpy"`` otherwise. If a library you name is not installed, a warning is shown and NumPy is used.
* ``fft_workers``: number of threads. The default uses all the CPUs available to the process (in containers too).

Tips
====

* **Create the** :class:`~curvelets.numpy.UDCT` **once and reuse it** for every image of the same shape. Across runs or processes, save it instead of rebuilding it (see `Saving a built transform`_).
* **Use float32** inputs when that precision is enough: they are faster and use half the memory.
* **Running several transforms in parallel** (for example one per process)? Set ``fft_workers=1``, or a small number, so they do not compete for the same cores.
* **FFTW is slow to start.** With the standard pyFFTW package, the first transform of each shape takes seconds (up to a minute for large 3D shapes) to prepare. It then runs fast, so it is worth it for many transforms of one shape. ``"auto"`` therefore never picks it; ask for it with ``fft_engine="fftw"``.

Saving a built transform
========================

Creating a transform builds its windows with ``fft_workers`` threads (by default, every CPU the process may use): on a 4-core laptop CPU, this takes about 1.5 s for a :math:`2048 \times 2048` transform with 5 scales and 0.5 s for :math:`128 \times 128 \times 128` with 3 scales. If you create the same transform often (in every run of a script, or in every worker of a pool), build it once and save it with :mod:`pickle`. A saved transform keeps everything it precomputed, and loads in about 0.2 s at :math:`2048 \times 2048`:

.. code-block:: python

    import pickle

    from curvelets.numpy import UDCT

    transform = UDCT(shape=(2048, 2048), num_scales=5)
    with open("udct_2048.pkl", "wb") as f:
        pickle.dump(transform, f)

    with open("udct_2048.pkl", "rb") as f:  # later, or in another process
        transform = pickle.load(f)

The file is large (about 280 MB at :math:`2048 \times 2048`). Loading it runs code from the file, so only load files you created. It is tied to the version of **Curvelets** that wrote it, so rebuild it after upgrading.

AMD CPUs
========

SciPy, the default when installed, works well on AMD CPUs. Avoid choosing Intel MKL explicitly there: it can be very slow on AMD CPUs for 3D transforms. ``"auto"`` handles this on its own: it times every installed library on your shape and keeps the fastest, so it uses MKL only for shapes where MKL is actually fast.

For FFTW, AMD provides an optimized version, `amd-fftw <https://github.com/amd/amd-fftw>`_. When pyFFTW is built against it, **Curvelets** detects it and prepares transforms almost instantly, so ``"auto"`` includes it among the libraries it times (the standard FFTW is left out because it is slow to prepare). To install it on Linux:

.. code-block:: sh

    # 1. Build amd-fftw (all three precisions) into $PREFIX
    git clone --depth 1 https://github.com/amd/amd-fftw.git && cd amd-fftw
    for opts in "" "--enable-single" "--enable-long-double"; do
      simd="--enable-sse2 --enable-avx --enable-avx2 --enable-avx512"
      [ "$opts" = "--enable-long-double" ] && simd=""
      ./configure --prefix=$PREFIX --enable-shared --enable-threads --enable-amd-opt $simd $opts CFLAGS="-O3 -march=native"
      make -j MAKEINFO=true && make install MAKEINFO=true && make distclean
    done
    cd ..

    # 2. Build pyFFTW against it, and use it at run time
    PYFFTW_INCLUDE=$PREFIX/include PYFFTW_LIB_DIR=$PREFIX/lib pip install --no-binary pyfftw pyfftw
    export LD_LIBRARY_PATH=$PREFIX/lib

Drop ``-march=native`` and ``--enable-avx512`` if the library must also run on other machines.

Intel CPUs
==========

Try Intel MKL: ``pip install curvelets[mkl]``. ``"auto"`` compares it with SciPy and uses the faster one.

Reference speedups
==================

Speedup over NumPy's FFT (``fft_engine="numpy"``), measured on 32 vCPUs of an AMD EPYC 9575F with ``fft_workers=32`` (forward / backward):

.. list-table::
   :header-rows: 1

   * - Transform
     - SciPy (default)
     - ``"auto"``
   * - 2D real, 512×512, float64
     - 0.8× / 0.8×
     - 1.0× / 1.0×
   * - 2D real, 2048×2048, float32
     - 1.3× / 1.1×
     - 1.8× / 2.1×
   * - 3D real, 128×128×128, float32
     - 1.7× / 1.8×
     - 1.7× / 2.0×
   * - 3D real, 256×256×256, float64
     - 2.6× / 2.3×
     - 3.5× / 3.1×
   * - 2D complex, 1024×1024, float64
     - 1.8× / 1.6×
     - 2.2× / 2.0×

Multithreading pays off for larger transforms; for small ones such as 512×512, NumPy's single-threaded FFT is about as fast. Speedups depend on the CPU, the shape and the number of cores, and these were measured on a shared machine, so treat them as indicative.
