.. _cpu-performance:

Speed on CPUs
#############

The NumPy transform (:class:`curvelets.numpy.UDCT`) spends most of its time in FFTs. By default it uses :mod:`numpy.fft`, which runs on one CPU core. Installing a multithreaded FFT library makes it several times faster. For GPUs, see the :ref:`PyTorch backend <torch-backend>`.

Installation
============

Install **Curvelets** with SciPy. This is the recommended setup on every CPU:

.. code-block:: sh

    pip install curvelets[scipy]

Optionally, add another FFT library. **Curvelets** compares the installed ones and uses the fastest:

.. code-block:: sh

    pip install curvelets[mkl]    # Intel MKL (Linux x86_64 and Windows x64)
    pip install curvelets[fftw]   # FFTW, through pyFFTW

Usage
=====

Nothing changes in your code: with the default ``fft_engine="auto"``, creating a transform times the installed libraries on its shape (once per shape) and keeps the fastest.

.. code-block:: python

    from curvelets.numpy import UDCT

    udct = UDCT(shape=(512, 512))
    udct.fft_engine  # the library in use, e.g. "scipy"
    coeffs = udct.forward(image)

You can also choose the library and the number of threads yourself:

.. code-block:: python

    udct = UDCT(shape=(512, 512), fft_engine="scipy", fft_workers=8)

* ``fft_engine``: ``"auto"`` (default), ``"numpy"``, ``"scipy"``, ``"mkl_fft"`` or ``"fftw"``. If the library is not installed, a warning is shown and NumPy is used.
* ``fft_workers``: number of threads. The default uses all the CPUs available to the process (in containers too).

Tips
====

* **Create the** :class:`~curvelets.numpy.UDCT` **once and reuse it** for every image of the same shape.
* **Use float32** inputs when that precision is enough: they are faster and use half the memory.
* **Running several transforms in parallel** (for example one per process)? Set ``fft_workers=1``, or a small number, so they do not compete for the same cores.
* **FFTW is slow to start.** With the standard pyFFTW package, the first transform of each shape takes seconds (up to a minute for large 3D shapes) to prepare. It then runs fast, so it is worth it for many transforms of one shape. ``"auto"`` therefore never picks it; ask for it with ``fft_engine="fftw"``.

AMD CPUs
========

SciPy works well on AMD CPUs. Intel MKL can be very slow on them in 3D; ``"auto"`` notices and uses SciPy instead.

For FFTW, AMD provides an optimized version, `amd-fftw <https://github.com/amd/amd-fftw>`_. When pyFFTW is built against it, **Curvelets** detects it, prepares transforms almost instantly, and ``"auto"`` includes it in its comparison. To install it on Linux:

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

Compared with the previous, single-threaded NumPy implementation, on 32 vCPUs of an AMD EPYC 9575F with ``fft_engine="auto"`` (forward / backward):

.. list-table::
   :header-rows: 1

   * - Transform
     - Speedup
   * - 2D real, 512×512, float64
     - 1.3× / 1.8×
   * - 2D real, 2048×2048, float32
     - 3.7× / 3.3×
   * - 3D real, 256×256×256, float64
     - 4.4× / 4.3×
   * - 2D complex, 1024×1024, float64
     - 2.9× / 2.9×

With only NumPy installed, it is 1.3–2.4 times faster than before. Speedups depend on the CPU, the shape and the number of cores.
