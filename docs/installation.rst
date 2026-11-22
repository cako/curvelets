.. _installation:

Installation
############

**Curvelets** supports Python 3.11 and above, and NumPy 1.20 and above.

Standard installation
=====================

Install the core NumPy package from PyPI:

.. code-block:: sh

    pip install curvelets

Optional dependencies
=====================

**Curvelets** provides optional extras for multithreaded CPU FFT engines and for the PyTorch backend:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Extra
     - Command
     - Description
   * - ``scipy`` (recommended)
     - ``pip install "curvelets[scipy]"``
     - Multithreaded CPU FFTs via :mod:`scipy.fft` (used by default when installed)
   * - ``torch``
     - ``pip install "curvelets[torch]"``
     - :ref:`PyTorch backend <torch-backend>` (:mod:`curvelets.torch`) with GPU, batching, and autograd support
   * - ``fftw``
     - ``pip install "curvelets[fftw]"``
     - FFTW backend via ``pyFFTW`` (including runtime detection of AMD's ``amd-fftw``)
   * - ``mkl``
     - ``pip install "curvelets[mkl]"``
     - Intel MKL FFT backend via ``mkl_fft`` (Linux ``x86_64`` and Windows ``x64``)

Multiple extras can be combined in a single command:

.. code-block:: sh

    pip install "curvelets[scipy,fftw,mkl,torch]"

See :ref:`cpu-performance` for choosing and tuning CPU FFT engines (including ``fft_engine="auto"`` and building against AMD's ``amd-fftw``), and :ref:`torch-backend` for GPU usage, batching, and CUDA graphs.
