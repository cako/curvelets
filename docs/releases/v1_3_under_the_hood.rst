Curvelets 1.3 under the hood
============================

The transform got faster mostly by doing less: fewer kernel launches, fewer copies, fewer samples touched. This page walks through six of the changes behind the numbers in the :doc:`1.3 release notes <v1_3>`, one mechanism at a time. Timings in each caption come from the pull request that introduced the change.

.. raw:: html
   :file: _diagrams/defs.html

One fused plan replaces the loop over wedges
--------------------------------------------

1.2 ran a chain of kernels for every wedge, so a 2048² transform launched about 1,450 of them and the GPU spent most of its time waiting for the next launch. 1.3 builds one plan whose launch count does not depend on the number of wedges. (:pr:`83`, :pr:`84`)

.. raw:: html
   :file: _diagrams/fused-plan.html

The loop is the cost, not the arithmetic. Collapsing 181 wedge chains into four stages took a 2048² forward from about 1,450 kernel launches to 38 and removed 902 blocking device-to-host copies. The backward transform is the mirror image.

CUDA graphs: record once, replay in one call
--------------------------------------------

After the fused plan, a small transform is still launch-bound: the host takes longer to issue the kernels than the GPU takes to run them. ``UDCT(cuda_graphs=True)`` records the whole flat transform once per shape and replays it. (:pr:`92`)

.. raw:: html
   :file: _diagrams/cuda-graphs.html

Both timelines share one time axis. The 5.7x gap at 512² is entirely host overhead. Replay also works under autograd: the transform is a tight frame, so each direction's gradient is the other direction, and both are replayed. Graphs are opt-in because for large inputs the GPU work dominates and the copies into static buffers cost more than they save (about 1.3 GB of traffic at 2048² with a batch of 8).

A Triton kernel writes each output element once
-----------------------------------------------

The fold adds many window samples into each output bin. The gather version did this level by level with a gather, a multiply and an ``index_add``. On CUDA, 1.3 stores the contributions sorted by target and runs one kernel per direction. (:pr:`90`, :pr:`94`)

.. raw:: html
   :file: _diagrams/triton-fold.html

Sorting by target turns a scatter into a loop. On an RTX 2000 Ada the fold itself is 23–86% faster and the whole transform is 10–36% faster. The exception is a single 512² forward without CUDA graphs (+7%): it is host-bound, and one Triton launch costs a little more host time than the few small torch operations it replaces. :pr:`90` first moved the folds to real arithmetic, which let conjugation happen on the fly and merged the two backward folds into one.

Half-spectrum transforms never build the other half
---------------------------------------------------

The spectrum of a real signal is Hermitian, so half of it is redundant. Both backends read and write only the stored half and get the conjugate half from the symmetry. (:pr:`85`, :pr:`90`, :pr:`97`)

.. raw:: html
   :file: _diagrams/half-spectrum.html

Memory and arithmetic follow the symmetry. The forward transform no longer builds a buffer the size of a full complex spectrum, which saves 10–15% of forward peak memory and up to 13% of time on a GPU. The NumPy backend uses the same idea and, together with cached real weights, is 1.3–2.4x faster without installing any extra library.

Windows are built only where they can be nonzero
------------------------------------------------

Constructing a ``UDCT`` was dominated by window building. 1.3 sorts each angle grid into buckets once, so every wedge's support is a contiguous run of buckets, and evaluates the window only inside its bounding box. (:pr:`103`, :pr:`107`)

.. raw:: html
   :file: _diagrams/sparse-windows.html

The saving is skipping work that multiplies by zero. NumPy windows stay bit-identical to 1.2. Other changes in the same builder: Kronecker products became broadcasts, flips became remaps of sparse indices, and the work runs on a thread pool. A 2048² build drops from 18.95 s to 1.50 s with NumPy and from 21.92 s to 2.06 s with torch, and 2-D peak memory falls 65% (NumPy) and 40% (torch) because angle functions are no longer one dense array per wedge. Torch now calls the same builder, which removed about 800 duplicated lines.

Choosing an FFT engine on NumPy
-------------------------------

The NumPy transform no longer has to use :mod:`numpy.fft`. The engine is resolved once, when the ``UDCT`` is created, so transforms pay nothing per call. See :ref:`cpu-performance` for how to install and choose one. (:pr:`97`)

.. raw:: html
   :file: _diagrams/fft-engines.html

Defaults avoid planning cost; opt-ins trade it for speed. SciPy is the default because it needs no planning and stayed within about 1.5x of the best engine everywhere on a 32-vCPU EPYC. Stock FFTW with ``FFTW_ESTIMATE`` was up to about 100x slower than SciPy on small transforms, so the ``fftw`` engine plans with ``FFTW_MEASURE``, and it detects AMD's FFTW fork, which plans quickly, at run time. On that machine ``"auto"`` picked SciPy for 2048² and 128³ and AMD FFTW for 512² and 256³. Its small proxy problem is not always right, which is why it is opt-in. Worker counts use the process's usable CPUs, not ``os.cpu_count()``.
