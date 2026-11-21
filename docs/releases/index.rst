.. _releases:

Releases
========

What changed in each version of **Curvelets**. The newest version is first. Each page lists the changes that matter to users; the pull requests behind them are linked, and the full change log of every release is on `GitHub <https://github.com/cako/curvelets/releases>`_.

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 14 18 68

   * - Version
     - Date
     - Highlights
   * - :doc:`1.3 <v1_3>`
     - In development
     - Much faster NumPy and PyTorch transforms, CUDA graphs and a Triton kernel, FFT engines for NumPy, monogenic transform in PyTorch, ``UDCT`` as an ``nn.Module``. See also :doc:`v1_3_under_the_hood`.
   * - :doc:`1.2 <v1_2>`
     - 2026-07-19
     - ``SparseWindow`` for both backends, faster FFTs and view-based indexing, curvelet denoising example, contributing guide.
   * - :doc:`1.1 <v1_1>`
     - 2025-12-29
     - Simpler PyTorch construction, tensor migration with ``.to()``, MNIST classification example.
   * - :doc:`1.0 <v1_0>`
     - 2025-12-29
     - First stable release, with the PyTorch backend. Includes 1.0.1 to 1.0.3.
   * - :doc:`Before 1.0 <pre_1_0>`
     - 2024-11 to 2025-12
     - Alpha and beta releases: the N-dimensional UDCT, the module refactor and the monogenic transform.

.. toctree::
   :maxdepth: 1
   :hidden:

   v1_3
   v1_2
   v1_1
   v1_0
   pre_1_0
