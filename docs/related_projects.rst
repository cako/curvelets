Related Projects
================

.. include:: table.md
   :parser: myst_parser.sphinx_

About each project
##################

* **Curvelab** implements the fast discrete curvelet transforms (FDCT) of :cite:t:`Candes2006a` in C++ and MATLAB, for 2D and 3D. It is proprietary, but free for academic use.
* **Curvelet.jl** is a Julia implementation of the UDCT :cite:p:`Nguyen2010` for 2D signals. The repository was archived and its last commit was in 2017.
* `Kymatio <https://www.kymat.io/>`_ implements the wavelet scattering transform :cite:p:`Bruna2013,Andreux2019` for PyTorch, TensorFlow and JAX. The scattering transform takes a modulus, so it is not invertible. It is not under active development: the last release is 0.3.0, and since mid-2024 the commits are limited to fixes in tests, CI and dependencies (the most recent is from June 2026). Version 0.3.0 does not import with current SciPy (it needs ``scipy.special.sph_harm``, which was removed).
* `dtcwt <https://dtcwt.readthedocs.io/>`_ is a Python port of the MATLAB dual-tree complex wavelet transform toolbox :cite:p:`Kingsbury2001`, with NumPy, OpenCL and TensorFlow backends. Its repository was archived in July 2024 and states that it is no longer maintained. Version 0.14 does not work with NumPy 2 (it calls the removed ``np.asfarray``).
* `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_ implements the DWT, the DTCWT and a DTCWT-based scattering network in PyTorch, with GPU support :cite:p:`Cotter2019`. It is not archived, but there have been no commits since September 2022.

How fast is it compared to other libraries?
###########################################

The table below shows the speed of the forward transform of one 2D image or 3D volume (batch of one, float32) on an NVIDIA RTX 3070 with 8 GB of memory, for the projects above that have a Python implementation. Each column is relative to its slowest library, which is 1x; higher is faster and the fastest is in bold. The times were measured with ``benchmarks/library_comparison.py``. The absolute times, in milliseconds, and the results for larger sizes (up to :math:`2048 \times 2048`) are in ``benchmarks/data/library_comparison.json``.

.. include:: library_speed_table.rst

The rows marked cpu run on the CPU: the NumPy implementation of **Curvelets** and the NumPy backend of ``dtcwt``. The OpenCL and TensorFlow backends of ``dtcwt`` were not tested. Kymatio was not run at :math:`2048 \times 2048`, because building its filter bank took over 20 minutes on the CPU.

For small and medium inputs, the GPU transform is limited by the cost of launching kernels, which CUDA graphs mostly remove (see :ref:`torch-backend`). At :math:`512 \times 512`, the forward transform takes 0.16 ms, against 0.63 ms for the DWT, 7.3 ms for the dual-tree wavelet transform and 165 ms for Kymatio's scattering transform. At :math:`2048 \times 2048` (not in the table) this advantage is gone: the DWT of Pytorch Wavelets takes 0.73 ms, which is 3x faster than **Curvelets** (2.1 ms), and CUDA graphs no longer help (the eager transform takes 1.75 ms).

The transforms are not equivalent. They differ in redundancy (below), in their directions (three for the DWT, six for the dual-tree transforms, and as many as configured for the curvelets; :ref:`sphx_glr_auto_examples_plot_13_transform_atoms.py` shows what a single coefficient looks like in each of them), and in invertibility: Kymatio's scattering transform is not invertible, so only its forward transform is timed.

How many coefficients does each transform return?
#################################################

The redundancy of a transform is the number of (real-valued) coefficients it returns per input sample. It goes from nearly one for the DWT to about four for the dual-tree wavelet transform and the curvelets in 2D. More coefficients cost memory and time, and buy more directions and stability to shifts, which is a large part of why the DWT is faster at large sizes.

.. include:: library_redundancy_table.rst

References
##########

.. bibliography::
   :filter: docname in docnames
