Related Projects
================

Curvelab
########

Curvelab implements the fast discrete curvelet transforms (FDCT) of :cite:t:`Candes2006a` in C++ and MATLAB, for 2D and 3D signals. It is proprietary and free for academic use only.

Curvelet.jl
###########

Curvelet.jl implements the UDCT :cite:p:`Nguyen2010` in Julia, for 2D signals. The repository is archived, and its last commit is from 2017.

Kymatio
#######

`Kymatio <https://www.kymat.io/>`_ implements the wavelet scattering transform :cite:p:`Bruna2013,Andreux2019` in Python, with PyTorch, TensorFlow and JAX frontends. The scattering transform takes a modulus, so it is not invertible. The project is not under active development: the last release is 0.3.0, and since mid-2024 the commits only fix tests, CI and dependencies. Version 0.3.0 does not import with current SciPy, because it needs ``scipy.special.sph_harm``, which was removed.

dtcwt
#####

`dtcwt <https://dtcwt.readthedocs.io/>`_ is a Python port of the MATLAB dual-tree complex wavelet transform toolbox :cite:p:`Kingsbury2001`, with NumPy, OpenCL and TensorFlow backends. The original repository was archived in July 2024 and states that it is no longer maintained. A fork, `xir4n/dtcwt <https://github.com/xir4n/dtcwt>`_, continued it, but it also looks unmaintained: its last merged change is from June 2024, and its open pull requests, including support for NumPy 2, have not been merged. Version 0.14 does not work with NumPy 2, because it calls the removed ``np.asfarray``.

Pytorch Wavelets
################

`Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_ implements the 2D DWT, the DTCWT and a DTCWT-based scattering network in PyTorch, with GPU support :cite:p:`Cotter2019`. Its last commit is from September 2022.

Summary
#######

.. include:: table.md
   :parser: myst_parser.sphinx_

Performance comparison
######################

The forward transform of one 2D image or 3D volume (batch of one, float32) was timed on an NVIDIA RTX 3070 with 8 GB of memory, with ``benchmarks/library_comparison.py``. In the table, each column is relative to its slowest library, which is 1x: Kymatio Scattering2D at 256x256 (162 ms) and 512x512 (165 ms), and dtcwt (NumPy) at 64x64x64 (169 ms). Higher is faster, and the fastest is in bold. The absolute times, in milliseconds, and the results for larger sizes (up to :math:`2048 \times 2048`) are in ``benchmarks/data/library_comparison.json``.

.. include:: library_speed_table.rst

The rows marked cpu run on the CPU: the NumPy implementation of **Curvelets** and the NumPy backend of ``dtcwt``. The OpenCL rows of ``dtcwt`` pass NumPy arrays in and out, so they include the copies between the host and the device. The TensorFlow row is the backend written for TensorFlow 1, run in graph mode with the outputs fetched to the host. Both agree with the NumPy backend. The OpenCL backend in 3D was not timed, because its output does not match the NumPy backend. Kymatio was not run at :math:`2048 \times 2048`, because building its filter bank took over 20 minutes on the CPU.

For small and medium inputs, the GPU transform is limited by the cost of launching kernels, which CUDA graphs mostly remove (see :ref:`torch-backend`). At :math:`512 \times 512`, the forward transform takes 0.16 ms, against 0.63 ms for the DWT, 2.9 ms for the TensorFlow backend of ``dtcwt``, 6.5 ms for its OpenCL backend, 7.3 ms for the dual-tree wavelet transform of Pytorch Wavelets and 165 ms for the scattering transform of Kymatio. At :math:`2048 \times 2048` (not in the table), the DWT of Pytorch Wavelets takes 0.73 ms, which is 3x faster than **Curvelets** (2.1 ms), and CUDA graphs no longer help (the eager transform takes 1.75 ms).

The transforms are not equivalent. They differ in redundancy (below), in their directions (three for the DWT, six for the dual-tree transforms, and as many as configured for the curvelets; see the :ref:`atom comparison <sphx_glr_auto_examples_plot_13_transform_atoms.py>`), and in invertibility: the scattering transform is not invertible, so only its forward transform is timed.

Redundancy comparison
#####################

The redundancy of a transform is the number of (real-valued) coefficients it returns per input sample. It goes from nearly one for the DWT to about four for the dual-tree wavelet transform and the curvelets in 2D. More coefficients cost memory and time, and buy more directions and stability to shifts, which is a large part of why the DWT is faster at large sizes.

.. include:: library_redundancy_table.rst

References
##########

.. bibliography::
   :filter: docname in docnames
