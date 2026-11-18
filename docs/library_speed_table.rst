.. list-table::
   :header-rows: 1

   * - Library
     - Device
     - 256x256
     - 512x512
     - 64x64x64
   * - Curvelets (torch) [#cudagraphs]_
     - cuda
     - **1925x**
     - **1048x**
     - **655x**
   * - Curvelets (NumPy)
     - cpu
     - 50x
     - 14x
     - 5x
   * - Kymatio Scattering2D
     - cuda
     - 1x
     - 1x
     - -
   * - pytorch_wavelets DWT
     - cuda
     - 258x
     - 262x
     - -
   * - pytorch_wavelets DTCWT
     - cuda
     - 22x
     - 23x
     - -
   * - dtcwt (NumPy)
     - cpu
     - 26x
     - 10x
     - 1x
   * - dtcwt (OpenCL)
     - opencl
     - 30x
     - 25x
     - -
   * - dtcwt (TensorFlow)
     - cuda
     - 54x
     - 56x
     - -

.. [#cudagraphs] Best mode of the torch implementation in each column; it was CUDA graphs (``cuda_graphs=True``) in all of them.
