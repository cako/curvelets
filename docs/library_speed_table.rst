.. list-table::
   :header-rows: 1

   * - Library
     - Device
     - 256x256
     - 512x512
     - 64x64x64
   * - Curvelets (torch)\*
     - cuda
     - **1925x**
     - **1048x**
     - **655x**
   * - Curvelets (NumPy)
     - cpu
     - 50x
     - 14x
     - 5x
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
   * - Kymatio Scattering2D
     - cuda
     - 1x
     - 1x
     - -

Speed relative to the slowest library of each column (1x): 256x256: Kymatio Scattering2D (162 ms); 512x512: Kymatio Scattering2D (165 ms); 64x64x64: dtcwt (NumPy) (169 ms).

\* All with CUDA graphs.
