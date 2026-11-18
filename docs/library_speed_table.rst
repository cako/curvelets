.. list-table::
   :header-rows: 1

   * - Library
     - Device
     - 256x256
     - 512x512
     - 64x64x64
   * - Curvelets (torch)\*
     - cuda
     - **1925.05x**
     - **1048.29x**
     - **655.27x**
   * - Curvelets (NumPy)
     - cpu
     - 49.57x
     - 14.32x
     - 5.06x
   * - pytorch_wavelets DWT
     - cuda
     - 257.68x
     - 261.92x
     - -
   * - pytorch_wavelets DTCWT
     - cuda
     - 22.36x
     - 22.51x
     - -
   * - dtcwt (NumPy)
     - cpu
     - 25.82x
     - 10.07x
     - 1.00x
   * - Kymatio Scattering2D
     - cuda
     - 1.00x
     - 1.00x
     - -

Speed relative to the slowest library of each column (1.00x): 256x256: Kymatio Scattering2D (162 ms); 512x512: Kymatio Scattering2D (165 ms); 64x64x64: dtcwt (NumPy) (169 ms).

\* Best mode of the torch implementation in each column (CUDA graphs, with or without the Triton fold, or eager): 256x256: CUDA graphs; 512x512: CUDA graphs; 64x64x64: CUDA graphs.
