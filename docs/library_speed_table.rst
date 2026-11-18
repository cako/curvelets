.. list-table::
   :header-rows: 1

   * - Library
     - Device
     - 256x256
     - 512x512
     - 1024x1024
     - 2048x2048
     - 64x64x64
     - Redundancy (2D)
   * - **curvelets (torch, CUDA graphs)**
     - cuda
     - 0.0841
     - 0.157
     - 0.55
     - 2.14
     - 0.258
     - 4.1x
   * - curvelets (torch, CUDA graphs, no Triton)
     - cuda
     - 0.0897
     - 0.174
     - 0.634
     - 2.5
     - 0.309
     - 4.1x
   * - curvelets (torch, eager)
     - cuda
     - 0.442
     - 0.448
     - 0.569
     - 1.75
     - 0.563
     - 4.1x
   * - curvelets (NumPy)
     - cpu
     - 3.26
     - 11.5
     - 52.8
     - 510
     - 33.4
     - 4.1x
   * - pytorch_wavelets DWT
     - cuda
     - 0.628
     - 0.628
     - 0.624
     - 0.729
     - -
     - 1.0x
   * - pytorch_wavelets DTCWT
     - cuda
     - 7.24
     - 7.31
     - 7.36
     - 9.77
     - -
     - 4.0x
   * - dtcwt (NumPy)
     - cpu
     - 6.27
     - 16.3
     - 73.6
     - 444
     - 169
     - 4.0x
   * - Kymatio Scattering2D
     - cuda
     - 162
     - 165
     - 162
     - -
     - -
     - 1.6x
