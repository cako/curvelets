<div align="center">
<h1>Curvelets</h1>

<strong>⚠️ This project is in <i>very</i> early development! Expect bugs! ⚠️
</strong>

<h3>
  <a href="#getting-started">Getting Started</a>
  <span> | </span>
  <a href="#FAQs">FAQs</a>
  <span> | </span>
  <a href="https://curvelets.readthedocs.io/en/latest/auto_examples/index.html">Examples</a>
</h3>
</div>

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/cako/curvelets/workflows/CI/badge.svg
[actions-link]:             https://github.com/cako/curvelets/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/Curvelets
[conda-link]:               https://github.com/conda-forge/Curvelets-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/cako/curvelets/discussions
[pypi-link]:                https://pypi.org/project/Curvelets/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/Curvelets
[pypi-version]:             https://img.shields.io/pypi/v/Curvelets
[rtd-badge]:                https://readthedocs.org/projects/Curvelets/badge/?version=latest
[rtd-link]:                 https://Curvelets.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
## Getting stated

```python
import numpy as np
from curvelets.numpy import SimpleUDCT

x = np.ones((128, 128))
C = SimpleUDCT(shape=x.shape)
y = C.forward(x)
np.testing.assert_allclose(x, C.backward(y))
```

You can use `SimpleUDCT` very similarly to `FDCT` from [curvelops](https://github.com/PyLops/curvelops), with a limitation that SimpleUDCT only operates arrays whose sizes are powers of 2.



## FAQs

### 0. Should I use this library?

Probably not! This library is a work in progress, pre-alpha library for a very specific version of the curvelet transform known as the Uniform Discrete Curvelet Transform (UDCT).

Consider using other, more mature projects.

|                                                             | Description                         | License                                  | N-D?       | Shape       | Invertible? |
| ----------------------------------------------------------- | ----------------------------------- | ---------------------------------------- | ---------- | ----------- | ----------- |
| curvelets                                                   | UDCT in Python                      | MIT                                      | N-D        | Powers of 2 | Exact       |
| [Curvelab](https://curvelet.org/software.php)               | FDCT in C++ and Matlab              | Proprietary (free for academic use only) | 2D, 3D     | Any         | Exact       |
| [curvelops](https://github.com/PyLops/curvelops)            | Curvelab Python wrapper             | MIT, depedends on Curvelab               | 2D, 3D     | Any         | Exact       |
| [Kymatio](https://www.kymat.io/)                            | Wavelet scattering transform        | BSD 3-clause                             | 1D, 2D, 3D | Any         | Approximate |
| [dtcwt](https://dtcwt.readthedocs.io)                       | Dual-Tree Complex Wavelet Transform | Custom BSD 2-clause                      | 1D, 2D, 3D | Any         | Exact       |
| [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io) | Discrete WT and Dual-Tree CWT       | MIT                                      | 2D         | Any         | Exact       |



### 1. What even *are* curvelets?

   Curvelets are like wavelets, but in 2D (3D, 4D, etc.). But so are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets, ... you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, how fast the signal is varying), "location" (equivalent to time in 1D) and the direction in which the signal is varying (which does not have 1D analog).

   What separates curvelets from the other X-lets are their interesting properties, including:
   * The curvelet transform has an exact inverse
   * The discrete curvelet transform has efficient decomposition ("analysis") and reconstruction ("synthesis") implementations [[1](#1), 2]
   * The curvelet transform is naturally N-dimensional
   * Curvelet basis functions yield an optimally sparse representation of wave phenomena (seismic data, ultrasound data, etc.) [3]
   * Curvelets have little redundancy, forming a _tight frame_ [4]
  
  You can find a good overview (plug: I wrote it!) of curvelets in the Medium article [Demystifying Curvelets](https://towardsdatascience.com/desmystifying-curvelets-c6d88faba0bf).

### 2. Why should I care about curvelets?
   Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

### 3. Why do we need another curvelet transform library?

There are three flavors of the discrete curvelet transform with available implementations. The first two are based on the Fast Discrete Curvelet Transform (FDCT) pioneered by Candès, Demanet, Donoho and Ying. They are the "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions of the FDCT. Both are implemented (2D and 3D for the wrapping version and 2D only for the USFFT version) in the proprietary [CurveLab Toolbox](http://www.curvelet.org/software.html) in Matlab and C++.

As of 2024, any non-academic use of the CurveLab Toolbox requires a commercial license. This includes libraries which wrap the CurveLab toolbox such as pybind11-based wrapper [curvelops](https://github.com/PyLops/curvelops), Python SWIG-based wrapper [PyCurvelab](https://github.com/slimgroup/PyCurvelab). Neither curvelops nor PyCurvelab package include any source code of Curvelab. It should be noted, however, that any library which ports or converts Curvelab code to another language is subject to Curvelab's license. Again, neither curvelops or PyCurvelab do so, and can therefore be freely distributed as per their licenses (both have MIT licenses).

A third flavor is the Uniform Discrete Curvelet Transform (UDCT) which does not have the same restrictive license as the FDCT. The UDCT was first implemented in Matlab (see [ucurvmd](https://github.com/nttruong7/ucurvmd) \[dead link\]) by one of its authors, Truong Nguyen. The 2D version was ported to Julia as the [Curvelet.jl](https://github.com/fundamental/Curvelet.jl) package, whose development has since been abandoned.

This library provides the first pure-Python implementation of the UDCT, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry processionals to use the UDCT more easily.

Note: The Candès FDCTs and Nguyen UDCT are not the only curvelet transforms. To my knowledge, there is another implementation of the 3D Discrete Curvelet Transform named the LR-FCT (Low-Redudancy Fast Curvelet Transform) by Woiselle, Stack and Fadili [5], but the code seems to have disappeared off the internet [6]. Moreover, there is also another type of continuous curvelet transform, the monogenic curvelet transform [7], but I have found no implementation available.

### 4. Can I use curvelets for deep-learning?
This is another facet of the "data-centric" vs "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering, as opposed to using conventional model architectures and letting the data guide the learning process. Alternatively, if the transform is used as a preprocessing step, it can be seen from as feature engineering.

My suggestion is to use curvelets and other transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality tranining data. It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks [8, 9] in some cases.

Another expected to consider is the availability of high-performance, GPU-accelerated and autodiff-friendly libraries. As far as I know, no curvelet library (including this one) satisfies those constraints. Alternative transforms can be found in [Kymatio](https://www.kymat.io/)  and [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/readme.html) which implemente the wavelets scattering [10] and dual-tree complex wavelet transform, respectively [11].

## References
[1] Candès, E., L. Demanet, D. Donoho, and L. Ying, 2006, *Fast Discrete Curvelet Transforms*: Multiscale Modeling & Simulation, 5, 861–899.

[2] Nguyen, T. T., and H. Chauris, 2010, *Uniform Discrete Curvelet Transform*: IEEE Transactions on Signal Processing, 58, 3618–3634.


[3] Candès, E. J., and L. Demanet, 2005, *The curvelet representation of wave propagators is optimally sparse*: Communications on Pure and Applied Mathematics, 58, 1472–1528.

[4] Candès, E. J., and D. L. Donoho, 2003, *New tight frames of curvelets and optimal representations of objects with piecewise C2 singularities*: Communications on Pure and Applied Mathematics, 57, 219–266.

[5] Woiselle, A., J.-L. Starck, and J. Fadili, 2010, *3D curvelet transforms and astronomical data restoration*: Applied and Computational Harmonic Analysis, 28, 171–188.

[6] Starck, Jean-Luc. *F-CUR3D – CosmoStat*: CosmoStat, 26 June 2017, www.cosmostat.org/software/f-cur3d. Accessed 25 Feb. 2024.

[7] Storath, M., 2010, *The monogenic curvelet transform*: 2010 IEEE International Conference on Image Processing.

[8] Luan, S., C. Chen, B. Zhang, J. Han, and J. Liu, 2018, *Gabor Convolutional Networks*: IEEE Transactions on Image Processing, 27, 4357–4366.

[9] Bruna, J., and S. Mallat, 2013, *Invariant Scattering Convolution Networks*: IEEE Transactions on Pattern Analysis and Machine Intelligence, 35, 1872–1886.

[10] Andreux, M., T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. Chaudhary, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, and M. Eickenberg, 2020, *Kymatio: Scattering Transforms in Python*: Journal of Machine Learning Research, 21(60), 1−6.

[11] Kingsbury, N., 2001, Complex Wavelets for Shift Invariant Analysis and Filtering of Signals: Applied and Computational Harmonic Analysis, 10, 234–253.

