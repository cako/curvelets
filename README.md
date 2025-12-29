# Curvelets

**Curvelets** is an open-source implementation of the Uniform Discrete Curvelet Transform (UDCT) in the Python programming language for N-dimensional signals.

<h3>
  <a href="#getting-started">Getting Started</a>
  <span> | </span>
  <a href="#features">Features</a>
  <span> | </span>
  <a href="#faqs">FAQs</a>
  <span> | </span>
  <a href="https://curvelets.readthedocs.io/en/latest/auto_examples/index.html">Examples</a>
</h3>

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status](https://readthedocs.org/projects/curvelets/badge/?version=latest)](https://curvelets.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/cako/curvelets/graph/badge.svg?token=T16E0ANTLR)](https://codecov.io/github/cako/curvelets)

[![PyPI version][pypi-version]][pypi-link]
![PyPI downloads](https://img.shields.io/pypi/dm/curvelets.svg?label=Pypi%20downloads)
![OS-support](https://img.shields.io/badge/OS-linux,win,osx-850A8B.svg)
![License](https://img.shields.io/github/license/cako/curvelets)

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/cako/curvelets/workflows/CI/badge.svg
[actions-link]:             https://github.com/cako/curvelets/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/cako/curvelets/discussions
[pypi-link]:                https://pypi.org/project/Curvelets/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/Curvelets
[pypi-version]:             https://img.shields.io/pypi/v/Curvelets
[rtd-badge]:                https://readthedocs.org/projects/Curvelets/badge/?version=latest
[rtd-link]:                 https://Curvelets.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

## Getting Started

### Installation

**Curvelets** can be installed directly from the PyPI index:

```sh
pip install curvelets
```

**Curvelets** supports Python 3.9 and above, NumPy 1.20 and above.

### First Steps

**Curvelets** provides a very simple interface to use the UDCT, `UDCT`. Its only required argument is the shape of the inputs, but you can also supply the number of "scale" or "resolutions" (`num_scales`) as well as the number of wedges per direction (`wedges_per_direction`). The more scales there are, the more granular the distinction between a slowly-varying and a highly-varying feature. The more wedges there are, the more granular the distinction between the directions of the features.

```python
import numpy as np
from curvelets.numpy import UDCT

x = np.ones((128, 128))
C = UDCT(shape=x.shape)
y = C.forward(x)
np.testing.assert_allclose(x, C.backward(y))
```

## Features

| Feature                  | Status                     |
|--------------------------|----------------------------|
| N-D                      | ✅                         |
| Arbitrary input shapes   | ✅                         |
| Real inputs              | ✅ [^real-ftn]             |
| Complex inputs           | ✅                         |
| Asymmetric directional resolution | ✅ [^asymmetric-ftn] |
| Wavelet at highest scale | ✅ [^wavelet-ftn]          |
| Monogenic coefficients   | ✅ [^monogenic-ftn]        |
| PyTorch bindings         | ✅ [^torch-ftn]            |

[^real-ftn]: Supports real inputs with reduced storage requirements which exploit the symmetry of the real-valued Fourier transform.
[^asymmetric-ftn]: The directional resolution is asymmetric in the sense that the number of wedges per direction is different for each direction.
[^wavelet-ftn]: Isotropic wavelets are supported.
[^monogenic-ftn]: The monogenic curvelet transform was originally defined for 2D signals by Storath 2010, but this implementation extends it to arbitrary N-D signals by using all Riesz transform components (one per dimension).
[^torch-ftn]: PyTorch bindings are supported. `UDCTModule` does not support the monogenic mode yet.

## FAQs

### What are curvelets?

Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

You can find a good overview (plug: I wrote it!) of curvelets in the Medium article [Demystifying Curvelets](https://medium.com/data-science/desmystifying-curvelets-c6d88faba0bf).

Curvelets are like wavelets, but in 2D (3D, 4D, etc.). So are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets… you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, how fast the signal is varying), "location" (equivalent to time in 1D) and the direction in which the signal is varying (which does not have 1D analog).

What separates curvelets from the other X-lets are their interesting properties, including:

* The curvelet transform has an *exact inverse*,
* Forward and inverse discrete curvelet transforms are *efficient* [[1](#1-candès-e-l-demanet-d-donoho-and-l-ying-2006-fast-discrete-curvelet-transforms-multiscale-modeling--simulation-5-861899), [2](#2-nguyen-t-t-and-h-chauris-2010-uniform-discrete-curvelet-transform-ieee-transactions-on-signal-processing-58-36183634)],
* The curvelet transform is *N-dimensional*,
* Curvelets are *optimally sparse* for wave phenomena (seismic, ultrasound, electromagnetic, etc.) [[3](#3-candès-e-j-and-l-demanet-2005-the-curvelet-representation-of-wave-propagators-is-optimally-sparse-communications-on-pure-and-applied-mathematics-58-14721528)],
* Curvelets have little redundancy, forming a [*tight frame*](https://en.wikipedia.org/wiki/Frame_(linear_algebra)#Tight_frames) [[4](#4-candès-e-j-and-d-l-donoho-2003-new-tight-frames-of-curvelets-and-optimal-representations-of-objects-with-piecewise-c2-singularities-communications-on-pure-and-applied-mathematics-57-219266)].

### Why do we need another curvelet transform library?

There are three flavors of the discrete curvelet transform with publicly available implementations [^f1]. The first two are based on the Fast Discrete Curvelet Transform (FDCT) pioneered by Candès, Demanet, Donoho and Ying. They are the "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions of the FDCT. Both are implemented (2D and 3D for the wrapping version and 2D for the USFFT version) in the proprietary [CurveLab Toolbox](http://www.curvelet.org/software.html) in Matlab and C++.

As of 2026, any non-academic use of the CurveLab Toolbox requires a commercial license. Any library which ports or converts Curvelab code to another language is also subject to Curvelab's license. While this does not include libraries which wrap the CurveLab toolbox and therefore do not contain any source code of Curvelab, their usage still requires Curvelab and therefore its license. Such wrappers include [curvelops](https://github.com/PyLops/curvelops), [PyCurvelab](https://github.com/slimgroup/PyCurvelab) which are both MIT licensed.

A third flavor is the Uniform Discrete Curvelet Transform (UDCT) which does not have the same restrictive license as the FDCT. The UDCT was first implemented in Matlab (see [ucurvmd](https://github.com/nttruong7/ucurvmd) [dead link]) by one of its authors, Truong Nguyen. The 2D version was ported to Julia as the [Curvelet.jl](https://github.com/fundamental/Curvelet.jl) package, whose development has since been abandoned.

**This library provides the first open-source, pure-Python implementation of the UDCT**, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry professionals to use the UDCT more easily. It also goes beyond the original implementation by providing support for complex signals, monogenic [[7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing)] extension for real signals and a wavelet transform at the highest scale.

[^f1]: The Candès FDCTs and UDCTs are not the only curvelet transforms. To my knowledge, there is another implementation of the 3D Discrete Curvelet Transform named the LR-FCT (Low-Redudancy Fast Curvelet Transform) by Woiselle, Stack and Fadili [[5](#5-woiselle-a-j-l-starck-and-j-fadili-2010-3d-curvelet-transforms-and-astronomical-data-restoration-applied-and-computational-harmonic-analysis-28-171188)], but the [code](www.cosmostat.org/software/f-cur3d) seems to be unavailable online. The monogenic curvelet transform [[7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing)] does not have a publicly available implementation. The [S2LET](https://astro-informatics.github.io/s2let/) package implements curvelets on the sphere [[8](#8-chan-j-y-h-b-leistedt-t-d-kitching-and-j-d-mcewen-2017-second-generation-curvelets-on-the-sphere-ieee-transactions-on-signal-processing-65-514)].

### Can I use curvelets for deep-learning?

This is another facet of the "data-centric" vs. "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering when used as part of the model architecture, or feature engineering when used as a preprocessing step.

It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks [[9](#9-luan-s-c-chen-b-zhang-j-han-and-j-liu-2018-gabor-convolutional-networks-ieee-transactions-on-image-processing-27-43574366)], [10](#10-bruna-j-and-s-mallat-2013-invariant-scattering-convolution-networks-ieee-transactions-on-pattern-analysis-and-machine-intelligence-35-18721886)] in some cases. My suggestion is to use curvelets or similar transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality training data.

Another aspect to consider is the availability of high-performance, GPU-accelerated, autodiff-friendly libraries. As far as I know, no curvelet library (including this one) satisfies those constraints. Alternative transforms can be found in [Kymatio](https://www.kymat.io/) and [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/) which implement the wavelet scattering transform [[11](#11-andreux-m-t-angles-g-exarchakis-r-leonarduzzi-g-rochette-l-thiry-j-zarka-s-mallat-j-andén-e-belilovsky-j-bruna-v-lostanlen-m-chaudhary-m-j-hirn-e-oyallon-s-zhang-c-cella-and-m-eickenberg-2020-kymatio-scattering-transforms-in-python-journal-of-machine-learning-research-2160-16)] and dual-tree complex wavelet transform [[12](#12-kingsbury-n-2001-complex-wavelets-for-shift-invariant-analysis-and-filtering-of-signals-applied-and-computational-harmonic-analysis-10-234253)], respectively. The former provides PyTorch, TensorFlow and JAX bindings, while the latter provides PyTorch bindings.

### Related Projects

| Project                                                     | Description                         | License                                  | N-D?       | Invertible? |
| ----------------------------------------------------------- | ----------------------------------- | ---------------------------------------- | ---------- | ----------- |
| [Curvelets](https://curvelets.readthedocs.io/)              | UDCT in Python                      | MIT                                      | N-D        | Exact       |
| [Curvelab](https://curvelet.org/software.php)               | FDCT in C++ and Matlab              | Proprietary (free for academic use only) | 2D, 3D     | Exact       |
| [curvelops](https://github.com/PyLops/curvelops)            | Curvelab Python wrapper             | MIT, depedends on Curvelab               | 2D, 3D     | Exact       |
| [Kymatio](https://www.kymat.io/)                            | Wavelet scattering transform        | BSD 3-clause                             | 1D, 2D, 3D | Approximate |
| [dtcwt](https://dtcwt.readthedocs.io)                       | Dual-Tree Complex Wavelet Transform | Custom BSD 2-clause                      | 1D, 2D, 3D | Exact       |
| [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io) | Discrete WT and Dual-Tree CWT       | MIT                                      | 2D         | Exact       |

## Credits

The original Matlab implementation was developed by one of the authors of the UDCT, Truong T. Nguyen. The Python implementation was developed by Carlos Alberto da Costa Filho and Duy Nguyen.

## References
##### [1] Candès, E., L. Demanet, D. Donoho, and L. Ying, 2006, *Fast Discrete Curvelet Transforms*: Multiscale Modeling & Simulation, 5, 861–899.

##### [2] Nguyen, T. T., and H. Chauris, 2010, *Uniform Discrete Curvelet Transform*: IEEE Transactions on Signal Processing, 58, 3618–3634.

##### [3] Candès, E. J., and L. Demanet, 2005, *The curvelet representation of wave propagators is optimally sparse*: Communications on Pure and Applied Mathematics, 58, 1472–1528.

##### [4] Candès, E. J., and D. L. Donoho, 2003, *New tight frames of curvelets and optimal representations of objects with piecewise C2 singularities*: Communications on Pure and Applied Mathematics, 57, 219–266.

##### [5] Woiselle, A., J.-L. Starck, and J. Fadili, 2010, *3D curvelet transforms and astronomical data restoration*: Applied and Computational Harmonic Analysis, 28, 171–188.

##### [6] Starck, Jean-Luc. *F-CUR3D – CosmoStat*: CosmoStat, 26 June 2017, www.cosmostat.org/software/f-cur3d. Accessed 25 Feb. 2024.

##### [7] Storath, M., 2010, *The monogenic curvelet transform*: 2010 IEEE International Conference on Image Processing.

##### [8] Chan, J. Y. H., B. Leistedt, T. D. Kitching, and J. D. McEwen, 2017, *Second-Generation Curvelets on the Sphere*: IEEE Transactions on Signal Processing, 65, 5–14.


##### [9] Luan, S., C. Chen, B. Zhang, J. Han, and J. Liu, 2018, *Gabor Convolutional Networks*: IEEE Transactions on Image Processing, 27, 4357–4366.

##### [10] Bruna, J., and S. Mallat, 2013, *Invariant Scattering Convolution Networks*: IEEE Transactions on Pattern Analysis and Machine Intelligence, 35, 1872–1886.

##### [11] Andreux, M., T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. Chaudhary, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, and M. Eickenberg, 2020, *Kymatio: Scattering Transforms in Python*: Journal of Machine Learning Research, 21(60), 1−6.

##### [12] Kingsbury, N., 2001, *Complex Wavelets for Shift Invariant Analysis and Filtering of Signals*: Applied and Computational Harmonic Analysis, 10, 234–253.
