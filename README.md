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
[![Documentation Status][rtd-badge]][rtd-link]
[![codecov](https://codecov.io/github/cako/curvelets/graph/badge.svg?token=T16E0ANTLR)](https://codecov.io/github/cako/curvelets)

[![PyPI version][pypi-version]][pypi-link]
![PyPI downloads](https://img.shields.io/pypi/dm/curvelets.svg?label=Pypi%20downloads)
![Python versions][python-versions]
![OS-support](https://img.shields.io/badge/OS-linux,win,osx-850A8B.svg)
![License](https://img.shields.io/github/license/cako/curvelets)

![NumPy][numpy-badge]
![PyTorch][pytorch-badge]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/cako/curvelets/workflows/CI/badge.svg
[actions-link]:             https://github.com/cako/curvelets/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/cako/curvelets/discussions
[numpy-badge]:               https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white
[pypi-link]:                https://pypi.org/project/Curvelets/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/Curvelets
[pypi-version]:             https://img.shields.io/pypi/v/Curvelets
[python-versions]:          https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue?logo=python&logoColor=white
[pytorch-badge]:             https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white
[rtd-badge]:                https://img.shields.io/readthedocs/curvelets
[rtd-link]:                 https://curvelets.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

## Getting Started

### Installation

**Curvelets** can be installed directly from PyPI:

```sh
pip install curvelets
```

See the [installation guide](https://curvelets.readthedocs.io/en/latest/installation.html) for optional dependencies (`scipy`, `fftw`, `mkl`, and `torch`).

### First Steps

**Curvelets** provides a very simple interface to use the UDCT, `UDCT`. Its only required argument is the shape of the inputs, but you can also supply the number of "scale" or "resolutions" (`num_scales`) as well as the number of wedges per direction (`wedges_per_direction`). The more scales there are, the more granular the distinction between a slowly-varying and a highly-varying feature. The more wedges there are, the more granular the distinction between the directions of the features. Explore the [direction resolution example](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_02_direction_resolution.html) to better understand the effect of the scales and the wedges on the decomposition.

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
| Asymmetric diretionality | ✅ [^asymmetric-ftn]       |
| Wavelet at highest scale | ✅ [^wavelet-ftn]          |
| Monogenic coefficients   | ✅ [^monogenic-ftn]        |
| PyTorch bindings         | ✅ [^torch-ftn]            |

[^real-ftn]: Supports real inputs with reduced storage requirements which exploit the symmetry of the real-valued Fourier transform.
[^asymmetric-ftn]: The directional resolution is asymmetric in the sense that the number of wedges per direction is different for each direction. See the [direction resolution example](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_02_direction_resolution.html) for an example.
[^wavelet-ftn]: Isotropic wavelets are supported. See the [curvelet vs wavelet example](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_05_curvelet_vs_wavelet.html) for an example.
[^monogenic-ftn]: The monogenic curvelet transform was originally defined for 2D signals by Storath [[7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing)], but this implementation extends it to arbitrary N-D signals by using all Riesz transform components (one per dimension). See the [monogenic vs real example](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_14_monogenic_vs_real.html) for where it outperforms the real transform: amplitude and phase, regularity estimation and local orientation at the isotropic scales.
[^torch-ftn]: PyTorch bindings are supported. See the [UDCT module gradcheck example](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_09_udct_module_gradcheck.html) for an example. The torch `UDCT` supports all transform kinds, monogenic included.

## FAQs

### What are curvelets?

Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

You can find a good overview (disclaimer: I wrote it!) of curvelets in the Medium article [Demystifying Curvelets](https://medium.com/data-science/desmystifying-curvelets-c6d88faba0bf).

Curvelets are like wavelets, but in 2D (3D, 4D, etc.). So are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets… you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, *how fast* the signal is varying), "location" (equivalent to time in 1D, that is, *where* the signal is varying), and the direction in which the signal is varying (no 1D analog).

What separates curvelets from the other X-lets are their interesting properties, including:

* The curvelet transform has an *exact inverse*,
* Forward and inverse discrete curvelet transforms are *efficient* [[1](#1-candès-e-l-demanet-d-donoho-and-l-ying-2006-fast-discrete-curvelet-transforms-multiscale-modeling--simulation-5-861899), [2](#2-nguyen-t-t-and-h-chauris-2010-uniform-discrete-curvelet-transform-ieee-transactions-on-signal-processing-58-36183634)],
* The curvelet transform is *N-dimensional*,
* Curvelets are *optimally sparse* for wave phenomena (seismic, ultrasound, electromagnetic, etc.) [[3](#3-candès-e-j-and-l-demanet-2005-the-curvelet-representation-of-wave-propagators-is-optimally-sparse-communications-on-pure-and-applied-mathematics-58-14721528)],
* Curvelets have little redundancy, forming a [*tight frame*](https://en.wikipedia.org/wiki/Frame_(linear_algebra)#Tight_frames) [[4](#4-candès-e-j-and-d-l-donoho-2003-new-tight-frames-of-curvelets-and-optimal-representations-of-objects-with-piecewise-c2-singularities-communications-on-pure-and-applied-mathematics-57-219266), [2](#2-nguyen-t-t-and-h-chauris-2010-uniform-discrete-curvelet-transform-ieee-transactions-on-signal-processing-58-36183634), [7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing)].

### Why do we need another curvelet transform library?

"Curvelet" names a family of constructions rather than a single transform, and each construction has its own implementations. It helps to keep three things apart: the curvelet constructions, the software that implements them, and other transforms that serve similar purposes.

#### Curvelet constructions

* **First-generation curvelets** apply ridgelet transforms to blocks of a band-pass (wavelet-like) decomposition [[18](#18-candès-e-j-and-d-l-donoho-2000-curvelets--a-surprisingly-effective-nonadaptive-representation-for-objects-with-edges-in-a-cohen-c-rabut-and-l-l-schumaker-eds-curve-and-surface-fitting-saint-malo-1999-vanderbilt-university-press-105120)]; their digital implementation is due to Starck et al. [[16](#16-starck-j-l-e-j-candès-and-d-l-donoho-2002-the-curvelet-transform-for-image-denoising-ieee-transactions-on-image-processing-11-670684)]. In 3D, the BeamCurvelet and RidCurvelet transforms follow the same recipe, for filaments and sheets respectively [[15](#15-woiselle-a-j-l-starck-and-j-fadili-2010-3d-curvelet-transforms-and-astronomical-data-restoration-applied-and-computational-harmonic-analysis-28-171188)]. These constructions are much more redundant than the later ones.
* **Second-generation curvelets** tile the frequency domain directly, with parabolic scaling (the width of a curvelet is about the square of its length), and form tight frames [[4](#4-candès-e-j-and-d-l-donoho-2003-new-tight-frames-of-curvelets-and-optimal-representations-of-objects-with-piecewise-c2-singularities-communications-on-pure-and-applied-mathematics-57-219266)]. Their discretizations differ:
  * the **Fast Discrete Curvelet Transform (FDCT)**, in its "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions [[1](#1-candès-e-l-demanet-d-donoho-and-l-ying-2006-fast-discrete-curvelet-transforms-multiscale-modeling--simulation-5-861899)];
  * the **Low-Redundancy Fast Curvelet Transform (LR-FCT)**, a 3D FDCT whose redundancy is about 10 instead of about 25 [[5](#5-woiselle-a-j-l-starck-and-j-fadili-2011-3-d-data-denoising-and-inpainting-with-the-low-redundancy-fast-curvelet-transform-journal-of-mathematical-imaging-and-vision-39-121139)];
  * the **Uniform Discrete Curvelet Transform (UDCT)**, which uses smooth frequency windows and uniform decimation, giving an N-dimensional tight frame with low redundancy [[2](#2-nguyen-t-t-and-h-chauris-2010-uniform-discrete-curvelet-transform-ieee-transactions-on-signal-processing-58-36183634)].
* **Extensions** of these constructions include monogenic curvelets, which add Riesz transforms to obtain an amplitude and a phase at every scale [[7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing), [17](#17-storath-m-2011-directional-multiscale-amplitude-and-phase-decomposition-by-the-monogenic-curvelet-transform-siam-journal-on-imaging-sciences-4-5778)], and curvelets on the sphere [[8](#8-chan-j-y-h-b-leistedt-t-d-kitching-and-j-d-mcewen-2017-second-generation-curvelets-on-the-sphere-ieee-transactions-on-signal-processing-65-514)].

#### Implementations

The FDCT, pioneered by Candès, Demanet, Donoho and Ying, is implemented in the proprietary [CurveLab Toolbox](http://www.curvelet.org/software.html) in MATLAB and C++: the wrapping version in 2D and 3D, and the USFFT version in 2D.

As of 2026, any non-academic use of the CurveLab Toolbox requires a commercial license. Beyond copyright, the FDCT algorithms themselves (both the USFFT and wrapping variants, in 2D and 3D) are patented in the United States: [US 7,840,625 B2](https://patents.google.com/patent/US7840625B2/en), "Methods for performing fast discrete curvelet transforms of data", assigned to Caltech and Stanford. The patent is expected to expire on 25 July 2029 [^f2]; until then, practicing the FDCT in the US may require a patent license from the assignees. This applies to any implementation, including a clean-room reimplementation that shares no code with CurveLab. Any library which ports or converts Curvelab code to another language is additionally subject to Curvelab's license. While this does not include libraries which wrap the CurveLab toolbox and therefore do not contain any source code of Curvelab, their usage still requires Curvelab and therefore its license. Such wrappers include [curvelops](https://github.com/PyLops/curvelops), [PyCurvelab](https://github.com/slimgroup/PyCurvelab), both MIT licensed.

The UDCT was first implemented in MATLAB ([ucurvmd](https://github.com/nttruong7/ucurvmd) [dead link]) by one of its authors, Truong Nguyen, and later in Julia for 2D signals ([Curvelet.jl](https://github.com/fundamental/Curvelet.jl), whose repository has been archived and read-only since 10 September 2019; its last commit is from 2017). It does not have the restrictions of the FDCT.

**This library provides the first open-source, pure-Python implementation of the UDCT**, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry professionals to use curvelets more easily. It also goes beyond the original implementation by providing support for complex signals, monogenic curvelets for real signals [[7](#7-storath-m-2010-the-monogenic-curvelet-transform-2010-ieee-international-conference-on-image-processing)], and a wavelet transform at the highest scale.

Other constructions have few or no implementations available. The implementation of the LR-FCT, [F-CUR3D](https://www.cosmostat.org/software/f-cur3d) [[6](#6-starck-jean-luc-f-cur3d--cosmostat-cosmostat-26-june-2017-wwwcosmostatorgsoftwaref-cur3d-accessed-25-feb-2026)], is unavailable. The [S2LET](https://astro-informatics.github.io/s2let/) package, licensed under the GPL-3.0, implements curvelets on the sphere [[8](#8-chan-j-y-h-b-leistedt-t-d-kitching-and-j-d-mcewen-2017-second-generation-curvelets-on-the-sphere-ieee-transactions-on-signal-processing-65-514)].

#### Similar transforms

Other multiscale, directional transforms, such as the dual-tree complex wavelet transform and the wavelet scattering transform, are not curvelets but serve similar purposes (see [What are curvelets?](#what-are-curvelets) for more "X-lets"). [Related Projects](#related-projects) compares the libraries that implement them with this one.

[^f2]: US 7,840,625 B2 was filed on 7 April 2006 (claiming priority to provisional application 60/669,267 of 7 April 2005) and granted on 23 November 2010 to inventors Candès, Donoho and Demanet. The 1205-day patent term adjustment under [35 U.S.C. 154(b)](https://www.law.cornell.edu/uscode/text/35/154) is printed on the [front page of the issued patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7840625), giving an expiration of 7 April 2026 + 1205 days = 25 July 2029 (see also its [entry on Google Patents](https://patents.google.com/patent/US7840625B2/en)). Patent rights cover making, using or selling the claimed method regardless of how the implementation was written ([35 U.S.C. 271(a)](https://www.law.cornell.edu/uscode/text/35/271)). No non-US counterparts of this patent are listed.

### Can I use the **Curvelets** package for deep learning?

Yes! The PyTorch transform, `UDCT`, is a differentiable `torch.nn.Module` that can be used to train deep networks.

### Should I use curvelets for deep learning?

This is yet another facet of the "data-centric" vs. "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering when used as part of the model architecture, or feature engineering when used as a preprocessing step.

It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks in some situations [[9](#9-luan-s-c-chen-b-zhang-j-han-and-j-liu-2018-gabor-convolutional-networks-ieee-transactions-on-image-processing-27-43574366), [11](#11-andreux-m-t-angles-g-exarchakis-r-leonarduzzi-g-rochette-l-thiry-j-zarka-s-mallat-j-andén-e-belilovsky-j-bruna-v-lostanlen-m-chaudhary-m-j-hirn-e-oyallon-s-zhang-c-cella-and-m-eickenberg-2020-kymatio-scattering-transforms-in-python-journal-of-machine-learning-research-2160-16)]. My suggestion is to use curvelets or similar transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality training data.

Another aspect to consider is the availability of high-performance, GPU-accelerated, autodiff-friendly libraries. As far as I know, no curvelet library (apart from this one) satisfies those constraints. Alternative transforms can be found in [Kymatio](https://www.kymat.io/) which implements the wavelet scattering transform [[10](#10-bruna-j-and-s-mallat-2013-invariant-scattering-convolution-networks-ieee-transactions-on-pattern-analysis-and-machine-intelligence-35-18721886)] in PyTorch, TensorFlow and JAX, and [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/) which implements the dual-tree complex wavelet transform [[12](#12-kingsbury-n-2001-complex-wavelets-for-shift-invariant-analysis-and-filtering-of-signals-applied-and-computational-harmonic-analysis-10-234253)] in PyTorch. A speed comparison is in [Related Projects](#related-projects).

## Related Projects

### Curvelab

Curvelab implements the fast discrete curvelet transforms (FDCT) [[1](#1-candès-e-l-demanet-d-donoho-and-l-ying-2006-fast-discrete-curvelet-transforms-multiscale-modeling--simulation-5-861899)] in C++ and MATLAB, for 2D and 3D signals. It is proprietary and free for academic use only.

### Curvelet.jl

Curvelet.jl implements the UDCT [[2](#2-nguyen-t-t-and-h-chauris-2010-uniform-discrete-curvelet-transform-ieee-transactions-on-signal-processing-58-36183634)] in Julia, for 2D signals. The repository has been archived and read-only since 10 September 2019, and its last commit is from 2017.

### Kymatio

[Kymatio](https://www.kymat.io/) implements the wavelet scattering transform [[10](#10-bruna-j-and-s-mallat-2013-invariant-scattering-convolution-networks-ieee-transactions-on-pattern-analysis-and-machine-intelligence-35-18721886), [13](#13-andreux-m-t-angles-g-exarchakis-r-leonarduzzi-g-rochette-l-thiry-j-zarka-s-mallat-j-andén-e-belilovsky-j-bruna-v-lostanlen-m-j-hirn-e-oyallon-s-zhang-c-cella-and-m-eickenberg-2019-kymatio-scattering-transforms-in-python-arxiv-preprint-arxiv181211214-paper)] in Python, with PyTorch, TensorFlow and JAX frontends. The scattering transform takes a modulus, so it is not invertible. The project is not under active development: the last release is 0.3.0, and since mid-2024 the commits only fix tests, CI and dependencies. Version 0.3.0 does not import with current SciPy.

### dtcwt

[dtcwt](https://dtcwt.readthedocs.io/) is a Python port of the MATLAB dual-tree complex wavelet transform toolbox [[12](#12-kingsbury-n-2001-complex-wavelets-for-shift-invariant-analysis-and-filtering-of-signals-applied-and-computational-harmonic-analysis-10-234253)], with NumPy, OpenCL and TensorFlow backends. The original repository was archived in July 2024 and states that it is no longer maintained. A fork, [xir4n/dtcwt](https://github.com/xir4n/dtcwt), continued it, but it also looks unmaintained: its last merged change is from June 2024, and its open pull requests, including support for NumPy 2, have not been merged. Version 0.14 does not work with NumPy 2.

### Pytorch Wavelets

[Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/) implements the 2D DWT, the DTCWT and a DTCWT-based scattering network in PyTorch, with GPU support [[14](#14-cotter-f-and-s-mclaughlin-2019-fbcotterpytorch_wavelets-zenodo-release-v110-zenodo-doi105281zenodo3471950)]. Its last commit is from September 2022.

### Summary

| Project                                                     | Algorithm                            | Language        | License                                  | N-dimensional?       | Invertible? |
| ----------------------------------------------------------- | ------------------------------------ | --------------- | ---------------------------------------- | ------------------- | ----------- |
| [Curvelets](https://curvelets.readthedocs.io/)              | UDCT                                 | Python          | MIT                                      | Yes        | Yes       |
| [Curvelab](https://curvelet.org/software.php)               | FDCT                                 | C++ and MATLAB  | Proprietary (free for academic use only) | 2D, 3D     | Yes       |
| [Curvelet.jl](https://github.com/fundamental/Curvelet.jl)   | UDCT                                 | Julia           | MIT                                      | 2D         | Yes       |
| [Kymatio](https://www.kymat.io/)                            | Wavelet scattering transform        | Python          | BSD 3-clause                             | 1D, 2D, 3D | No |
| [dtcwt](https://dtcwt.readthedocs.io)                       | Dual-Tree Complex Wavelet Transform  | Python          | BSD 2-Clause                      | 1D, 2D, 3D | Yes       |
| [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io) | Discrete WT and Dual-Tree CWT       | Python          | MIT                                      | 2D         | Yes       |

### Performance comparison

The forward transform of one 2D image or 3D volume (batch of one, float32) was timed on an NVIDIA RTX 3070 with 8 GB of memory, with `benchmarks/library_comparison.py`. In the table, each column is relative to its slowest library, which is 1x: Kymatio Scattering2D at 256x256 (162 ms) and 512x512 (165 ms), and dtcwt (NumPy) at 64x64x64 (169 ms). Higher is faster, and the fastest is in bold. The absolute times, in milliseconds, and the results for larger sizes (up to $2048 \times 2048$) are in `benchmarks/data/library_comparison.json`.

| Library | Device | 256x256 | 512x512 | 64x64x64 |
|---|---|---:|---:|---:|
| Curvelets (torch)[^cudagraphs] | cuda | **1925x** | **1048x** | **655x** |
| Curvelets (NumPy) | cpu | 50x | 14x | 5x |
| Kymatio Scattering2D | cuda | 1x | 1x | - |
| pytorch_wavelets DWT | cuda | 258x | 262x | - |
| pytorch_wavelets DTCWT | cuda | 22x | 23x | - |
| dtcwt (NumPy) | cpu | 26x | 10x | 1x |
| dtcwt (OpenCL) | opencl | 30x | 25x | - |
| dtcwt (TensorFlow) | cuda | 54x | 56x | - |

[^cudagraphs]: Best mode of the torch implementation in each column; it was CUDA graphs (`cuda_graphs=True`) in all of them.

The rows marked cpu run on the CPU: the NumPy implementation of **Curvelets** and the NumPy backend of `dtcwt`. The OpenCL rows of `dtcwt` pass NumPy arrays in and out, so they include the copies between the host and the device. The TensorFlow row is the backend written for TensorFlow 1, run in graph mode with the outputs fetched to the host. Both agree with the NumPy backend. The OpenCL backend in 3D was not timed, because its output does not match the NumPy backend. Kymatio was not run at $2048 \times 2048$, because building its filter bank took over 20 minutes on the CPU.

For small and medium inputs, the GPU transform is limited by the cost of launching kernels, which CUDA graphs mostly remove. At $512 \times 512$, the forward transform takes 0.16 ms, against 0.63 ms for the DWT, 2.9 ms for the TensorFlow backend of `dtcwt`, 6.5 ms for its OpenCL backend, 7.3 ms for the dual-tree wavelet transform of Pytorch Wavelets and 165 ms for the scattering transform of Kymatio. At $2048 \times 2048$ (not in the table), the DWT of Pytorch Wavelets takes 0.73 ms, which is 3x faster than **Curvelets** (2.1 ms), and CUDA graphs no longer help (the eager transform takes 1.75 ms).

The transforms are not equivalent. They differ in redundancy (below), in their directions (three for the DWT, six for the dual-tree transforms, and as many as configured for the curvelets; see the [atom comparison](https://curvelets.readthedocs.io/en/latest/auto_examples/plot_13_transform_atoms.html)), and in invertibility: the scattering transform is not invertible, so only its forward transform is timed.

### Redundancy comparison

The redundancy of a transform is the number of (real-valued) coefficients it returns per input sample. It goes from nearly one for the DWT to about four for the dual-tree wavelet transform and the curvelets in 2D. More coefficients cost memory and time, and buy more directions and stability to shifts, which is a large part of why the DWT is faster at large sizes.

| Library | 256x256 | 64x64x64 |
|---|---:|---:|
| Curvelets | 4.06x | 7.73x |
| pytorch_wavelets DWT | 1.06x | - |
| pytorch_wavelets DTCWT | 4.00x | - |
| dtcwt | 4.00x | 8.00x |
| Kymatio Scattering2D | 1.63x | - |

## Credits

The original MATLAB implementation was developed by one of the authors of the UDCT, Truong T. Nguyen. For a full list of authors, see [AUTHORS](https://github.com/cako/curvelets/blob/main/AUTHORS.md).

## References
##### [1] Candès, E., L. Demanet, D. Donoho, and L. Ying, 2006, *Fast Discrete Curvelet Transforms*: Multiscale Modeling & Simulation, 5, 861–899.

##### [2] Nguyen, T. T., and H. Chauris, 2010, *Uniform Discrete Curvelet Transform*: IEEE Transactions on Signal Processing, 58, 3618–3634.

##### [3] Candès, E. J., and L. Demanet, 2005, *The curvelet representation of wave propagators is optimally sparse*: Communications on Pure and Applied Mathematics, 58, 1472–1528.

##### [4] Candès, E. J., and D. L. Donoho, 2003, *New tight frames of curvelets and optimal representations of objects with piecewise C2 singularities*: Communications on Pure and Applied Mathematics, 57, 219–266.

##### [5] Woiselle, A., J.-L. Starck, and J. Fadili, 2011, *3-D Data Denoising and Inpainting with the Low-Redundancy Fast Curvelet Transform*: Journal of Mathematical Imaging and Vision, 39, 121–139.

##### [6] Starck, Jean-Luc. *F-CUR3D – CosmoStat*: CosmoStat, 26 June 2017, www.cosmostat.org/software/f-cur3d. Accessed 25 Feb. 2026.

##### [7] Storath, M., 2010, *The monogenic curvelet transform*: 2010 IEEE International Conference on Image Processing.

##### [8] Chan, J. Y. H., B. Leistedt, T. D. Kitching, and J. D. McEwen, 2017, *Second-Generation Curvelets on the Sphere*: IEEE Transactions on Signal Processing, 65, 5–14.


##### [9] Luan, S., C. Chen, B. Zhang, J. Han, and J. Liu, 2018, *Gabor Convolutional Networks*: IEEE Transactions on Image Processing, 27, 4357–4366.

##### [10] Bruna, J., and S. Mallat, 2013, *Invariant Scattering Convolution Networks*: IEEE Transactions on Pattern Analysis and Machine Intelligence, 35, 1872–1886.

##### [11] Andreux, M., T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. Chaudhary, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, and M. Eickenberg, 2020, *Kymatio: Scattering Transforms in Python*: Journal of Machine Learning Research, 21(60), 1−6.

##### [12] Kingsbury, N., 2001, *Complex Wavelets for Shift Invariant Analysis and Filtering of Signals*: Applied and Computational Harmonic Analysis, 10, 234–253.

##### [13] Andreux, M., T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, and M. Eickenberg, 2019, *Kymatio: Scattering Transforms in Python*: arXiv preprint arXiv:1812.11214. [(paper)](https://arxiv.org/abs/1812.11214)

##### [14] Cotter, F., and S. McLaughlin, 2019, *fbcotter/pytorch_wavelets: Zenodo Release* (v1.1.0): Zenodo, [doi:10.5281/zenodo.3471950](https://zenodo.org/records/3471950).

##### [15] Woiselle, A., J.-L. Starck, and J. Fadili, 2010, *3D curvelet transforms and astronomical data restoration*: Applied and Computational Harmonic Analysis, 28, 171–188.

##### [16] Starck, J.-L., E. J. Candès, and D. L. Donoho, 2002, *The curvelet transform for image denoising*: IEEE Transactions on Image Processing, 11, 670–684.

##### [17] Storath, M., 2011, *Directional Multiscale Amplitude and Phase Decomposition by the Monogenic Curvelet Transform*: SIAM Journal on Imaging Sciences, 4, 57–78.

##### [18] Candès, E. J., and D. L. Donoho, 2000, *Curvelets – a surprisingly effective nonadaptive representation for objects with edges*: in A. Cohen, C. Rabut, and L. L. Schumaker, eds., Curve and Surface Fitting: Saint-Malo 1999, Vanderbilt University Press, 105–120.
