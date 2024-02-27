# Curvelets

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
## TODO

Beyond providing a new package for curvelets, I would like this package to serve as a template for modern (pure Python) packaging. The overarching goal is to have a project/package where, once a tag is pushed to master, this branch is immediately deployed in pip, conda-forge, docs, etc. Some packages for inspiration:
  * https://github.com/kymatio/kymatio/tree/main
  * https://github.com/PyLops/pylops
  * https://github.com/scientific-python/repo-review

The `curvelets` follows the cookiecutter package as provided by https://learn.scientific-python.org, and as such, many of the tasks below may already be implemented. Other tasks may require making the repo public (e.g., docs via GitHub pages).

### CI/CD
* Set up automated tests
* Set up automatic versioning (from GitHub tags? see: https://learn.scientific-python.org/development/guides/packaging-simple/#versioning and https://github.com/ofek/hatch-vcs)
* Create and deploy pip wheels from GitHub actions onto PyPI (see: https://learn.scientific-python.org/development/guides/gha-pure/ and https://packaging.python.org/en/latest/tutorials/packaging-projects/)
* Create conda-forge package (see: https://conda-forge.org/docs/maintainer/adding_pkgs/)

### Docs
* Write docs in the style of: https://pylops.github.io/curvelops/, that is, including:
  - Overview
  - Installation
  - Examples
  - API
  - Contributing
* Deploy docs with Github actions and github pages
* Set up citation using Zenodo
* Add more examples of curvelet algorithms

### Logic
* Port to PyTorch
* Improve style of code
* Document at least user-facing functions
* Figure out how to restructure the output of the curvelet transform into a single ND cube instead of lists of lists
* Explore alterations to the curvelet transform, e.g.:
  * An "allcurvelets=False" extension where the last scale is a wavelet transform (sum all windows of the last scale and do not decimate)
  * Support for complex signals (remove `imf = 2 * imf + imfl` line at the end, but by properly accounting for the complex part of the signal)
  * A monogenic extension (see [7])


## Frequently Asked Questions

### 1. What even *are* curvelets?

   Curvelets are like wavelets, but in 2D (3D, 4D, etc.). But so are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets, ... you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, how fast the signal is varying), "location" (equivalent to time in 1D) and the direction in which the signal is varying (which does not have 1D analog).

   What separates curvelets from the other X-lets are their interesting properties, including:
   * The curvelet transform has an exact inverse
   * The discrete curvelet transform has efficient decomposition ("analysis") and reconstruction ("synthesis") implementations [1, 2]
   * The curvelet transform is naturally N-dimensional
   * Curvelet basis functions yield an optimally sparse representation of wave phenomena (seismic data, ultrasound data, etc.) [3]
   * Curvelets have little redundancy, forming a _tight frame_ [4]

### 2. Why should I care about curvelets?
   Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

### 3. Why do we need another curvelet transform library?

There are three flavors of the discrete curvelet transform with available implementations. The first two are based on the Fast Discrete Curvelet Transform (FDCT) pioneered by Candès, Demanet, Donoho and Ying. They are the "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions of the FDCT. Both are implemented (2D and 3D for the wrapping version and 2D only for the USFFT version) in the proprietary [CurveLab Toolbox](http://www.curvelet.org/software.html) in Matlab and C++.

As of 2024, any non-academic use of the CurveLab Toolbox requires a commercial license. This includes libraries which wrap the CurveLab toolbox such as pybind11-based wrapper [curvelops](https://github.com/PyLops/curvelops), Python SWIG-based wrapper [PyCurvelab](https://github.com/slimgroup/PyCurvelab). Neither curvelops nor PyCurvelab package include any source code of Curvelab. It should be noted, however, that any library which ports or converts Curvelab code to another language is subject to Curvelab's license. Again, neither curvelops or PyCurvelab do so, and can therefore be freely distributed as per their licenses (both have MIT licenses).

The third flavor is the Uniform Discrete Curvelet Transform (UDCT) which does not have the same restrictive license as the FDCT. The UDCT was first implemented in Matlab (see [ucurvmd](https://github.com/nttruong7/ucurvmd)) by one of its authors, Truong Nguyen. The 2D version was ported to Julia as the [Curvelet.jl](https://github.com/fundamental/Curvelet.jl) package, whose development has since been abandoned.

This library provides the first pure-Python implementation of the UDCT, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry processionals to use the UDCT more easily.

Note: The Candès FDCTs and Nguyen UDCT are not the only curvelet transforms. To my knowledge, there is another implementation of the 3D Discrete Curvelet Transform named the LR-FCT (Low-Redudancy Fast Curvelet Transform) by Woiselle, Stack and Fadili [5], but the code seems to have disappeared off the internet [6]. Moreover, there is also another type of continuous curvelet transform, the monogenic curvelet transform [7], but no implementation seems to be available.

### 4. Can I use curvelets for deep-learning?
This is another facet of the "data-centric" vs "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering, as opposed to using conventional model architectures and letting the data guide the learning process. Alternatively, if the transform is used as a preprocessing step, it can be seen from a feature engineering perspective.

My suggestion is to use curvelets and other transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality tranining data. It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks [8, 9] in some cases.

Another expected to consider is the availability of high-performance, GPU-accelerated and autodiff-friendly libraries. As far as I know, no curvelet library (including this one) satisfies those constraints. If you need a high-performance transform, I would consider using [Kymatio](https://www.kymat.io/) [10] which has a GPU-accelerated PyTorch implementation.

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
