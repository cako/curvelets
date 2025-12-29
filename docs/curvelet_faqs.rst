Curvelet FAQs
=============

What are curvelets?
###################
Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

You can find a good overview (disclaimer: I wrote it!) of curvelets in `Demystifying Curvelets <https://medium.com/data-science/desmystifying-curvelets-c6d88faba0bf>`_.

Curvelets are like wavelets, but in 2D (3D, 4D, etc.). So are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets… you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, *how fast* the signal is varying), "location" (equivalent to time in 1D, that is, *where* the signal is varying), and the direction in which the signal is varying (no 1D analog).

What separates curvelets from the other X-lets are their interesting properties, including:

* The curvelet transform has an *exact inverse*,

* Forward and inverse discrete curvelet transforms are *efficient* :cite:`Candes2006a,Nguyen2010`,

* The curvelet transform is *N-dimensional*,

* Curvelets are *optimally sparse* for wave phenomena (seismic, ultrasound, electromagnetic, etc.) :cite:`Candes2005`,

.. |tight_frame_link| raw:: html

   <i><a href="https://en.wikipedia.org/wiki/Frame_(linear_algebra)#Tight_frames">tight frame</a></i>

* Curvelets have little redundancy, forming a |tight_frame_link| :cite:`Candes2004,Nguyen2010,Storath2010`.

Why do we need another curvelet transform library?
##################################################

There are three flavors of the discrete curvelet transform with publicly available implementations [#f1]_. The first two are based on the Fast Discrete Curvelet Transform (FDCT) pioneered by Candès, Demanet, Donoho and Ying. They are the "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions of the FDCT. Both are implemented (2D and 3D for the wrapping version and 2D for the USFFT version) in the proprietary `CurveLab Toolbox <http://www.curvelet.org/software.html>`_ in MATLAB and C++.

As of 2026, any non-academic use of the CurveLab Toolbox requires a commercial license. Any library which ports or converts Curvelab code to another language is also subject to Curvelab's license.
While this does not include libraries which wrap the CurveLab toolbox and therefore do not contain any source code of Curvelab, their usage still requires Curvelab and therefore its license. Such wrappers include `curvelops <https://github.com/PyLops/curvelops>`_, `PyCurvelab <https://github.com/slimgroup/PyCurvelab>`_, both MIT licensed.

A third flavor is the **Uniform Discrete Curvelet Transform (UDCT)** which does not have the same restrictive license as the FDCT. The UDCT was first implemented in MATLAB (`ucurvmd <https://github.com/nttruong7/ucurvmd>`_ [dead link]) by one of its authors, Truong Nguyen. 

**This library provides the first open-source, pure-Python implementation of the UDCT**, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry professionals to use curvelets more easily. It also goes beyond the original implementation by providing a the support for complex signals, monogenic extension for real signals :cite:`Storath2010`, and a wavelet transform at the highest scale.

.. [#f1] The FDCTs and the UDCT are not the only curvelet transforms. To my knowledge, there is another implementation of the 3D Discrete Curvelet Transform named the LR-FCT (Low-Redudancy Fast Curvelet Transform) :cite:`Woiselle2010`, but the implementation, `F-CUR3D <https://www.cosmostat.org/software/f-cur3d>`__, is unavailable. The monogenic curvelet transform of :cite:t:`Storath2010` does not have a publicly available implementation. The `S2LET <https://astro-informatics.github.io/s2let/>`_ package implements curvelets on the sphere :cite:`Chan2017`.


Can I use the **Curvelets** package for deep learning?
######################################################

Yes! The package provides a PyTorch module, :class:`~curvelets.torch.UDCTModule`, which can be used to train deep networks.

Should I use curvelets for deep learning?
#########################################

This is yet another facet of the "data-centric" vs. "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering when used as part of the model architecture, or feature engineering when used as a preprocessing step.

It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks in some situations :cite:`Luan2018,Andreux2018`. My suggestion is to use curvelets or similar transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality training data.

Another aspect to consider is the availability of high-performance, GPU-accelerated, autodiff-friendly libraries. As far as I know, no curvelet library (apart from this one) satisfies those constraints. Alternative transforms can be found in `Kymatio <https://www.kymat.io/>`_ which implements the wavelet scattering transform :cite:`Bruna2013` in PyTorch, TensorFlow and JAX, and `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_ which implements the dual-tree complex wavelet transform :cite:`Kingsbury2001` in PyTorch.

Related Projects
################
.. include:: table.md
   :parser: myst_parser.sphinx_



References
##########

.. bibliography::
   :filter: docname in docnames
