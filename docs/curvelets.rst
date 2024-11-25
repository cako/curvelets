Curvelet FAQs
=============

What are curvelets?
###################
Curvelets have a long history and rich history in signal processing. They have been used for a multitude of tasks related in areas such as biomedical imaging (ultrasound, MRI), seismic imaging, synthetic aperture radar, among others. They allow us to extract useful features which can be used to attack problems such as segmentation, inpaining, classification, adaptive subtraction, etc.

You can find a good overview (plug: I wrote it!) of curvelets in the Medium article `Demystifying Curvelets <https://towardsdatascience.com/desmystifying-curvelets-c6d88faba0bf>`_.

Curvelets are like wavelets, but in 2D (3D, 4D, etc.). But so are steerable wavelets, Gabor wavelets, wedgelets, beamlets, bandlets, contourlets, shearlets, wave atoms, platelets, surfacelets, ... you get the idea. Like wavelets, these "X-lets" allow us to separate a signal into different "scales" (analog to frequency in 1D, that is, how fast the signal is varying), "location" (equivalent to time in 1D) and the direction in which the signal is varying (which does not have 1D analog).

What separates curvelets from the other X-lets are their interesting properties, including:

* The curvelet transform has an exact inverse,

* The discrete curvelet transform has efficient decomposition ("analysis") and reconstruction ("synthesis") implementations :cite:`Candes2006a,Nguyen2010`,

* The curvelet transform is naturally N-dimensional,

* Curvelet basis functions yield an optimally sparse representation of wave phenomena (seismic data, ultrasound data, etc.) :cite:`Candes2005`,

* Curvelets have little redundancy, forming a *tight frame* :cite:`Candes2004`.

Why do we need another curvelet transform library?
##################################################

There are three flavors of the discrete curvelet transform with available implementations. The first two are based on the Fast Discrete Curvelet Transform (FDCT) pioneered by Candès, Demanet, Donoho and Ying. They are the "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions of the FDCT. Both are implemented (2D and 3D for the wrapping version and 2D for the USFFT version) in the proprietary `CurveLab Toolbox <http://www.curvelet.org/software.html>`_ in Matlab and C++.

As of 2024, any non-academic use of the CurveLab Toolbox requires a commercial license. Any library which ports or converts Curvelab code to another language is also subject to Curvelab's license.
While this does not include libraries which wrap the CurveLab toolbox and therefore do not contain any source code of Curvelab, their usage still requires Curvelab and therefore its license. Such wrappers include `curvelops <https://github.com/PyLops/curvelops>`_, `PyCurvelab <https://github.com/slimgroup/PyCurvelab>`_ which are both MIT licensed.

A third flavor is the Uniform Discrete Curvelet Transform (UDCT) which does not have the same restrictive license as the FDCT. The UDCT was first implemented in Matlab (see `ucurvmd <https://github.com/nttruong7/ucurvmd>`_ [dead link] by one of its authors, Truong Nguyen. The 2D version was ported to Julia as the `Curvelet.jl <https://github.com/fundamental/Curvelet.jl>`_ package, whose development has since been abandoned.

**This library provides the first open-source, pure-Python implementation of the UDCT**, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry processionals to use the UDCT more easily. Another implementation `ucurv <https://github.com/yud08/ucurv>`_, possibly more efficient that this one, has been developed independently by Duy Nguyen.

.. note::
   The Candès FDCTs and Nguyen UDCT are not the only curvelet transforms. To my knowledge, there is another implementation of the 3D Discrete Curvelet Transform named the LR-FCT (Low-Redudancy Fast Curvelet Transform) by Woiselle, Stack and Fadili :cite:`Woiselle2010`, but the `code <www.cosmostat.org/software/f-cur3d>`__ seems to have disappeared from the internet . Moreover, there is also another type of continuous curvelet transform, the monogenic curvelet transform :cite:`Storath2010`, but I have found no implementation available. Lastly, the `S2LET <https://astro-informatics.github.io/s2let/>`_ package implements the equivalent of curvelets but on the sphere :cite:`Chan2017`.


Can I use curvelets for deep-learning?
######################################

This is another facet of the "data-centric" vs "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering, as opposed to using conventional model architectures and letting the data guide the learning process. Alternatively, if the transform is used as a preprocessing step, it can be seen from as feature engineering.

My suggestion is to use curvelets and other transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality tranining data. It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks :cite:`Luan2018`, :cite:`Andreux2018` in some cases.

Another expected to consider is the availability of high-performance, GPU-accelerated and autodiff-friendly libraries. As far as I know, no curvelet library (including this one) satisfies those constraints. Alternative transforms can be found in `Kymatio <https://www.kymat.io/>`_ and `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_ which implement the wavelet scattering transform :cite:`Bruna2013` and dual-tree complex wavelet transform :cite:`Kingsbury2001`, respectively.

Related Projects
################

.. include:: table.md
   :parser: myst_parser.sphinx_



References
##########

.. bibliography::
