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

"Curvelet" names a family of constructions rather than a single transform, and each construction has its own implementations. It helps to keep three things apart: the curvelet constructions, the software that implements them, and other transforms that serve similar purposes.

Curvelet constructions
**********************

* **First-generation curvelets** apply ridgelet transforms to blocks of a band-pass (wavelet-like) decomposition :cite:`Candes2000`; :cite:t:`Starck2002` gave their digital implementation. In 3D, the BeamCurvelet and RidCurvelet transforms follow the same recipe, for filaments and sheets respectively :cite:`Woiselle2010`. These constructions are much more redundant than the later ones.
* **Second-generation curvelets** tile the frequency domain directly, with parabolic scaling (the width of a curvelet is about the square of its length), and form tight frames :cite:`Candes2004`. Their discretizations differ:

  * the **Fast Discrete Curvelet Transform (FDCT)**, in its "wrapping" and "USFFT" (unequally-spaced Fast Fourier Transform) versions :cite:`Candes2006a`;
  * the **Low-Redundancy Fast Curvelet Transform (LR-FCT)**, a 3D FDCT whose redundancy is about 10 instead of about 25 :cite:`Woiselle2011`;
  * the **Uniform Discrete Curvelet Transform (UDCT)**, which uses smooth frequency windows and uniform decimation, giving an N-dimensional tight frame with low redundancy :cite:`Nguyen2010`.

* **Extensions** of these constructions include monogenic curvelets, which add Riesz transforms to obtain an amplitude and a phase at every scale :cite:`Storath2010,Storath2011`, and curvelets on the sphere :cite:`Chan2017`.

Implementations
***************

The FDCT, pioneered by Candès, Demanet, Donoho and Ying, is implemented in the proprietary `CurveLab Toolbox <http://www.curvelet.org/software.html>`_ in MATLAB and C++: the wrapping version in 2D and 3D, and the USFFT version in 2D.

As of 2026, any non-academic use of the CurveLab Toolbox requires a commercial license. Any library which ports or converts Curvelab code to another language is also subject to Curvelab's license.
While this does not include libraries which wrap the CurveLab toolbox and therefore do not contain any source code of Curvelab, their usage still requires Curvelab and therefore its license. Such wrappers include `curvelops <https://github.com/PyLops/curvelops>`_, `PyCurvelab <https://github.com/slimgroup/PyCurvelab>`_, both MIT licensed.

The UDCT was first implemented in MATLAB (`ucurvmd <https://github.com/nttruong7/ucurvmd>`_ [dead link]) by one of its authors, Truong Nguyen, and later in Julia for 2D signals (`Curvelet.jl <https://github.com/fundamental/Curvelet.jl>`_, whose repository has been archived and read-only since 10 September 2019; its last commit is from 2017). It does not have the restrictions of the FDCT.

**This library provides the first open-source, pure-Python implementation of the UDCT**, borrowing heavily from Nguyen's original implementation. The goal of this library is to allow industry professionals to use curvelets more easily. It also goes beyond the original implementation by providing support for complex signals, monogenic curvelets for real signals :cite:`Storath2010`, and a wavelet transform at the highest scale.

Other constructions have few or no implementations available. The implementation of the LR-FCT, `F-CUR3D <https://www.cosmostat.org/software/f-cur3d>`__, is unavailable. The `S2LET <https://astro-informatics.github.io/s2let/>`_ package, licensed under the GPL-3.0, implements curvelets on the sphere :cite:`Chan2017`.

Similar transforms
******************

Other multiscale, directional transforms, such as the dual-tree complex wavelet transform and the wavelet scattering transform, are not curvelets but serve similar purposes (see `What are curvelets?`_ for more "X-lets"). :doc:`related_projects` compares the libraries that implement them with this one.


Can I use the **Curvelets** package for deep learning?
######################################################

Yes! The PyTorch transform, :class:`~curvelets.torch.UDCT`, is a differentiable ``torch.nn.Module`` that can be used to train deep networks.

Should I use curvelets for deep learning?
#########################################

This is yet another facet of the "data-centric" vs. "model-centric" debate in machine learning. Exploiting curvelets is a type of model engineering when used as part of the model architecture, or feature engineering when used as a preprocessing step.

It has been shown that fixed filter banks can be useful in speeding up training and improving performance of deep neural networks in some situations :cite:`Luan2018,Andreux2018`. My suggestion is to use curvelets or similar transforms for small to mid-sized datasets, especially in niche areas without a wide variety of high-quality training data.

Another aspect to consider is the availability of high-performance, GPU-accelerated, autodiff-friendly libraries. As far as I know, no curvelet library (apart from this one) satisfies those constraints. Alternative transforms can be found in `Kymatio <https://www.kymat.io/>`_ which implements the wavelet scattering transform :cite:`Bruna2013` in PyTorch, TensorFlow and JAX, and `Pytorch Wavelets <https://pytorch-wavelets.readthedocs.io/>`_ which implements the dual-tree complex wavelet transform :cite:`Kingsbury2001` in PyTorch. A speed comparison is in :doc:`related_projects`.

References
##########

.. bibliography::
   :filter: docname in docnames
