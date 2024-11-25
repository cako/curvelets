.. toctree::
    :maxdepth: 2
    :hidden:

    self
    curvelets.rst
    auto_examples/index.rst
    API <source/modules.rst>


Overview
========

Curvelets is an open-source implementation of the Uniform Discrete Curvelet Transform (UDCT) :footcite:`Nguyen2010` in the Python programming language for N-dimensional signals.

Getting Started
###############

Installation
------------
Curvelets can be installed directly from the PyPI index:

.. code-block:: sh

    pip install curvelets

Curvelets supports Python 3.9 and above, NumPy 1.20 and above.

First Steps
-----------

Curvelets provides a very simple interface to use the UDCT, :obj:`SimpleUDCT <curvelets.numpy.SimpleUDCT>`.
Its only required argument is the shape of the inputs, but you can also supply the number of "scale" or "resolutions" (``nscales``) as well as the number of bands per direction (``nbands_per_direction``).
The more scales there are, the more granular the distinction between a slowly-varying and a highly-varying feature. The more bands there are, the more granular the distinction between the directions of the features. Explore the :ref:`sphx_glr_auto_examples_plot_02_direction_resolution.py` example to better understand the effect of the scales and the bands on the decomposition.

.. code-block:: python

    import numpy as np
    from curvelets.numpy import SimpleUDCT

    x = np.ones((128, 128))
    C = SimpleUDCT(shape=x.shape)
    y = C.forward(x)
    np.testing.assert_allclose(x, C.backward(y))


Credits
#######
The original Matlab implementation was developed by one of the authors of the UDCT, Truong T. Nguyen.

.. footbibliography::
