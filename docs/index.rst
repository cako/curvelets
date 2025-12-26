.. toctree::
    :maxdepth: 2
    :hidden:

    self
    curvelets.rst
    auto_examples/index.rst
    API <source/modules.rst>

.. image:: montage_all_border.png
  :alt: Four scales of curvelets


Overview
========

**Curvelets** is an open-source implementation of the Uniform Discrete Curvelet Transform (UDCT) :footcite:`Nguyen2010` in the Python programming language for N-dimensional signals.

Getting Started
###############

Installation
------------
**Curvelets** can be installed directly from the PyPI index:

.. code-block:: sh

    pip install curvelets

**Curvelets** supports Python 3.9 and above, NumPy 1.20 and above.

First Steps
-----------

**Curvelets** provides a very simple interface to use the UDCT, :obj:`UDCT <curvelets.numpy.UDCT>`.
Its only required argument is the shape of the inputs, but you can also supply the number of "scale" or "resolutions" (``num_scales``) as well as the number of wedges per direction (``wedges_per_direction``).
The more scales there are, the more granular the distinction between a slowly-varying and a highly-varying feature. The more wedges there are, the more granular the distinction between the directions of the features. Explore the :ref:`sphx_glr_auto_examples_plot_02_direction_resolution.py` example to better understand the effect of the scales and the wedges on the decomposition.

.. code-block:: python

    import numpy as np
    from curvelets.numpy import UDCT

    x = np.ones((128, 128))
    C = UDCT(shape=x.shape)
    y = C.forward(x)
    np.testing.assert_allclose(x, C.backward(y))


Credits
#######
The original Matlab implementation was developed by one of the authors of the UDCT, Truong T. Nguyen.
The Python implementation was developed by Carlos Alberto da Costa Filho and Duy Nguyen.

.. footbibliography::
