.. _torch-backend:

PyTorch backend
###############

The :mod:`curvelets.torch` package provides the same transform as :mod:`curvelets.numpy` on PyTorch tensors, on the CPU or on a GPU, with autograd support. Install it with ``pip install curvelets[torch]``.

Two classes are available:

* :class:`~curvelets.torch.UDCT` is the transform itself: ``forward`` returns the nested coefficient structure, ``vect`` flattens it to a single vector and ``struct`` does the opposite.
* :class:`~curvelets.torch.UDCTModule` is a :class:`torch.nn.Module` that wraps a :class:`~curvelets.torch.UDCT`. It maps an image straight to the flat coefficient vector and back, and is differentiable, so it can be used inside a network.

Moving to a GPU
===============

A :class:`~curvelets.torch.UDCTModule` moves like any other module:

.. code-block:: python

    import torch
    from curvelets.torch import UDCTModule

    module = UDCTModule(shape=(512, 512)).to("cuda")
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    y = module(x)
    y.abs().sum().backward()  # gradients flow through the transform

Moving a transform with ``.to(device)`` moves everything it holds: the windows, their precomputed periodized index maps (and, for the complex transform, the flipped maps), and the decimation ratios. The index maps are built once, when the transform is created, and are kept across device moves, so nothing is recomputed on the first call after ``.to("cuda")``.

The integer metadata that determines the shape of the coefficients (:meth:`~curvelets.torch.UDCT.struct` and ``coefficient_shapes``) is kept in a private copy on the CPU, which has two consequences:

* shapes are computed without touching the device, and
* the backward pass of :class:`~curvelets.torch.UDCTModule`, which calls :meth:`~curvelets.torch.UDCT.struct`, works on CUDA devices.

The public ``decimation_ratios`` attribute still follows ``.to()``.

.. _torch-batching:

Batching
========

Every transform accepts leading batch dimensions. For a transform of shape ``shape``:

* :meth:`~curvelets.torch.UDCT.forward` maps ``(*batch, *shape)`` to coefficients whose arrays have shape ``(*batch, *wedge_shape)``;
* :meth:`~curvelets.torch.UDCT.vect` maps those to a tensor of shape ``(*batch, ncoef)`` and :meth:`~curvelets.torch.UDCT.struct` goes back;
* :meth:`~curvelets.torch.UDCT.backward` maps coefficients back to ``(*batch, *shape)``.

:class:`~curvelets.torch.UDCTModule` transforms a whole batch in one call, so :func:`torch.vmap` is no longer needed (it still works). The trailing dimensions are checked against the transform's shape.

.. code-block:: python

    import torch
    from curvelets.torch import UDCTModule

    module = UDCTModule(shape=(28, 28))
    images = torch.randn(64, 1, 28, 28)  # (batch, channel, height, width)
    coefficients = module(images)  # (64, 1, ncoef)

Filtering coefficients
----------------------

:meth:`UDCT.filter(image, fn) <curvelets.torch.UDCT.filter>` is a shortcut for the forward transform, ``fn`` applied to the coefficients, and the backward transform. ``fn`` receives a single tensor of shape ``(*batch, ncoef)`` in :meth:`~curvelets.torch.UDCT.vect` order (call :meth:`~curvelets.torch.UDCT.struct` inside it for per-wedge access), and the nested coefficient lists are never built.

.. code-block:: python

    from curvelets.torch import UDCT

    transform = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    noisy = torch.randn(4, 32, 32)
    denoised = transform.filter(noisy, lambda c: c * (c.abs() > 0.5))

See :ref:`sphx_glr_auto_examples_plot_10_udct_mnist.py` for batching in a training loop.

.. _torch-precision:

Precision
=========

A transform has a fixed precision, set when it is created with ``dtype`` (``torch.float32`` or ``torch.float64``; the default is :func:`torch.get_default_dtype`). A complex dtype names the precision of its parts: ``torch.complex64`` means ``torch.float32`` and ``torch.complex128`` means ``torch.float64``. The windows are computed in double precision and stored once in that dtype, so nothing is cast on each call.

.. code-block:: python

    import torch
    from curvelets.torch import UDCT, UDCTModule

    transform = UDCT(shape=(256, 256), dtype=torch.float64)
    module = UDCTModule(shape=(256, 256), dtype=torch.float64)

Inputs and coefficients of another precision are not cast silently: they raise a :class:`TypeError` that names the fix. To change the precision of a :class:`~curvelets.torch.UDCTModule` afterwards, use ``.to(dtype)``, ``.float()`` or ``.double()``. ``.to(torch.complex64)`` keeps the windows real and sets the matching precision. :meth:`~curvelets.torch.UDCT.struct` no longer takes its dtype from the last forward call; it uses the transform's.

.. note::

   **Breaking change.** Previously, float64 inputs to a default-constructed transform were accepted and cast. They now raise a :class:`TypeError`. Either pass ``dtype=torch.float64`` when creating the transform, call ``.double()`` on the module, or cast the input to the transform's precision.

.. _torch-performance:

Performance tips
================

* **Create the transform once and reuse it.** The first call on a device builds an execution plan (a few launches run the whole transform, whatever the number of wedges); later calls reuse it. Moving the transform with ``.to()`` rebuilds it for the new device.
* **Batch your inputs.** Pass a ``(batch, *shape)`` tensor rather than looping over images in Python (see `Batching`_).
* **Use float32 unless you need float64.** It is the default, and is much faster on most GPUs.
* **Avoid synchronizing in your own loop.** The transform itself never waits on the GPU; calls such as ``.item()`` or ``.cpu()`` between transforms do, and leave the GPU idle.
* ``torch.compile`` and :func:`torch.vmap` work with the transform.

Measured numbers, and the scripts that produce them, are in ``benchmarks/`` (see :ref:`contributing`).

.. _torch-cuda-graphs:

CUDA graphs
===========

For small and medium inputs on a GPU (up to about :math:`1024 \times 1024`), Python takes longer to launch the transform's kernels than the GPU takes to run them. Passing ``cuda_graphs=True`` records each transform once per input shape and replays it afterwards, which removes most of that overhead:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(512, 512), cuda_graphs=True).to("cuda")
    x = torch.randn(512, 512, device="cuda")
    y = transform.forward(x)  # recorded on the first call, replayed afterwards

It works with :class:`~curvelets.torch.UDCTModule`, :meth:`~curvelets.torch.UDCT.filter` and autograd, and has no effect on CPU tensors. Each recorded graph keeps some GPU memory; :meth:`~curvelets.torch.UDCT.clear_cuda_graphs` frees it.

Use it for small inputs (on an RTX 4090, a :math:`512 \times 512` forward goes from 0.37 to 0.065 ms). For large inputs it can be slightly slower, so measure on your own sizes.
