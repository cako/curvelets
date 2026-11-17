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
