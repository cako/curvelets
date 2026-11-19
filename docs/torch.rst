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

Performance
===========

The transform is designed to keep a GPU busy instead of waiting on the Python process. This section describes what the implementation does; measured numbers and the hardware they come from are in ``benchmarks/GPU_PERFORMANCE.md`` (see :ref:`contributing` for how to reproduce them).

No device-to-host synchronizations
----------------------------------

A blocking copy from the GPU to the CPU (``.item()``, ``.tolist()``, ``.cpu()`` on a CUDA tensor) stalls the host until all queued kernels finish, so the GPU then sits idle while Python enqueues the next ones. The forward and backward transforms issue none:

* the per-wedge scale factors are computed from a CPU copy of the decimation ratios, and
* the normalization of the lowpass band is computed with plain Python ``math``, not with a device tensor.

You can verify this on your own setup. The call below raises if anything in the transform synchronizes with the device:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(512, 512)).to("cuda")
    x = torch.randn(512, 512, device="cuda")
    transform.forward(x)  # warm up
    torch.cuda.set_sync_debug_mode("error")
    transform.forward(x)
    torch.cuda.set_sync_debug_mode("default")

Fused execution plan
--------------------

Applying the transform wedge by wedge launches a handful of tiny kernels per wedge, so the cost grows with the number of wedges. :class:`~curvelets.torch.UDCT` instead builds, lazily and on first use, an execution plan that runs the whole transform in a fixed number of steps:

* **forward**: one full-size FFT, one gather of every window sample, one fold into a flat buffer in :meth:`~curvelets.torch.UDCT.vect` order, and one batched inverse FFT per run of consecutive wedges with the same shape;
* **backward**: batched FFTs per group, one gather, one scatter and one inverse FFT.

The number of kernel launches therefore no longer depends on the number of wedges (for a :math:`2048 \times 2048` input with 181 wedges, a forward went from about 1450 launches to 38). All operations are out-of-place and differentiable, so autograd and :func:`torch.vmap` keep working, and the plan is rebuilt when the transform is moved to another device.

:class:`~curvelets.torch.UDCTModule` goes straight to and from the flat :meth:`~curvelets.torch.UDCT.vect` layout. The straightforward per-wedge functions are kept as the reference implementation that the plan is tested against (real and complex transforms, curvelet and wavelet modes, 2D and 3D, non-square shapes), so the public API and the numerical results are unchanged.

Device-specific folds
---------------------

The *fold* step turns window samples into the coefficient spectra (and back). It has one implementation per device class, chosen from the device of the transform:

* **GPUs and other accelerators** use a gather-only fold: deterministic (no atomics, so results are bitwise reproducible), with few launches, followed by batched FFTs.
* **CPUs** use sorted scatter-adds applied row by row, which avoids the redundant arithmetic a gather would cost there. The forward folds all wedges at once. The backward is cache-blocked: chunks of about 2\ :sup:`19` coefficients are transformed and folded into the spectrum while they are still in cache, so the buffer holding all coefficient spectra never exists.

For the real transform the FFT is a half-spectrum transform (``rfftn``), and the backward reconstructs the result with ``irfftn``, which halves the FFT work and memory. Indices use 32 bits, and the per-wedge scale factors are folded into the fold weights.

Moving a module between devices with ``.to()`` rebuilds the plan with the fold that suits the new device; you do not select it yourself.
