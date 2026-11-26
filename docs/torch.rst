.. _torch-backend:

PyTorch backend
###############

The :mod:`curvelets.torch` package provides the same transform as :mod:`curvelets.numpy` on PyTorch tensors, on the CPU or on a GPU, with autograd support. Install it with ``pip install curvelets[torch]``.

:class:`~curvelets.torch.UDCT` is the transform, and it is a :class:`torch.nn.Module` with no learnable parameters:

* ``udct(x)`` (:meth:`~curvelets.torch.UDCT.forward`) returns the nested coefficient structure and :meth:`~curvelets.torch.UDCT.backward` inverts it, as in :mod:`curvelets.numpy`; ``udct(x, structured=False)`` returns the flat tensor directly, and ``backward`` accepts either layout; ``vect`` flattens the nested structure to a single tensor and ``struct`` does the opposite;
* ``forward(x, structured=False)`` returns the flat coefficient tensor directly, and :meth:`~curvelets.torch.UDCT.backward` accepts it, which is the convenient layout inside a network.

.. _torch-gradients:

Gradients
=========

.. versionchanged:: 1.3
   :class:`~curvelets.torch.UDCT` takes its gradients from the adjoint transform in both layouts. Before, only :class:`~curvelets.torch.UDCTModule` did; :class:`~curvelets.torch.UDCT` recorded autograd through the whole transform.

Every transform is differentiable, in both layouts. The UDCT is a tight frame, so the backward transform is the adjoint of the forward transform: the gradient of the forward transform is the backward transform of the incoming gradient, and vice versa. :class:`~curvelets.torch.UDCT` uses exactly that (through a custom :class:`torch.autograd.Function`) instead of recording every FFT, gather and fold in the autograd graph, which keeps gradients fast and memory-light. Second derivatives (``create_graph=True``), :func:`torch.vmap`, :mod:`torch.func` and :func:`torch.compile` work.

.. note::

   :class:`~curvelets.torch.UDCTModule` is deprecated. Instead of ``UDCTModule(shape, ..., transform_type=kind)(x)``, use ``UDCT(shape, ..., transform_kind=kind).forward(x, structured=False)``.

Moving to a GPU
===============

A :class:`~curvelets.torch.UDCT` moves like any other module, alone or as a submodule of a network:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    udct = UDCT(shape=(512, 512)).to("cuda")
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    y = udct.forward(x, structured=False)
    y.abs().sum().backward()  # gradients flow through the transform

The windows are not registered as buffers, so ``state_dict()`` stays empty: they are rebuilt from the constructor arguments rather than stored in checkpoints.

Moving a transform with ``.to(device)`` moves everything it holds: the windows, their precomputed periodized index maps (and, for the complex transform, the flipped maps), and the decimation ratios. The index maps are built once, when the transform is created, and are kept across device moves, so nothing is recomputed on the first call after ``.to("cuda")``.

The integer metadata that determines the shape of the coefficients (:meth:`~curvelets.torch.UDCT.struct` and ``coefficient_shapes``) is kept in a private copy on the CPU, which has two consequences:

* shapes are computed without touching the device, and
* gradients through the nested layout, which call :meth:`~curvelets.torch.UDCT.struct`, work on CUDA devices.

The public ``decimation_ratios`` attribute still follows ``.to()``.

.. _torch-batching:

Batching
========

.. versionadded:: 1.3

Every transform accepts leading batch dimensions. For a transform of shape ``shape``:

* :meth:`~curvelets.torch.UDCT.forward` maps ``(*batch, *shape)`` to coefficients whose arrays have shape ``(*batch, *wedge_shape)``;
* :meth:`~curvelets.torch.UDCT.vect` maps those to a tensor of shape ``(*batch, ncoef)`` and :meth:`~curvelets.torch.UDCT.struct` goes back;
* :meth:`~curvelets.torch.UDCT.backward` maps coefficients back to ``(*batch, *shape)``;
* ``forward(x, structured=False)`` and :meth:`~curvelets.torch.UDCT.backward` (given the flat tensor) map ``(*batch, *shape)`` to ``(*batch, ncoef)`` and back.

A whole batch is transformed in one call (:func:`torch.vmap` also works). The trailing dimensions are checked against the transform's shape.

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    udct = UDCT(shape=(28, 28))
    images = torch.randn(64, 1, 28, 28)  # (batch, channel, height, width)
    coefficients = udct.forward(images, structured=False)  # (64, 1, ncoef)

Filtering coefficients
----------------------

.. versionadded:: 1.3

:meth:`UDCT.filter(image, fn) <curvelets.torch.UDCT.filter>` is a shortcut for the forward transform, ``fn`` applied to the coefficients, and the backward transform. ``fn`` receives a single tensor of shape ``(*batch, ncoef)`` in :meth:`~curvelets.torch.UDCT.vect` order (call :meth:`~curvelets.torch.UDCT.struct` inside it for per-wedge access), and the nested coefficient lists are never built.

.. code-block:: python

    from curvelets.torch import UDCT

    transform = UDCT(shape=(32, 32), num_scales=3, wedges_per_direction=3)
    noisy = torch.randn(4, 32, 32)
    denoised = transform.filter(noisy, lambda c: c * (c.abs() > 0.5))

See :ref:`sphx_glr_auto_examples_plot_10_udct_mnist.py` for batching in a training loop.

.. _torch-precision:

Monogenic transform
===================

``transform_kind="monogenic"`` gives the same coefficients as :mod:`curvelets.numpy`: each wedge is a real tensor of shape ``(*batch, *wedge_shape, ndim + 2)`` holding the scalar coefficient (real and imaginary parts) and one Riesz component per dimension. :meth:`~curvelets.torch.UDCT.backward` (from nested or flat coefficients) returns the ``ndim + 1`` components that numpy returns as a tuple, stacked along a new first axis, ``(ndim + 1, *batch, *shape)``, with the reconstruction of ``f`` first (``f, *riesz = udct.backward(coeffs)``). :meth:`~curvelets.torch.UDCT.monogenic` computes :math:`(f, -R_1 f, \ldots, -R_n f)` directly, in the same layout.

The transform runs on the real transform's fused plan: one batched call transforms ``f`` together with its Riesz-filtered copies. It therefore has the same gradients (through the adjoint), batching and CUDA graphs as the real transform, at about ``ndim + 1`` times its cost (up to ``2 ndim + 1`` when axes have even lengths, where the Riesz-filtered copies are complex).

Precision
=========

.. versionchanged:: 1.3
   Transforms have a fixed precision; inputs of another precision raise instead of being cast.

A transform has a fixed precision, set when it is created with ``dtype`` (``torch.float32`` or ``torch.float64``; the default is :func:`torch.get_default_dtype`). A complex dtype names the precision of its parts: ``torch.complex64`` means ``torch.float32`` and ``torch.complex128`` means ``torch.float64``. The windows are computed in double precision and stored once in that dtype, so nothing is cast on each call.

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(256, 256), dtype=torch.float64)

Inputs and coefficients of another precision are not cast silently: they raise a :class:`TypeError` that names the fix. To change the precision of a :class:`~curvelets.torch.UDCT` afterwards, use ``.to(dtype)``, ``.float()`` or ``.double()``. ``.to(torch.complex64)`` keeps the windows real and sets the matching precision. :meth:`~curvelets.torch.UDCT.struct` returns views in the dtype of the vector it is given.

.. _torch-performance:

Performance tips
================

* **Create the transform once and reuse it.** Building the windows takes up to a few seconds for large inputs; across processes, save the built transform (see `Saving a built transform`_). The first call on a device builds an execution plan (a few launches run the whole transform, whatever the number of wedges); later calls reuse it. Moving the transform with ``.to()`` rebuilds it for the new device.
* **Batch your inputs.** Pass a ``(batch, *shape)`` tensor rather than looping over images in Python (see `Batching`_).
* **Use float32 unless you need float64.** It is the default, and is much faster on most GPUs.
* **Avoid synchronizing in your own loop.** The transform itself never waits on the GPU; calls such as ``.item()`` or ``.cpu()`` between transforms do, and leave the GPU idle.
* ``torch.compile`` and :func:`torch.vmap` work with the transform.

Saving a built transform
------------------------

Creating a transform builds its windows on the CPU, with the same code as :mod:`curvelets.numpy`: on a 4-core laptop CPU, about 2 s for a :math:`2048 \times 2048` transform with 5 scales and 0.5–1 s for :math:`128 \times 128 \times 128` with 3 scales. Moving the transform to a GPU does not rebuild them. If you create the same transform often (in every run of a script, or in every data-loader worker), build it once and save it with :func:`torch.save`. A saved transform keeps everything it precomputed, and loads in about 0.1–0.2 s at :math:`2048 \times 2048`:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(2048, 2048), num_scales=5)
    torch.save(transform, "udct_2048.pt")

    # later, or in another process
    transform = torch.load("udct_2048.pt", weights_only=False)

``pickle`` works too. The file is large (about 100 MB at :math:`2048 \times 2048`). Loading it runs code from the file, so only load files you created. It is tied to the version of ``curvelets`` that wrote it, so rebuild it after upgrading. The same applies to :class:`curvelets.numpy.UDCT` (see :ref:`cpu-performance`).

Measured numbers, and the scripts that produce them, are in ``benchmarks/`` (see :ref:`contributing`).

.. _torch-cuda-graphs:

CUDA graphs
===========

.. versionadded:: 1.3

For small and medium inputs on a GPU (up to about :math:`1024 \times 1024`), Python takes longer to launch the transform's kernels than the GPU takes to run them. Passing ``cuda_graphs=True`` records each transform once per input shape and replays it afterwards, which removes most of that overhead:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(512, 512), cuda_graphs=True).to("cuda")
    x = torch.randn(512, 512, device="cuda")
    y = transform.forward(x)  # recorded on the first call, replayed afterwards

It works with every forward and backward entry point, :meth:`~curvelets.torch.UDCT.filter` and autograd, and has no effect on CPU tensors. Each recorded graph keeps some GPU memory; :meth:`~curvelets.torch.UDCT.clear_cuda_graphs` frees it.

Use it for small inputs (on an RTX 4090, a :math:`512 \times 512` forward goes from 0.37 to 0.065 ms). For large inputs it can be slightly slower, so measure on your own sizes.

.. _torch-cpp:

C++ (libtorch) runtime
======================

.. versionadded:: 1.4

Models using :class:`~curvelets.torch.UDCT` can be run and trained inside a Python-free C++ application via ``libtorch`` on Linux (GCC/Clang), macOS (AppleClang), and Windows (MSVC). The C++ ``curvelets::UDCT`` module supports N-D real, complex, and monogenic transforms, ``.to(device)`` / ``.to(dtype)``, and 1st- and 2nd-order adjoint autograd.

Export a plan once from Python with :meth:`~curvelets.torch.UDCT.save_plan`:

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    udct = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3, dtype=torch.float32)
    udct.save_plan("udct_plan.pt")

Then build ``cpp/`` from source with CMake (requires CMake >= 3.18, a C++20 compiler, and ``libtorch``, either standalone or from a PyTorch installation) — or pull it into an existing CMake project via ``FetchContent`` (link ``curvelets::torch``):

.. code-block:: cmake

    include(FetchContent)
    FetchContent_Declare(
        curvelets
        GIT_REPOSITORY https://github.com/cako/curvelets.git
        GIT_TAG main
        SOURCE_SUBDIR cpp
    )
    FetchContent_MakeAvailable(curvelets)

    target_link_libraries(my_app PRIVATE curvelets::torch)

Load and use the exported plan in C++:

.. code-block:: cpp

    #include "curvelets/udct.hpp"

    curvelets::UDCT udct("udct_plan.pt");
    torch::Tensor x = torch::randn({4, 64, 64}, torch::requires_grad());
    torch::Tensor coeffs = udct->forward(x);    // flat (*batch, ncoef) layout
    torch::Tensor recon = udct->backward(coeffs);  // (*batch, *shape) or (ndim + 1, *batch, *shape) for monogenic
    recon.sum().backward();

``udct->forward(x)`` returns the flat ``(*batch, ncoef)`` coefficient layout (matching ``forward(x, structured=False)`` in Python), and ``udct->backward(coeffs)`` returns ``(*batch, *shape)`` (or ``(ndim + 1, *batch, *shape)`` for monogenic transforms).

See `cpp/README.md <https://github.com/cako/curvelets/blob/main/cpp/README.md>`_ and `cpp/examples/train_model.cpp <https://github.com/cako/curvelets/blob/main/cpp/examples/train_model.cpp>`_ for CMake build instructions and a full training example.

**Current limitations:**

* Plans are generated once in Python per transform configuration (``shape``, ``num_scales`` / ``angular_wedges_config``, ``high_frequency_mode``, ``transform_kind``, etc.) and loaded from ``.pt`` files in C++.
* C++ consumers build ``cpp/`` from source with CMake (no prebuilt C++ binaries are distributed yet).
