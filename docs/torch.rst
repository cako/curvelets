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

Building a transform
--------------------

Creating a transform builds its windows, once, on the CPU. :mod:`curvelets.torch` uses the same window builder as :mod:`curvelets.numpy` and converts the result to tensors, so both backends build identical windows at the same cost. The builder evaluates each window only where it can be nonzero and spreads the work over the CPU cores with threads. On a 4-core laptop CPU (Intel i7-8565U), building a :math:`2048 \times 2048` transform with 5 scales takes about 2 s, and a :math:`128 \times 128 \times 128` transform with 3 scales about 0.5 s. Moving the transform to a GPU afterwards does not rebuild anything.

If you create the same transform repeatedly, for example in every run of a script or in every worker of a data loader, build it once and save it. Transforms pickle with everything they precompute, and loading one takes a fraction of the time it takes to build it (about 0.1–0.2 s instead of 2 s for the :math:`2048 \times 2048` transform above):

.. code-block:: python

    import pickle

    from curvelets.numpy import UDCT

    transform = UDCT(shape=(2048, 2048), num_scales=5)
    with open("udct_2048.pkl", "wb") as f:
        pickle.dump(transform, f)

    # later, or in another process
    with open("udct_2048.pkl", "rb") as f:
        transform = pickle.load(f)

For :mod:`curvelets.torch`, :func:`torch.save` and :func:`torch.load` (with ``weights_only=False``) work the same way for a :class:`~curvelets.torch.UDCT` and a :class:`~curvelets.torch.UDCTModule`; ``pickle`` works too.

Keep in mind that:

* the file holds the windows and their index maps, so it is large: about 110–130 MB for the :math:`2048 \times 2048` transform above;
* unpickling runs code from the file, so only load files you created yourself;
* a saved transform is tied to the version of ``curvelets`` that created it, so rebuild it after upgrading.

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

Real-arithmetic folds and ``torch.compile``
-------------------------------------------

On accelerators, the gather fold works on the real view of complex data (:func:`torch.view_as_real`) with a pair of weights per contribution, applied to the real and imaginary parts. A term can therefore enter conjugated without materializing a conjugate. For the real transform this means:

* the forward gathers straight from the ``rfftn`` half spectrum, so the ``[F, conj F]`` buffer, as large as a full complex spectrum, is gone;
* the plain and conjugated backward folds are merged into one, removing the extra ``A + conj(B)`` pass;
* the ``1/size`` normalization of each wedge's inverse FFT is folded into the forward weights, and the inverse FFT runs with ``norm="forward"``.

Every step is a real gather, multiply or sum. This avoids the slow complex-by-real multiply kernel and lets TorchInductor fuse the fold into Triton kernels. The accelerator path of the fused plan's flat forward and backward transforms traces into a single graph, without graph breaks (checked in the test suite for the real and complex transforms). The FFTs themselves remain cuFFT calls. CPU scatter folds are unchanged.

Batches that exceed the L2 cache
--------------------------------

When the working set of a batch spills out of the GPU's L2 cache, every stage of the transform runs at DRAM bandwidth. For inputs whose single row is already larger than the L2 budget (for example :math:`2048 \times 2048` in 2D or :math:`128 \times 128 \times 128` in 3D on an RTX 4090, with 72 MB of L2), running the rows one at a time is 30–44% faster than one batch of 8, because each row has enough GPU work to hide its kernel launches.

On accelerators with a known L2 size, the execution plan records a working-set budget (``UDCTPlan.batch_chunk_bytes``, set to the L2 size when the plan is built; ``None`` on CPUs and on devices with an unknown L2 size). When one row's working set (one spectrum and two coefficient vectors) is larger than the budget, the forward and backward transforms run the batch row by row, writing every row into one preallocated output. Under autograd, :func:`torch.vmap` or ``torch.compile`` the pieces are concatenated instead. Every other batch runs whole: splitting batches whose rows fit in L2 was measured to be slower, since each extra piece costs a full set of kernel launches.

This is automatic and does not change the results.

.. _torch-cuda-graphs:

CUDA graphs
===========

For single transforms up to about :math:`1024 \times 1024`, and for the backward at larger sizes, the transform is *launch-bound*: Python takes longer to enqueue a few dozen kernels than the GPU takes to run them. Passing ``cuda_graphs=True`` to :class:`~curvelets.torch.UDCT` or :class:`~curvelets.torch.UDCTModule` captures each flat transform once per input shape and dtype as a CUDA graph, and replays it afterwards: one copy into a static input buffer, one graph launch, and one copy out of the static output buffer.

.. code-block:: python

    import torch
    from curvelets.torch import UDCT

    transform = UDCT(shape=(512, 512), cuda_graphs=True).to("cuda")
    x = torch.randn(512, 512, device="cuda")
    y = transform.forward(x)  # captured on the first call, replayed afterwards

Details:

* It applies to every forward and backward entry point, to :meth:`~curvelets.torch.UDCT.filter` and to :class:`~curvelets.torch.UDCTModule`. The ``cuda_graphs`` attribute can be switched at any time.
* Calls that need gradients work too: the gradient of either transform is the other one (the transform is a tight frame), so both the forward and the backward pass replay.
* Under :func:`torch.vmap` or ``torch.compile``, inside another stream capture, and for CPU tensors, calls run as usual.
* The most recently used ``cuda_graph_cache_size`` graphs are kept (default 8; each graph holds a private memory pool of about one call's intermediates). :meth:`~curvelets.torch.UDCT.clear_cuda_graphs` drops all of them and frees that memory. Copies and pickles of a transform start with an empty cache.

When to use it
--------------

Use it for small and medium inputs, where launches dominate: on an RTX 4090, a single :math:`512 \times 512` forward drops from 0.37 to 0.065 ms and the backward from 0.55 to 0.087 ms. For large inputs the GPU work dominates and the copies in and out of the graph buffers make replay slower (6.7 ms against 5.5 ms for a :math:`2048 \times 2048` forward with a batch of 8). Measure on your own sizes with ``benchmarks/compile_experiments.py``.
