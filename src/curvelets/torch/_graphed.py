"""CUDA-graph replay of the fused UDCT for one fixed input shape."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch

from ._dtypes import check_dtype
from ._udct import UDCT

_Fn = Callable[[torch.Tensor], torch.Tensor]


def _capture(
    fn: _Fn, static_input: torch.Tensor, warmup: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Warm ``fn`` up on a side stream, then capture one call into a CUDA graph."""
    side = torch.cuda.Stream(device=static_input.device)
    side.wait_stream(torch.cuda.current_stream(static_input.device))
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn(static_input)
    torch.cuda.current_stream(static_input.device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn(static_input)
    return graph, static_output


class GraphedUDCT:
    """
    Forward and backward UDCT replayed as CUDA graphs, for one input shape.

    A single transform launches a few dozen small kernels, and for inputs up to
    about 1024² (and the backward transform at larger sizes) launching them
    from Python takes longer than running them on the GPU. This class captures
    the flat forward and backward transforms once, as CUDA graphs, so each
    call is one copy into a static input buffer plus one graph launch. On an
    RTX 4090 that makes a 512² transform 6-12x faster.

    Parameters
    ----------
    transform : UDCT
        A real or complex transform whose tensors are on a CUDA device.
    batch_shape : tuple[int, ...], optional
        Leading batch dimensions of every input. Default is ``()``, a single
        image of shape ``transform.shape``.
    compile : bool, optional
        Compile the transforms with ``torch.compile`` before capturing, so the
        folds run as fused Triton kernels (needs Triton). Default is False.
    copy_outputs : bool, optional
        Return copies of the graphs' static output buffers (default). With
        False, the returned tensors are the buffers themselves, which the next
        call of the same method overwrites.
    warmup : int, optional
        Calls before capture, which build cuFFT plans and compiled kernels.
        Default is 3.

    Notes
    -----
    The graphs are inference-only (no autograd) and keep their own memory
    pools, about the size of one forward and one backward call's
    intermediates. Inputs must match the captured shape exactly.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCT, GraphedUDCT
    >>> transform = UDCT(shape=(512, 512), num_scales=3, wedges_per_direction=3)
    >>> transform.apply_to_tensors(lambda t: t.to("cuda"))  # doctest: +SKIP
    >>> graphed = GraphedUDCT(transform, batch_shape=(8,))  # doctest: +SKIP
    >>> images = torch.randn(8, 512, 512, device="cuda")  # doctest: +SKIP
    >>> coefficients = graphed.forward(images)  # (8, ncoef), vect order  # doctest: +SKIP
    >>> reconstruction = graphed.backward(coefficients)  # doctest: +SKIP
    """

    def __init__(
        self,
        transform: UDCT,
        batch_shape: tuple[int, ...] = (),
        *,
        compile: bool = False,  # pylint: disable=redefined-builtin
        copy_outputs: bool = True,
        warmup: int = 3,
    ) -> None:
        kind = transform._transform_kind  # pylint: disable=protected-access
        if kind == "monogenic":
            msg = "GraphedUDCT supports the real and complex transforms only."
            raise NotImplementedError(msg)
        device = transform.windows[0][0][0].values.device
        if device.type != "cuda":
            msg = f"GraphedUDCT needs a transform on a CUDA device, got {device}."
            raise ValueError(msg)
        self._transform = transform
        self._batch_shape = tuple(batch_shape)
        self._copy_outputs = copy_outputs
        plan = transform._get_plan()  # pylint: disable=protected-access
        forward: _Fn = plan.forward_flat
        backward: _Fn = plan.backward_flat
        if compile:
            forward = torch.compile(forward)
            backward = torch.compile(backward)
        input_dtype = (
            transform.coefficient_dtype if kind == "complex" else transform.dtype
        )
        image = torch.zeros(
            (*self._batch_shape, *transform.shape), dtype=input_dtype, device=device
        )
        vector = torch.zeros(
            (*self._batch_shape, plan.ncoef),
            dtype=transform.coefficient_dtype,
            device=device,
        )
        with torch.no_grad(), warnings.catch_warnings():
            # Compiling warns that the complex FFT calls are not code-generated;
            # they stay cuFFT calls by design (the folds are real arithmetic).
            warnings.filterwarnings(
                "ignore", message=".*code generation for complex operators.*"
            )
            self._forward_graph, self._forward_out = _capture(forward, image, warmup)
            self._backward_graph, self._backward_out = _capture(
                backward, vector, warmup
            )
        self._image = image
        self._vector = vector

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of the images :meth:`forward` accepts."""
        return tuple(self._image.shape)

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        """Shape of the flat coefficients: ``(*batch_shape, ncoef)``."""
        return tuple(self._vector.shape)

    @property
    def ncoef(self) -> int:
        """Number of coefficients per image."""
        return int(math.prod(self._vector.shape[len(self._batch_shape) :]))

    def _result(self, static_output: torch.Tensor) -> torch.Tensor:
        return static_output.clone() if self._copy_outputs else static_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward transform to flat coefficients (the layout of :meth:`UDCT.vect`).

        Parameters
        ----------
        image : torch.Tensor
            Input of shape :attr:`input_shape`, real for the real transform.

        Returns
        -------
        torch.Tensor
            Coefficients of shape :attr:`coefficient_shape`. Use
            :meth:`UDCT.struct` for the nested layout.
        """
        if tuple(image.shape) != self.input_shape:
            msg = (
                f"Expected input of shape {self.input_shape}, got {tuple(image.shape)}."
            )
            raise ValueError(msg)
        self._transform._check_input(image)  # pylint: disable=protected-access
        self._image.copy_(image)
        self._forward_graph.replay()
        return self._result(self._forward_out)

    def backward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Backward transform of flat coefficients.

        Parameters
        ----------
        coefficients : torch.Tensor
            Coefficients of shape :attr:`coefficient_shape`.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape :attr:`input_shape`.
        """
        if tuple(coefficients.shape) != self.coefficient_shape:
            msg = (
                f"Expected coefficients of shape {self.coefficient_shape}, "
                f"got {tuple(coefficients.shape)}."
            )
            raise ValueError(msg)
        check_dtype(coefficients, self._transform.dtype, "coefficients")
        self._vector.copy_(coefficients)
        self._backward_graph.replay()
        return self._result(self._backward_out)
