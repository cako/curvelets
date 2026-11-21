"""CUDA-graph replay of the fused transform (``UDCT(cuda_graphs=True)``).

A single transform launches a few dozen small kernels. For small and medium
inputs, launching them from Python takes longer than running them on the GPU,
so the cache captures each flat transform once per input shape as a CUDA
graph and replays it: one copy into a static input buffer, one graph launch,
and one copy out of the static output buffer.

Gradients always come from the adjoint autograd Functions
(:mod:`._autograd`), whose forward and backward are both plain calls without
gradients, so both replay. Under ``torch.vmap`` or ``torch.compile``, inside another stream
capture, and for CPU tensors, calls run as usual.

On an RTX 4090 a single 512² forward drops from 0.37 to 0.065 ms and the
backward from 0.55 to 0.087 ms. For large inputs the GPU work dominates and
the copies make replay slower (6.7 vs 5.5 ms for a 2048² forward, batch 8).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch

from ._autograd import BackwardWithAdjoint, ForwardWithAdjoint
from ._foldmap import _functorch_wrapped

_Key = tuple[str, tuple[int, ...], torch.dtype, torch.device]
_Entry = tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]

DEFAULT_MAX_GRAPHS = 8
_WARMUP_CALLS = 3
# Device types whose calls are replayed (the tests add "cpu", with an eager
# stand-in for the capture, to cover the dispatch and cache logic anywhere).
GRAPH_DEVICE_TYPES = frozenset({"cuda"})


def _replay_possible(tensor: torch.Tensor) -> bool:
    """A graph device, and not in a functorch transform, a trace, or a capture.

    Graphs record no autograd history; callers check this only for calls that
    cannot need gradients (see :func:`_may_need_grad`).
    """
    return (
        tensor.device.type in GRAPH_DEVICE_TYPES
        and not torch.compiler.is_compiling()
        and not _functorch_wrapped(tensor)
        and not (tensor.is_cuda and torch.cuda.is_current_stream_capturing())
    )


def _may_need_grad(tensor: torch.Tensor) -> bool:
    """Whether a call must go through the adjoint Functions.

    Inside ``torch.func`` transforms ``requires_grad`` does not tell, so those
    go through them too (the Functions have vmap rules).
    """
    return bool(
        torch.is_grad_enabled() and (tensor.requires_grad or _functorch_wrapped(tensor))
    )


def _capture(  # pragma: no cover - needs a CUDA device (tests/torch/test_cuda_graphs.py)
    fn: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor
) -> _Entry:
    """Warm ``fn`` up on a side stream (cuFFT plans), then capture one call."""
    static_input = example.detach().clone(memory_format=torch.contiguous_format)
    device = static_input.device
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(_WARMUP_CALLS):
            fn(static_input)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn(static_input)
    return static_input, graph, static_output


class CudaGraphCache:
    """Captured transforms keyed by direction, input shape, dtype and device.

    The least recently used graph is dropped beyond ``max_entries``; each
    graph keeps a private memory pool of about one call's intermediates.
    Copies (``copy.deepcopy``, pickling) start empty.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_GRAPHS) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __getstate__(self) -> dict[str, Any]:
        return {"max_entries": self.max_entries}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["max_entries"])  # type: ignore[misc]  # pylint: disable=unnecessary-dunder-call

    def clear(self) -> None:
        """Drop every captured graph (and its memory pool)."""
        self._entries.clear()

    def run(
        self,
        name: str,
        fn: Callable[[torch.Tensor], torch.Tensor],
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the graph of ``fn`` for ``tensor``'s shape, capturing it first.

        Returns a copy of the static output, so results survive later calls.
        """
        key: _Key = (name, tuple(tensor.shape), tensor.dtype, tensor.device)
        entry = self._entries.get(key)
        if entry is None:
            with torch.no_grad():
                entry = _capture(fn, tensor)
            self._entries[key] = entry
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        static_input, graph, static_output = entry
        static_input.copy_(tensor)
        graph.replay()
        return static_output.clone()


class CudaGraphsMixin:
    """The ``cuda_graphs`` switch of :class:`~curvelets.torch.UDCT`.

    Expects ``_cuda_graphs`` (bool), ``_graphs`` (:class:`CudaGraphCache`) and
    ``_get_plan()`` on the class it is mixed into.
    """

    _cuda_graphs: bool
    _graphs: CudaGraphCache

    def _get_plan(self) -> Any:  # pragma: no cover - provided by UDCT
        raise NotImplementedError

    @property
    def cuda_graphs(self) -> bool:
        """Whether CUDA calls replay CUDA graphs (see the class docstring).

        .. versionadded:: 1.3
        """
        return self._cuda_graphs

    @cuda_graphs.setter
    def cuda_graphs(self, enabled: bool) -> None:
        self._cuda_graphs = enabled
        if not enabled:
            self._graphs.clear()

    @property
    def cuda_graph_cache_size(self) -> int:
        """Number of captured CUDA graphs kept (least recently used dropped).

        .. versionadded:: 1.3
        """
        return self._graphs.max_entries

    @cuda_graph_cache_size.setter
    def cuda_graph_cache_size(self, size: int) -> None:
        self._graphs.max_entries = size
        self._graphs.clear()

    def clear_cuda_graphs(self) -> None:
        """Drop all captured CUDA graphs and their memory.

        .. versionadded:: 1.3
        """
        self._graphs.clear()

    def _forward_flat(self, image: torch.Tensor) -> torch.Tensor:
        """Flat forward of checked input; its gradient is the backward transform."""
        if _may_need_grad(image):
            # Runs _forward_flat again without gradients, so from graphs too.
            return ForwardWithAdjoint.apply(image, self)
        forward: Callable[[torch.Tensor], torch.Tensor] = self._get_plan().forward_flat
        if self._cuda_graphs and _replay_possible(image):
            return self._graphs.run("forward", forward, image)
        return forward(image)

    def _backward_flat(self, vector: torch.Tensor) -> torch.Tensor:
        """Flat backward of checked input; its gradient is the forward transform."""
        if _may_need_grad(vector):
            return BackwardWithAdjoint.apply(vector, self)
        backward: Callable[[torch.Tensor], torch.Tensor] = (
            self._get_plan().backward_flat
        )
        if self._cuda_graphs and _replay_possible(vector):
            return self._graphs.run("backward", backward, vector)
        return backward(vector)
