"""CUDA-graph replay of the fused transform (``UDCT(cuda_graphs=True)``).

A single transform launches a few dozen small kernels. For small and medium
inputs, launching them from Python takes longer than running them on the GPU,
so the cache captures each flat transform once per input shape as a CUDA
graph and replays it: one copy into a static input buffer, one graph launch,
and one copy out of the static output buffer.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch

from ._foldmap import _functorch_wrapped

_Key = tuple[str, tuple[int, ...], torch.dtype, torch.device]
_Entry = tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]

DEFAULT_MAX_GRAPHS = 8
_WARMUP_CALLS = 3
# Device types whose calls are replayed (the tests add "cpu", with an eager
# stand-in for the capture, to cover the dispatch and cache logic anywhere).
GRAPH_DEVICE_TYPES = frozenset({"cuda"})


def _replay_possible(tensor: torch.Tensor) -> bool:
    """A graph device, and not in a functorch transform, a trace, or a capture."""
    return (
        tensor.device.type in GRAPH_DEVICE_TYPES
        and not torch.compiler.is_compiling()
        and not _functorch_wrapped(tensor)
        and not (tensor.is_cuda and torch.cuda.is_current_stream_capturing())
    )


def _needs_grad(tensor: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and tensor.requires_grad


def graphable(tensor: torch.Tensor) -> bool:
    """Whether a call on ``tensor`` can be replayed from a graph as it is.

    Graphs record no autograd history, so the call must not need gradients.
    """
    return _replay_possible(tensor) and not _needs_grad(tensor)


def graphable_with_adjoint(tensor: torch.Tensor) -> bool:
    """Whether a call that needs gradients can still use graphs.

    It goes through an autograd Function whose forward and backward (the
    adjoint transform) both run without gradients, and so from graphs.
    """
    return _replay_possible(tensor) and _needs_grad(tensor)


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
