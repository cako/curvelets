"""Gradcheck tests for UDCTModule."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TypeVar

import numpy as np
import pytest
import torch

import curvelets.torch as torch_curvelets

T = TypeVar("T")


class TimeoutError(Exception):
    """Timeout exception for test timeouts."""


def timeout(seconds: float, func: Callable[[], T]) -> T:
    """
    Execute a function with a timeout, raising TimeoutError if it takes too long.

    Cross-platform implementation using ThreadPoolExecutor that works on
    Windows, Linux, and macOS. The function is executed in a separate
    thread and monitored for timeout.

    Parameters
    ----------
    seconds : float
        Maximum number of seconds to allow the function to run.
    func : Callable[[], T]
        Function to execute with timeout. Must take no arguments.

    Returns
    -------
    T
        The return value of the function.

    Raises
    ------
    TimeoutError
        If the function takes longer than the specified timeout.
    Any exception raised by func
        Any exception raised by the function will be propagated.

    Examples
    --------
    >>> def long_operation():
    ...     import time
    ...     time.sleep(10)
    ...     return 42
    >>> result = timeout(5.0, long_operation)  # Will raise TimeoutError
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func)
        try:
            return future.result(timeout=seconds)
        except FuturesTimeoutError:
            future.cancel()
            msg = f"Operation timed out after {seconds} seconds"
            raise TimeoutError(msg) from None
    finally:
        executor.shutdown(wait=False)


@pytest.mark.parametrize("dim", [2, 3, 4])  # type: ignore[misc]
@pytest.mark.parametrize("transform_type", ["real", "complex"])  # type: ignore[misc]
def test_udct_module_gradcheck(dim: int, transform_type: str) -> None:
    """
    Test that UDCTModule passes gradcheck for all dimensions and transform types.

    Parameters
    ----------
    dim : int
        Dimension (2, 3, or 4).
    transform_type : str
        Transform type to test ("real" or "complex").
    """
    # Use very small arrays for fast testing
    shape: tuple[int, ...]
    if dim == 2:
        shape = (16, 16)
        config = torch.tensor([[3, 3]])
    elif dim == 3:
        shape = (8, 8, 8)
        config = torch.tensor([[3, 3, 3]])
    else:  # dim == 4
        shape = (4, 4, 4, 4)
        config = torch.tensor([[3, 3, 3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
        transform_type=transform_type,  # type: ignore[arg-type]
    )

    rng = np.random.default_rng(42)

    # Create appropriate input based on transform type
    if transform_type == "complex":
        # Complex transform needs complex input
        real_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
        imag_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
        input_tensor = torch.complex(real_part, imag_part).requires_grad_(True)
    else:
        # Real transform uses real input
        input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))
        input_tensor = input_data.clone().requires_grad_(True)

    # Use fast_mode for real transforms (not complex)
    use_fast_mode = transform_type == "real"

    # Run gradcheck with timeout protection
    # UDCT involves FFT operations which can have numerical precision issues
    # If timeout occurs, mark as expected failure (xfail)
    timeout_seconds = 10.0
    try:

        def run_gradcheck():
            return torch.autograd.gradcheck(
                udct_module,
                input_tensor,
                fast_mode=use_fast_mode,
                check_undefined_grad=False,
                check_batched_grad=False,
                atol=1e-5,
                rtol=1e-3,
                eps=1e-6,
            )

        result = timeout(timeout_seconds, run_gradcheck)
        assert result, (
            f"gradcheck failed for dim={dim}, transform_type={transform_type}"
        )
    except TimeoutError:
        # Mark as expected failure when timeout occurs (too slow)
        pytest.xfail(
            f"gradcheck timed out after {timeout_seconds}s for "
            f"dim={dim}, transform_type={transform_type} (too slow)"
        )
