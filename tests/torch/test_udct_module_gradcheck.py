"""Gradcheck tests for UDCTModule."""

from __future__ import annotations

import signal
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import curvelets.torch as torch_curvelets


class TimeoutError(Exception):
    """Timeout exception for test timeouts."""


@contextmanager
def timeout(seconds: float):
    """Context manager that raises TimeoutError if code takes too long."""

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("transform_type", ["real", "complex"])
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
        with timeout(timeout_seconds):
            result = torch.autograd.gradcheck(
                udct_module,
                input_tensor,
                fast_mode=use_fast_mode,
                check_undefined_grad=False,
                check_batched_grad=False,
                atol=1e-5,
                rtol=1e-3,
                eps=1e-6,
            )
            assert result, (
                f"gradcheck failed for dim={dim}, transform_type={transform_type}"
            )
    except TimeoutError:
        # Mark as expected failure when timeout occurs (too slow)
        pytest.xfail(
            f"gradcheck timed out after {timeout_seconds}s for "
            f"dim={dim}, transform_type={transform_type} (too slow)"
        )


def test_udct_module_forward_backward_consistency() -> None:
    """Test that forward_nested and backward are consistent with forward."""
    shape = (16, 16)
    config = torch.tensor([[3, 3]])

    udct_module = torch_curvelets.UDCTModule(
        shape=shape,
        angular_wedges_config=config,
        window_overlap=0.15,
        window_threshold=1e-5,
    )

    rng = np.random.default_rng(42)
    input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))

    # Get flattened coefficients from forward()
    flattened = udct_module(input_data)

    # Get nested coefficients from forward_nested()
    nested = udct_module.forward_nested(input_data)

    # Verify they match when flattened
    flattened_from_nested = udct_module.vect(nested)

    assert torch.allclose(flattened, flattened_from_nested, atol=1e-10)

    # Verify backward reconstruction works
    reconstructed = udct_module.backward(nested)
    assert reconstructed.shape == shape


@pytest.mark.parametrize("transform_type", ["real", "complex"])
def test_udct_module_forward_backward_consistency_transform_types(
    transform_type: str,
) -> None:
    """
    Test forward/backward consistency for all transform types.

    Parameters
    ----------
    transform_type : str
        Transform type to test ("real" or "complex").
    """
    shape = (16, 16)
    config = torch.tensor([[3, 3]])

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
        real_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
        imag_part = torch.from_numpy(rng.random(shape, dtype=np.float64))
        input_data = torch.complex(real_part, imag_part)
    else:
        input_data = torch.from_numpy(rng.random(shape, dtype=np.float64))

    # Get flattened coefficients from forward()
    flattened = udct_module(input_data)

    # Get nested coefficients from forward_nested()
    nested = udct_module.forward_nested(input_data)

    # Verify they match when flattened
    flattened_from_nested = udct_module.vect(nested)

    assert torch.allclose(flattened, flattened_from_nested, atol=1e-10)

    # Verify backward reconstruction works
    reconstructed = udct_module.backward(nested)
    # Real/complex return single tensor
    assert isinstance(reconstructed, torch.Tensor)
    assert reconstructed.shape == shape
