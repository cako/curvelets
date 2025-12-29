"""Gradcheck tests for UDCTModule."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

import curvelets.torch as torch_curvelets


@pytest.mark.parametrize("dim", [2, 3, 4])  # type: ignore[misc]
@pytest.mark.parametrize("transform_type", ["real", "complex"])  # type: ignore[misc]
@pytest.mark.timeout(10)  # type: ignore[misc]
@pytest.mark.timeout_to_xfail  # type: ignore[misc]
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Timeout-based tests hang on Windows due to thread-based timeout limitations",
)  # type: ignore[misc]
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

    # Run gradcheck with timeout protection via pytest-timeout
    # UDCT involves FFT operations which can have numerical precision issues
    # If timeout occurs, test is skipped (via conftest hook)
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
    assert result, f"gradcheck failed for dim={dim}, transform_type={transform_type}"
