"""Test for uncovered forward transform error path.

Tests cover:
- Lines 520-526: Complex input to real transform error
"""

from __future__ import annotations

import pytest

from curvelets.numpy import UDCT
from curvelets.numpy._forward_transform import _apply_forward_transform


def test_complex_input_to_real_transform_error(rng):
    """Test error when complex array is passed to real transform (lines 520-526)."""
    # Create transform to get parameters, windows, and decimation_ratios
    transform = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        use_complex_transform=False,
    )

    # Create complex input
    data = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))

    # Call _apply_forward_transform directly with use_complex_transform=False
    # This should trigger the error at lines 520-526
    with pytest.raises(
        ValueError,
        match="Real transform requires real-valued input.*Use use_complex_transform=True",
    ):
        _apply_forward_transform(
            data,
            transform.parameters,
            transform.windows,
            transform.decimation_ratios,
            use_complex_transform=False,
        )
