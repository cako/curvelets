"""Tests for uncovered UDCT error paths and edge cases.

Tests cover:
- Lines 286, 288: Default assignments for num_scales and wedges_per_direction
- Lines 291-292: num_scales < 2 error
- Lines 294-295: wedges_per_direction < 3 error
- Line 320: Auto-select window_overlap for wedges_per_direction > 5
- Complex input with transform_kind="real" raises ValueError
"""

from __future__ import annotations

import pytest

from curvelets.numpy import UDCT


def test_num_scales_default_assignment():
    """Test default assignment for num_scales=None (line 286)."""
    # num_scales=None should default to 3
    transform = UDCT(shape=(64, 64), num_scales=None, wedges_per_direction=3)
    assert transform.parameters.num_scales == 3


def test_wedges_per_direction_default_assignment(rng):
    """Test default assignment for wedges_per_direction=None (line 288)."""
    # wedges_per_direction=None should default to 3
    transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=None)
    # Verify it works (wedges_per_direction is used internally)
    data = rng.normal(size=(64, 64))
    coeffs = transform.forward(data)
    assert len(coeffs) == 3


def test_num_scales_too_small():
    """Test error for num_scales < 2 (lines 291-292)."""
    with pytest.raises(ValueError, match="num_scales must be >= 2"):
        UDCT(shape=(64, 64), num_scales=1, wedges_per_direction=3)


def test_wedges_per_direction_too_small():
    """Test error for wedges_per_direction < 3 (lines 294-295)."""
    with pytest.raises(ValueError, match="wedges_per_direction must be >= 3"):
        UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=2)


def test_window_overlap_auto_select_wedges_6(rng):
    """Test auto-select window_overlap for wedges_per_direction=6 (line 320)."""
    # wedges_per_direction=6 should hit the else branch (line 320)
    # which sets window_overlap=0.5
    transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=6)
    # Verify it works
    data = rng.normal(size=(64, 64))
    coeffs = transform.forward(data)
    assert len(coeffs) == 3


def test_complex_input_real_transform(rng):
    """Test complex input with transform_kind="real" raises ValueError."""
    # Create transform with transform_kind="real"
    transform = UDCT(
        shape=(64, 64),
        num_scales=3,
        wedges_per_direction=3,
        transform_kind="real",
    )

    # Pass complex input - should raise ValueError
    data = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    with pytest.raises(
        ValueError,
        match="Real transform requires real-valued input.*Use transform_kind='complex'",
    ):
        transform.forward(data)
