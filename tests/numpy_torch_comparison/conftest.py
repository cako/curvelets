"""Fixtures for NumPy vs PyTorch comparison tests."""

from __future__ import annotations

from tests.torch.conftest import _double_precision_default  # noqa: F401  (autouse)

# Small, non-square, non-power-of-2 shapes for each dimension
TEST_SHAPES = {
    2: (18, 24),
    3: (10, 12, 14),
    4: (6, 8, 10, 12),
}
