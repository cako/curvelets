"""Tests for UDCT._apply method (dtype transfers)."""

from __future__ import annotations

import pytest
import torch

from curvelets.torch import UDCT


class TestUDCTApply:
    """Test that UDCT._apply correctly transfers internal tensors."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def udct(self) -> UDCT:
        """Create a UDCT for testing."""
        return UDCT(shape=(28, 28), num_scales=2, wedges_per_direction=3)

    def test_half_transfers_windows(self, udct: UDCT) -> None:
        """Test that .half() transfers window tensors to float16."""
        udct = udct.half()

        # Check window values are float16 (indices remain int64)
        window = udct.windows[0][0][0]
        assert window.values.dtype == torch.float16

    def test_half_transfers_decimation_ratios(self, udct: UDCT) -> None:
        """Test that .half() transfers decimation_ratios to float16."""
        udct = udct.half()

        # Decimation ratios are int64 and should remain unchanged
        # (half() only affects floating point tensors)
        dec_ratio = udct.decimation_ratios[0]
        assert dec_ratio.dtype == torch.int64

    def test_float_transfers_windows(self, udct: UDCT) -> None:
        """Test that .float() transfers window tensors to float32."""
        # First convert to half, then back to float
        udct = udct.half().float()

        window = udct.windows[0][0][0]
        assert window.values.dtype == torch.float32

    def test_double_transfers_windows(self, udct: UDCT) -> None:
        """Test that .double() transfers window tensors to float64."""
        udct = udct.double()

        window = udct.windows[0][0][0]
        assert window.values.dtype == torch.float64

    def test_float_forward_pass(self, udct: UDCT) -> None:
        """Test that forward pass works after .float() conversion."""
        udct = udct.float()

        x = torch.randn(28, 28, dtype=torch.float32)
        y = udct.forward_vect(x)

        assert y.dtype == torch.complex64

    def test_double_forward_pass(self, udct: UDCT) -> None:
        """Test that forward pass works after .double() conversion."""
        udct = udct.double()

        x = torch.randn(28, 28, dtype=torch.float64)
        y = udct.forward_vect(x)

        assert y.dtype == torch.complex128

    def test_chained_dtype_conversions(self, udct: UDCT) -> None:
        """Test that chained dtype conversions work correctly."""
        # float32 -> float64 -> float16 -> float32
        udct = udct.double().half().float()

        window = udct.windows[0][0][0]
        assert window.values.dtype == torch.float32

        x = torch.randn(28, 28, dtype=torch.float32)
        y = udct.forward_vect(x)
        assert y.dtype == torch.complex64
