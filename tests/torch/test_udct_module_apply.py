"""Tests for UDCTModule._apply method (dtype transfers)."""

from __future__ import annotations

import pytest
import torch

from curvelets.torch import UDCTModule


class TestUDCTModuleApply:
    """Test that UDCTModule._apply correctly transfers internal tensors."""

    @pytest.fixture  # type: ignore[misc]
    def udct_module(self) -> UDCTModule:
        """Create a UDCTModule for testing."""
        return UDCTModule(shape=(28, 28), num_scales=2, wedges_per_direction=3)

    def test_half_transfers_windows(self, udct_module: UDCTModule) -> None:
        """Test that .half() transfers window tensors to float16."""
        udct_module = udct_module.half()

        # Check window values are float16 (indices remain int64)
        window_idx, window_val = udct_module.windows[0][0][0]
        assert window_val.dtype == torch.float16

    def test_half_transfers_decimation_ratios(self, udct_module: UDCTModule) -> None:
        """Test that .half() transfers decimation_ratios to float16."""
        udct_module = udct_module.half()

        # Decimation ratios are int64 and should remain unchanged
        # (half() only affects floating point tensors)
        dec_ratio = udct_module.decimation_ratios[0]
        assert dec_ratio.dtype == torch.int64

    def test_float_transfers_windows(self, udct_module: UDCTModule) -> None:
        """Test that .float() transfers window tensors to float32."""
        # First convert to half, then back to float
        udct_module = udct_module.half().float()

        window_idx, window_val = udct_module.windows[0][0][0]
        assert window_val.dtype == torch.float32

    def test_double_transfers_windows(self, udct_module: UDCTModule) -> None:
        """Test that .double() transfers window tensors to float64."""
        udct_module = udct_module.double()

        window_idx, window_val = udct_module.windows[0][0][0]
        assert window_val.dtype == torch.float64

    def test_float_forward_pass(self, udct_module: UDCTModule) -> None:
        """Test that forward pass works after .float() conversion."""
        udct_module = udct_module.float()

        x = torch.randn(28, 28, dtype=torch.float32)
        y = udct_module(x)

        assert y.dtype == torch.complex64

    def test_double_forward_pass(self, udct_module: UDCTModule) -> None:
        """Test that forward pass works after .double() conversion."""
        udct_module = udct_module.double()

        x = torch.randn(28, 28, dtype=torch.float64)
        y = udct_module(x)

        assert y.dtype == torch.complex128

    def test_chained_dtype_conversions(self, udct_module: UDCTModule) -> None:
        """Test that chained dtype conversions work correctly."""
        # float32 -> float64 -> float16 -> float32
        udct_module = udct_module.double().half().float()

        window_idx, window_val = udct_module.windows[0][0][0]
        assert window_val.dtype == torch.float32

        x = torch.randn(28, 28, dtype=torch.float32)
        y = udct_module(x)
        assert y.dtype == torch.complex64
