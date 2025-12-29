"""PyTorch nn.Module wrapper for UDCT with autograd support."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from ._udct import UDCT
from ._typing import UDCTCoefficients, UDCTWindows


class _UDCTFunction(torch.autograd.Function):
    """Private autograd Function that uses backward transform as gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        image: torch.Tensor,
        udct: UDCT,
    ) -> torch.Tensor:
        """
        Forward pass: compute forward transform and flatten coefficients.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving information for backward pass.
        image : torch.Tensor
            Input image tensor.
        udct : UDCT
            UDCT instance to use for transform.

        Returns
        -------
        torch.Tensor
            Flattened curvelet coefficients.
        """
        # Compute forward transform to get nested coefficients
        coefficients = udct.forward(image)

        # Save UDCT instance and coefficient template for backward
        ctx.udct = udct
        ctx.coefficient_template = coefficients

        # Flatten coefficients to single tensor
        flattened = udct.vect(coefficients)

        return flattened

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        """
        Backward pass: use backward transform to compute gradient.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context containing saved information from forward pass.
        grad_output : torch.Tensor
            Gradient w.r.t. flattened coefficients.

        Returns
        -------
        tuple[torch.Tensor | None, None]
            Gradient w.r.t. input image, and None for UDCT (not differentiable).
        """
        udct = ctx.udct
        template = ctx.coefficient_template

        # Restructure gradient to match coefficient structure
        grad_coefficients = udct.struct(grad_output, template)

        # Use backward transform to compute gradient w.r.t. input
        grad_input = udct.backward(grad_coefficients)

        return grad_input, None


class UDCTModule(nn.Module):
    """
    PyTorch nn.Module wrapper for UDCT with autograd support.

    This module provides the same interface as UDCT but can be used as a
    PyTorch module with automatic differentiation. When called, it returns
    flattened coefficients as a single tensor, enabling gradient computation
    through the backward transform.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    angular_wedges_config : torch.Tensor
        Configuration specifying number of angular wedges per scale and dimension.
        Shape is (num_scales-1, ndim).
    window_overlap : float, optional
        Window overlap parameter. Default is 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency band parameters.
        Default is (pi/3, 2*pi/3, 2*pi/3, 4*pi/3).
    window_threshold : float, optional
        Threshold for sparse window storage. Default is 1e-6.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. Default is "curvelet".
    use_complex_transform : bool, optional
        Whether to use complex transform mode. Default is False.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCTModule
    >>> 
    >>> # Create module
    >>> udct = UDCTModule(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3]]))
    >>> 
    >>> # Use with autograd
    >>> input_tensor = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
    >>> output = udct(input_tensor)  # Returns flattened coefficients tensor
    >>> 
    >>> # Test with gradcheck
    >>> torch.autograd.gradcheck(udct, input_tensor, atol=1e-5, rtol=1e-3)
    True
    >>> 
    >>> # Still has UDCT interface methods
    >>> coeffs_nested = udct.forward_nested(input_tensor)  # Returns nested coefficients
    >>> reconstructed = udct.backward(coeffs_nested)  # Reconstruct from nested coefficients
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor,
        window_overlap: float = 0.15,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-6,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        use_complex_transform: bool = False,
    ) -> None:
        super().__init__()
        self._udct = UDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
            use_complex_transform=use_complex_transform,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute forward transform and return flattened coefficients.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.

        Returns
        -------
        torch.Tensor
            Flattened curvelet coefficients as a single tensor.
        """
        return _UDCTFunction.apply(image, self._udct)

    def forward_nested(self, image: torch.Tensor) -> UDCTCoefficients:
        """
        Apply forward curvelet transform and return nested coefficients.

        This method provides access to the nested coefficient structure,
        similar to UDCT.forward().

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients organized by scale, direction, and wedge.
        """
        return self._udct.forward(image)

    def backward(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """
        Apply backward (inverse) curvelet transform.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Curvelet coefficients from forward transform.

        Returns
        -------
        torch.Tensor
            Reconstructed image with shape self.shape.
        """
        return self._udct.backward(coefficients)

    def vect(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """
        Vectorize curvelet coefficients.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Curvelet coefficients.

        Returns
        -------
        torch.Tensor
            1D tensor containing all coefficients.
        """
        return self._udct.vect(coefficients)

    def struct(
        self, vector: torch.Tensor, template: UDCTCoefficients
    ) -> UDCTCoefficients:
        """
        Restructure vectorized coefficients to nested list format.

        Parameters
        ----------
        vector : torch.Tensor
            1D tensor of coefficients.
        template : UDCTCoefficients
            Template coefficients for structure information.

        Returns
        -------
        UDCTCoefficients
            Restructured coefficients.
        """
        return self._udct.struct(vector, template)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the transform."""
        return self._udct.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._udct.ndim

    @property
    def num_scales(self) -> int:
        """Number of scales."""
        return self._udct.num_scales

    @property
    def windows(self) -> UDCTWindows:
        """Curvelet windows in sparse format."""
        return self._udct.windows

    @property
    def decimation_ratios(self) -> list[torch.Tensor]:
        """Decimation ratios for each scale."""
        return self._udct.decimation_ratios
