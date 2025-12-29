"""PyTorch nn.Module wrapper for UDCT with autograd support."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from ._udct import UDCT
from ._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows


class _UDCTFunction(torch.autograd.Function):
    """Private autograd Function that uses backward transform as gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        image: torch.Tensor,
        udct: UDCT,
        transform_type: Literal["real", "complex", "monogenic"],
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
        transform_type : {"real", "complex", "monogenic"}
            Type of transform to apply.

        Returns
        -------
        torch.Tensor
            Flattened curvelet coefficients.
        """
        # Compute forward transform based on type
        if transform_type == "monogenic":
            coefficients = udct.forward_monogenic(image)
            flattened = udct.vect_monogenic(coefficients)
        elif transform_type == "complex":
            coefficients = udct.forward(image)
            flattened = udct.vect_complex(coefficients)
        else:  # transform_type == "real"
            coefficients = udct.forward(image)
            flattened = udct.vect(coefficients)

        # Save UDCT instance, coefficient template, and transform type for backward
        ctx.udct = udct
        ctx.coefficient_template = coefficients
        ctx.transform_type = transform_type

        return flattened

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
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
        tuple[torch.Tensor | None, None, None]
            Gradient w.r.t. input image, and None for UDCT and transform_type
            (not differentiable).
        """
        udct = ctx.udct
        template = ctx.coefficient_template
        transform_type = ctx.transform_type

        # Restructure gradient and compute backward based on type
        if transform_type == "monogenic":
            grad_coefficients = udct.struct_monogenic(grad_output, template)
            # backward_monogenic returns tuple of (scalar, riesz_1, ..., riesz_ndim)
            components = udct.backward_monogenic(grad_coefficients)
            # Sum all components to get final gradient
            grad_input = sum(components)
        elif transform_type == "complex":
            grad_coefficients = udct.struct_complex(grad_output, template)
            grad_input = udct.backward(grad_coefficients)
        else:  # transform_type == "real"
            grad_coefficients = udct.struct(grad_output, template)
            grad_input = udct.backward(grad_coefficients)

        return grad_input, None, None


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
    transform_type : {"real", "complex", "monogenic"}, optional
        Type of transform to use:
        
        - "real": Real transform (default). Each band captures both positive
          and negative frequencies combined.
        - "complex": Complex transform. Positive and negative frequency bands
          are separated into different directions.
        - "monogenic": Monogenic transform. Produces scalar plus Riesz components
          for each band.
        
        Default is "real".

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCTModule
    >>> 
    >>> # Create module with real transform (default)
    >>> udct = UDCTModule(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3]]))
    >>> input_tensor = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
    >>> output = udct(input_tensor)  # Returns flattened coefficients tensor
    >>> 
    >>> # Create module with complex transform
    >>> udct_complex = UDCTModule(
    ...     shape=(64, 64),
    ...     angular_wedges_config=torch.tensor([[3, 3]]),
    ...     transform_type="complex"
    ... )
    >>> output_complex = udct_complex(input_tensor)
    >>> 
    >>> # Create module with monogenic transform
    >>> udct_monogenic = UDCTModule(
    ...     shape=(64, 64),
    ...     angular_wedges_config=torch.tensor([[3, 3]]),
    ...     transform_type="monogenic"
    ... )
    >>> output_monogenic = udct_monogenic(input_tensor)
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
        transform_type: Literal["real", "complex", "monogenic"] = "real",
    ) -> None:
        super().__init__()
        self._transform_type = transform_type
        # For real/complex transforms, pass use_complex_transform to UDCT
        # For monogenic, it doesn't matter since we use forward_monogenic
        use_complex_transform = transform_type == "complex"
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
        return _UDCTFunction.apply(image, self._udct, self._transform_type)

    def forward_nested(
        self, image: torch.Tensor
    ) -> UDCTCoefficients | MUDCTCoefficients:
        """
        Apply forward curvelet transform and return nested coefficients.

        This method provides access to the nested coefficient structure,
        similar to UDCT.forward() or UDCT.forward_monogenic().

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.

        Returns
        -------
        UDCTCoefficients | MUDCTCoefficients
            Curvelet coefficients organized by scale, direction, and wedge.
            Returns MUDCTCoefficients for monogenic transform, UDCTCoefficients
            for real/complex transforms.
        """
        if self._transform_type == "monogenic":
            return self._udct.forward_monogenic(image)
        return self._udct.forward(image)

    def backward(
        self, coefficients: UDCTCoefficients | MUDCTCoefficients
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Apply backward (inverse) curvelet transform.

        Parameters
        ----------
        coefficients : UDCTCoefficients | MUDCTCoefficients
            Curvelet coefficients from forward transform.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Reconstructed image with shape self.shape.
            For monogenic transform, returns tuple of (scalar, riesz_1, ..., riesz_ndim).
        """
        if self._transform_type == "monogenic":
            return self._udct.backward_monogenic(coefficients)  # type: ignore[arg-type]
        return self._udct.backward(coefficients)  # type: ignore[arg-type]

    def vect(
        self, coefficients: UDCTCoefficients | MUDCTCoefficients
    ) -> torch.Tensor:
        """
        Vectorize curvelet coefficients.

        Parameters
        ----------
        coefficients : UDCTCoefficients | MUDCTCoefficients
            Curvelet coefficients.

        Returns
        -------
        torch.Tensor
            1D tensor containing all coefficients.
        """
        if self._transform_type == "monogenic":
            return self._udct.vect_monogenic(coefficients)  # type: ignore[arg-type]
        elif self._transform_type == "complex":
            return self._udct.vect_complex(coefficients)  # type: ignore[arg-type]
        return self._udct.vect(coefficients)  # type: ignore[arg-type]

    def struct(
        self,
        vector: torch.Tensor,
        template: UDCTCoefficients | MUDCTCoefficients,
    ) -> UDCTCoefficients | MUDCTCoefficients:
        """
        Restructure vectorized coefficients to nested list format.

        Parameters
        ----------
        vector : torch.Tensor
            1D tensor of coefficients.
        template : UDCTCoefficients | MUDCTCoefficients
            Template coefficients for structure information.

        Returns
        -------
        UDCTCoefficients | MUDCTCoefficients
            Restructured coefficients.
        """
        if self._transform_type == "monogenic":
            return self._udct.struct_monogenic(vector, template)  # type: ignore[arg-type]
        elif self._transform_type == "complex":
            return self._udct.struct_complex(vector, template)  # type: ignore[arg-type]
        return self._udct.struct(vector, template)  # type: ignore[arg-type]

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
