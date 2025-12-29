"""PyTorch nn.Module wrapper for UDCT with autograd support."""

from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from typing import Literal

import torch
from torch import nn

from ._typing import UDCTCoefficients, UDCTWindows
from ._udct import UDCT


class _UDCTFunction(torch.autograd.Function):  # type: ignore[misc]  # pylint: disable=abstract-method
    """Private autograd Function that uses backward transform as gradient."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx: torch.autograd.function.FunctionCtx,
        image: torch.Tensor,
        udct: UDCT,
        transform_type: Literal["real", "complex"],
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
        transform_type : ``"real"`` or ``"complex"``
            Type of transform to apply.

        Returns
        -------
        torch.Tensor
            Flattened curvelet coefficients.
        """
        # Compute forward transform and vectorize
        # UDCT handles dispatch based on transform_kind
        coefficients = udct.forward(image)
        flattened = udct.vect(coefficients)

        # Save UDCT instance and transform type for backward
        ctx.udct = udct
        ctx.transform_type = transform_type

        return flattened

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
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

        # Restructure gradient and compute backward
        # UDCT handles dispatch based on transform_kind
        grad_coefficients = udct.struct(grad_output)
        grad_input = udct.backward(grad_coefficients)

        return grad_input, None, None


class UDCTModule(nn.Module):  # type: ignore[misc]
    """
    PyTorch nn.Module wrapper for UDCT with autograd support.

    PyTorch module with automatic differentiation. When called, it returns
    flattened coefficients as a single tensor, enabling gradient computation
    through the backward transform. The backward transform is automatically
    used in the autograd graph when computing gradients.

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
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage. Default is 1e-6.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. Default is "curvelet".
    transform_type : ``"real"`` or ``"complex"``, optional
        Type of transform to use:

        - ``"real"``: Real transform (default). Each band captures both positive
          and negative frequencies combined.
        - ``"complex"``: Complex transform. Positive and negative frequency bands
          are separated into different directions.

        Default is ``"real"``.

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
    >>> # Test with gradcheck
    >>> torch.autograd.gradcheck(udct, input_tensor, atol=1e-5, rtol=1e-3)
    True
    >>>
    >>> # Use struct() to convert flattened coefficients to nested structure
    >>> coeffs_nested = udct.struct(output.detach())
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor,
        window_overlap: float = 0.15,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-6,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_type: Literal["real", "complex"] = "real",
    ) -> None:
        super().__init__()
        self._transform_type = transform_type
        # Pass transform_kind to UDCT (UDCTModule only supports "real" and "complex")
        transform_kind = transform_type  # "real" or "complex"
        self._udct = UDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_kind,
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
        # Delegate to UDCT - it handles dispatch based on transform_kind
        return self._udct.vect(coefficients)

    def struct(
        self,
        vector: torch.Tensor,
    ) -> UDCTCoefficients:
        """
        Restructure vectorized coefficients to nested list format.

        Parameters
        ----------
        vector : torch.Tensor
            1D tensor of coefficients.

        Returns
        -------
        UDCTCoefficients
            Restructured coefficients. The dtype is preserved from the forward
            transform if available, otherwise uses the vector's dtype.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCTModule
        >>> udct = UDCTModule(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3]]))
        >>> input_tensor = torch.randn(64, 64)
        >>> output = udct(input_tensor)
        >>> coeffs_nested = udct.struct(output.detach())
        >>> len(coeffs_nested) > 0
        True
        """
        # Delegate to UDCT - it handles dispatch based on transform_kind
        return self._udct.struct(vector)

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
