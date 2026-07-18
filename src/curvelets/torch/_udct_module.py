"""PyTorch nn.Module wrapper for UDCT with autograd support."""

from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from typing_extensions import Self

from ._udct import UDCT
from .typing import UDCTCoefficients, UDCTWindows


class _UDCTFunction(torch.autograd.Function):  # type: ignore[misc]  # pylint: disable=abstract-method
    """Private autograd Function that uses backward transform as gradient.

    This implementation uses the modern setup_context pattern to support
    functorch transforms like vmap, grad, and jacrev.
    """

    # Enable automatic vmap rule generation for functorch compatibility
    generate_vmap_rule = True

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        image: torch.Tensor,
        udct: UDCT,
    ) -> torch.Tensor:
        """
        Forward pass: compute forward transform and flatten coefficients.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        udct : UDCT
            UDCT instance to use for transform.

        Returns
        -------
        torch.Tensor
            Flattened curvelet coefficients.
        """
        # Compute forward transform and vectorize
        # UDCT handles dispatch based on transform_kind
        coefficients = udct.forward(image)
        return udct.vect(coefficients)

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, UDCT],
        output: torch.Tensor,  # noqa: ARG004
    ) -> None:
        """
        Save context for backward pass.

        This method is required for functorch compatibility (vmap, grad, etc.).

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving information for backward pass.
        inputs : tuple
            Tuple of (image, udct) from forward.
        output : torch.Tensor
            Output from forward (unused).
        """
        _, udct = inputs
        ctx.udct = udct  # ty: ignore[unresolved-attribute]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        """
        Backward pass: use backward transform to compute gradient.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context containing saved information from forward pass.
        *grad_outputs : torch.Tensor
            Gradient w.r.t. flattened coefficients (single tensor expected).

        Returns
        -------
        tuple[torch.Tensor | None, None]
            Gradient w.r.t. input image, and None for UDCT (not differentiable).
        """
        udct = ctx.udct  # ty: ignore[unresolved-attribute]
        grad_output = grad_outputs[0]

        # Restructure gradient and compute backward
        # UDCT handles dispatch based on transform_kind
        grad_coefficients = udct.struct(grad_output)
        grad_input = udct.backward(grad_coefficients)

        return grad_input, None


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
    angular_wedges_config : torch.Tensor, optional
        Configuration tensor specifying the number of angular wedges per scale
        and dimension. Shape is (num_scales - 1, dimension), where num_scales
        includes the lowpass scale. If provided, cannot be used together with
        num_scales/wedges_per_direction. Default is None.
    num_scales : int, optional
        Total number of scales (including lowpass scale 0). Must be >= 2.
        Used when angular_wedges_config is not provided. Default is 3.
    wedges_per_direction : int, optional
        Number of angular wedges per direction at the coarsest scale.
        The number of wedges doubles at each finer scale. Must be >= 3.
        Used when angular_wedges_config is not provided. Default is 3.
    window_overlap : float, optional
        Window overlap parameter controlling the smoothness of window transitions.
        If None and using num_scales/wedges_per_direction, automatically chosen
        based on wedges_per_direction. Default is None (auto) or 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency band parameters.
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage. Default is 1e-5.
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
    >>> # Create module using num_scales (simplified interface)
    >>> udct = UDCTModule(shape=(64, 64), num_scales=3, wedges_per_direction=3)
    >>> input_tensor = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
    >>> output = udct(input_tensor)  # Returns flattened coefficients tensor
    >>>
    >>> # Create module using angular_wedges_config (advanced interface)
    >>> cfg = torch.tensor([[3, 3], [6, 6]], dtype=torch.int64)
    >>> udct2 = UDCTModule(shape=(64, 64), angular_wedges_config=cfg)
    >>> output2 = udct2(input_tensor)
    >>>
    >>> # Create module with complex transform
    >>> udct_complex = UDCTModule(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
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
        angular_wedges_config: torch.Tensor | None = None,
        num_scales: int | None = None,
        wedges_per_direction: int | None = None,
        window_overlap: float | None = None,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-5,
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
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_kind,
        )

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Self:
        """
        Apply a function to all tensors, including internal UDCT tensors.

        This method is called by PyTorch's ``.to()``, ``.cuda()``, ``.cpu()``,
        and ``.half()`` methods. It ensures that the internal UDCT tensors
        (windows, decimation ratios) are moved along with the module's
        parameters and buffers.

        Parameters
        ----------
        fn : Callable[[torch.Tensor], torch.Tensor]
            Function to apply to each tensor.
        recurse : bool, optional
            Whether to recursively apply to submodules. Default is True.

        Returns
        -------
        Self
            The module instance.
        """
        super()._apply(fn, recurse)
        self._udct.apply_to_tensors(fn)
        return self

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
