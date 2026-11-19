"""PyTorch nn.Module wrapper for UDCT with autograd support."""

from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from collections.abc import Callable
from typing import Literal, Self

import torch
from torch import nn

from ._autograd import ForwardWithAdjoint
from ._udct import UDCT
from .typing import UDCTCoefficients, UDCTWindows


class UDCTModule(nn.Module):  # type: ignore[misc]
    """
    PyTorch nn.Module wrapper for UDCT with autograd support.

    PyTorch module with automatic differentiation. When called on an input of
    shape ``(*batch, *shape)``, it returns flattened coefficients of shape
    ``(*batch, ncoef)``, enabling gradient computation through the backward
    transform. Batches are processed natively (no ``torch.vmap`` needed). The backward transform is automatically
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
    dtype : torch.dtype, optional
        Precision of the transform (``torch.float32`` or ``torch.float64``; a
        complex dtype means its real precision); inputs must match it. ``.to(dtype)``, ``.float()`` and ``.double()``
        change it. Default is :func:`torch.get_default_dtype`.
    cuda_graphs : bool, optional
        Replay the transform and its gradient from CUDA graphs on CUDA (see
        :class:`UDCT`); worthwhile for small and medium inputs. Default is
        False.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCTModule
    >>>
    >>> # Create module using num_scales (simplified interface)
    >>> udct = UDCTModule(
    ...     shape=(64, 64), num_scales=3, wedges_per_direction=3, dtype=torch.float64
    ... )
    >>> input_tensor = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)
    >>> output = udct(input_tensor)  # Returns flattened coefficients tensor
    >>>
    >>> # Create module using angular_wedges_config (advanced interface)
    >>> cfg = torch.tensor([[3, 3], [6, 6]], dtype=torch.int64)
    >>> udct2 = UDCTModule(shape=(64, 64), angular_wedges_config=cfg).double()
    >>> output2 = udct2(input_tensor)
    >>>
    >>> # Create module with complex transform
    >>> udct_complex = UDCTModule(
    ...     shape=(64, 64),
    ...     num_scales=3,
    ...     wedges_per_direction=3,
    ...     transform_type="complex",
    ...     dtype=torch.float64,
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
        dtype: torch.dtype | None = None,
        cuda_graphs: bool = False,
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
            dtype=dtype,
            cuda_graphs=cuda_graphs,
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

    @property
    def dtype(self) -> torch.dtype:
        """Real precision of the transform."""
        return self._udct.dtype

    @property
    def cuda_graphs(self) -> bool:
        """Whether CUDA calls replay CUDA graphs (see :class:`UDCT`)."""
        return self._udct.cuda_graphs

    @cuda_graphs.setter
    def cuda_graphs(self, enabled: bool) -> None:
        self._udct.cuda_graphs = enabled

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
        self._udct._check_input(image)  # pylint: disable=protected-access
        # Gradient: the backward (adjoint) transform, no graph through the plan.
        return ForwardWithAdjoint.apply(image, self._udct)

    def coefficient_shapes(self) -> list[list[list[tuple[int, ...]]]]:
        """
        Calculate shapes of all curvelet-domain coefficient arrays.

        Computes the expected shape of every coefficient wedge across all
        scales, directions, and angular wedges without executing a forward
        transform.

        Returns
        -------
        list[list[list[tuple[int, ...]]]]
            Nested list containing the shape of each coefficient wedge.
            Structure is `shapes[scale_idx][direction_idx][wedge_idx]`.
        """
        return self._udct.coefficient_shapes()

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
