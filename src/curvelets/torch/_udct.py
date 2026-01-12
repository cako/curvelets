"""Main UDCT class for PyTorch implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import logging
from collections.abc import Callable
from math import prod, sqrt
from typing import Any, Literal

import numpy as np
import torch

from ._backward_transform import _apply_backward_transform
from ._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_real,
)
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT, from_sparse_new
from .typing import UDCTCoefficients, UDCTWindows


class UDCT:
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms.

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
        Radial frequency parameters defining the frequency bands.
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-5.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" creates a single ring-shaped window (bandpass filter only,
        no angular components) at the highest scale with decimation=1.
        Default is "curvelet".
    transform_kind : {"real", "complex", "monogenic"}, optional
        Type of transform to use:

        - "real" (default): Real transform where each band captures both
          positive and negative frequencies combined.
        - "complex": Complex transform which separates positive and negative
          frequency components into different bands. Each band is scaled by
          :math:`\\sqrt{0.5}`.
        - "monogenic": Monogenic transform that extends the curvelet transform
          by applying Riesz transforms, producing ndim+1 components per band
          (scalar plus all Riesz components).

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    high_frequency_mode : str
        High frequency mode.
    transform_kind : str
        Type of transform being used ("real", "complex", or "monogenic").
    parameters : ParamUDCT
        Internal UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list
        Decimation ratios for each scale/direction.

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCT
    >>> # Create a 2D transform using num_scales (simplified interface)
    >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
    >>> data = torch.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> torch.allclose(data, recon, atol=1e-4)
    True
    >>> # Create using angular_wedges_config (advanced interface)
    >>> cfg = torch.tensor([[3, 3], [6, 6]], dtype=torch.int64)
    >>> transform2 = UDCT(shape=(64, 64), angular_wedges_config=cfg)
    >>> coeffs2 = transform2.forward(data)
    >>> recon2 = transform2.backward(coeffs2)
    >>> torch.allclose(data, recon2, atol=1e-4)
    True
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
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
    ) -> None:
        # Validate transform_kind
        if transform_kind not in ("real", "complex", "monogenic"):
            msg = f"transform_kind must be 'real', 'complex', or 'monogenic', got {transform_kind!r}"
            raise ValueError(msg)

        # Store basic attributes
        self._high_frequency_mode = high_frequency_mode
        self._transform_kind = transform_kind
        self._coefficient_dtype: torch.dtype | None = None

        # Calculate necessary parameters
        params_dict = self._initialize_parameters(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
        )

        # Create ParamUDCT object
        self._parameters = ParamUDCT(
            shape=params_dict["internal_shape"],
            angular_wedges_config=params_dict["angular_wedges_config"],
            window_overlap=params_dict["window_overlap"],
            radial_frequency_params=params_dict["radial_frequency_params"],
            window_threshold=params_dict["window_threshold"],
        )

        # Calculate windows
        self._windows, self._decimation_ratios, self._indices = (
            self._initialize_windows()
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the transform.

        :no-index:
        """
        return self._parameters.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._parameters.ndim

    @property
    def num_scales(self) -> int:
        """Number of scales."""
        return self._parameters.num_scales

    @property
    def windows(self) -> UDCTWindows:
        """
        Curvelet windows in sparse format.

        :no-index:
        """
        return self._windows

    @property
    def decimation_ratios(self) -> list[torch.Tensor]:
        """
        Decimation ratios for each scale.

        :no-index:
        """
        return self._decimation_ratios

    @staticmethod
    def _compute_optimal_window_overlap(
        wedges_per_scale: torch.Tensor,
    ) -> float:
        """
        Compute optimal window_overlap from Nguyen & Chauris (2010) constraint.

        The constraint :math:`(2^{s/N})(1+2a)(1+a) < N` must hold for all scales.
        This method solves for the theoretical maximum at each scale,
        takes the minimum across all scales, and returns 10% of this value.

        The 10% factor was empirically determined to give better reconstruction
        quality than both the theoretical maximum and previous hardcoded defaults.

        Parameters
        ----------
        wedges_per_scale : torch.Tensor
            Tensor of wedge counts per scale.

        Returns
        -------
        float
            Optimal window_overlap value (10% of theoretical maximum).

        References
        ----------
        .. [1] Nguyen, T. T., and H. Chauris, 2010, "Uniform Discrete Curvelet
           Transform": IEEE Transactions on Signal Processing, 58, 3618-3634.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> # For wedges_per_direction=3, num_scales=3
        >>> wedges_per_scale = torch.tensor([3, 6], dtype=torch.int64)
        >>> alpha = UDCT._compute_optimal_window_overlap(wedges_per_scale)
        >>> 0.1 < alpha < 0.2  # Expected around 0.11
        True
        """
        min_overlap = float("inf")

        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            num_wedges_float = float(num_wedges.item())
            k = 2 ** (scale_idx / num_wedges_float)
            # Solve: 2k*a^2 + 3k*a + (k - N) = 0
            # a = (-3 + sqrt(1 + 8*N/k)) / 4
            discriminant = 1 + 8 * num_wedges_float / k
            a_max = (-3 + sqrt(discriminant)) / 4
            min_overlap = min(min_overlap, a_max)

        # Use 10% of theoretical max for optimal reconstruction quality
        return 0.1 * min_overlap

    @staticmethod
    def _compute_from_angular_wedges_config(
        angular_wedges_config: torch.Tensor,
        window_overlap: float | None,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute angular wedges configuration and window overlap from provided config.

        Parameters
        ----------
        angular_wedges_config : torch.Tensor
            Configuration tensor specifying the number of angular wedges per scale
            and dimension. Shape is (num_scales - 1, dimension), where num_scales
            includes the lowpass scale.
        window_overlap : float | None
            Window overlap parameter. If None, automatically computed using
            the Nguyen & Chauris (2010) constraint formula.

        Returns
        -------
        tuple[torch.Tensor, float]
            Tuple of (computed_angular_wedges_config, computed_window_overlap).
        """
        # Use provided angular_wedges_config directly
        computed_angular_wedges_config = angular_wedges_config

        # Validate that all wedge counts are divisible by 3
        # According to Nguyen & Chauris (2010), the decimation ratio formula
        # uses integer division by 3, so wedges must be divisible by 3
        invalid_mask = computed_angular_wedges_config % 3 != 0
        if invalid_mask.any():
            invalid_values = torch.unique(
                computed_angular_wedges_config[invalid_mask]
            ).tolist()
            msg = (
                f"All values in angular_wedges_config must be divisible by 3. "
                f"Found invalid values: {invalid_values}. "
                "According to the Nguyen & Chauris (2010) paper specification, "
                "the decimation ratio formula requires integer division by 3. "
                "Recommended values are: 3, 6, 12"
            )
            raise ValueError(msg)

        # Use provided window_overlap or compute optimal from first dimension
        if window_overlap is not None:
            computed_window_overlap = window_overlap
        else:
            # Compute from first dimension's wedges (all dimensions should be consistent)
            wedges_per_scale = angular_wedges_config[:, 0]
            computed_window_overlap = UDCT._compute_optimal_window_overlap(
                wedges_per_scale
            )

        return computed_angular_wedges_config, computed_window_overlap

    @staticmethod
    def _compute_from_num_scales(
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        dimension: int,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute angular wedges configuration and window overlap from num_scales.

        Parameters
        ----------
        num_scales : int | None
            Total number of scales (including lowpass scale 0). Must be >= 2.
            If None, defaults to 3.
        wedges_per_direction : int | None
            Number of angular wedges per direction at the coarsest scale.
            Must be >= 3. If None, defaults to 3.
        window_overlap : float | None
            Window overlap parameter. If None, auto-selected based on
            wedges_per_direction.
        dimension : int
            Number of dimensions.

        Returns
        -------
        tuple[torch.Tensor, float]
            Tuple of (computed_angular_wedges_config, computed_window_overlap).

        Raises
        ------
        ValueError
            If num_scales < 2 or wedges_per_direction < 3.
        """
        # Use num_scales/wedges_per_direction
        if num_scales is None:
            num_scales = 3
        if wedges_per_direction is None:
            wedges_per_direction = 3

        if num_scales < 2:
            msg = "num_scales must be >= 2"
            raise ValueError(msg)
        if wedges_per_direction < 3:
            msg = "wedges_per_direction must be >= 3"
            raise ValueError(msg)
        if wedges_per_direction % 3 != 0:
            msg = (
                f"wedges_per_direction={wedges_per_direction} must be divisible by 3. "
                "According to the Nguyen & Chauris (2010) paper specification, "
                "the decimation ratio formula requires integer division by 3. "
                "Recommended values are: 3, 6, 12"
            )
            raise ValueError(msg)

        # Convert to angular_wedges_config
        wedges_per_scale = wedges_per_direction * 2 ** torch.arange(
            num_scales - 1, dtype=torch.int64
        )
        computed_angular_wedges_config = wedges_per_scale.unsqueeze(1).repeat(
            1, dimension
        )

        # Auto-select window_overlap if not provided
        if window_overlap is None:
            computed_window_overlap = UDCT._compute_optimal_window_overlap(
                wedges_per_scale
            )
        else:
            computed_window_overlap = window_overlap

        # Validate window_overlap
        for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
            num_wedges_float = float(num_wedges.item())
            const = (
                2 ** (scale_idx / num_wedges_float)
                * (1 + 2 * computed_window_overlap)
                * (1 + computed_window_overlap)
            )
            if const >= num_wedges_float:
                msg = (
                    f"window_overlap={computed_window_overlap:.3f} does not respect the relationship "
                    f"(2^{scale_idx}/{num_wedges_float})(1+2a)(1+a) = {const:.3f} < {num_wedges_float} for scale {scale_idx + 1}"
                )
                logging.warning(msg)

        return computed_angular_wedges_config, computed_window_overlap

    def _initialize_parameters(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor | None,
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        radial_frequency_params: tuple[float, float, float, float] | None,
        window_threshold: float,
    ) -> dict[str, Any]:
        """
        Calculate all necessary parameters for UDCT initialization.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input data.
        angular_wedges_config : torch.Tensor | None
            Configuration tensor, or None to use num_scales/wedges_per_direction.
        num_scales : int | None
            Number of scales (used when angular_wedges_config is None).
        wedges_per_direction : int | None
            Wedges per direction (used when angular_wedges_config is None).
        window_overlap : float | None
            Window overlap parameter.
        radial_frequency_params : tuple[float, float, float, float] | None
            Radial frequency parameters.
        window_threshold : float
            Window threshold.

        Returns
        -------
        dict
            Dictionary containing all calculated parameters.
        """
        dimension = len(shape)

        # Determine which initialization style to use
        if angular_wedges_config is not None:
            if num_scales is not None or wedges_per_direction is not None:
                msg = "Cannot specify both angular_wedges_config and num_scales/wedges_per_direction"
                raise ValueError(msg)
            computed_angular_wedges_config, computed_window_overlap = (
                self._compute_from_angular_wedges_config(
                    angular_wedges_config, window_overlap
                )
            )
        else:
            computed_angular_wedges_config, computed_window_overlap = (
                self._compute_from_num_scales(
                    num_scales, wedges_per_direction, window_overlap, dimension
                )
            )

        # Compute num_scales from computed_angular_wedges_config for validation
        computed_num_scales = 1 + len(computed_angular_wedges_config)

        # Validate scale requirements
        if computed_num_scales < 2:
            msg = "requires at least 2 scales total (num_scales >= 2)"
            raise ValueError(msg)

        # Calculate internal shape
        internal_shape = shape

        # Set default radial_frequency_params if not provided
        if radial_frequency_params is None:
            computed_radial_frequency_params: tuple[float, float, float, float] = (
                np.pi / 3,
                2 * np.pi / 3,
                2 * np.pi / 3,
                4 * np.pi / 3,
            )
        else:
            computed_radial_frequency_params = radial_frequency_params

        return {
            "dimension": dimension,
            "internal_shape": internal_shape,
            "angular_wedges_config": computed_angular_wedges_config,
            "window_overlap": computed_window_overlap,
            "radial_frequency_params": computed_radial_frequency_params,
            "window_threshold": window_threshold,
        }

    def _initialize_windows(
        self,
    ) -> tuple[UDCTWindows, list[torch.Tensor], dict[int, dict[int, torch.Tensor]]]:
        """
        Calculate curvelet windows, decimation ratios, and indices.

        Returns
        -------
        tuple
            (windows, decimation_ratios, indices)
        """
        window_computer = UDCTWindow(self._parameters, self._high_frequency_mode)
        return window_computer.compute()

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
        parts: list[torch.Tensor] = []
        for scale in coefficients:
            for direction in scale:
                for wedge_coeff in direction:
                    parts.append(wedge_coeff.flatten())
        return torch.cat(parts)

    def struct(self, vector: torch.Tensor) -> UDCTCoefficients:
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
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3]]))
        >>> data = torch.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs)
        True
        """
        # Dispatch based on transform_kind
        if self._transform_kind == "monogenic":
            return self._struct_monogenic(vector)
        if self._transform_kind == "complex":
            return self._struct_complex(vector)
        # real
        return self._struct_real(vector)

    def _struct_real(self, vector: torch.Tensor) -> UDCTCoefficients:
        """Private method for real coefficient restructuring (no input validation)."""
        begin_idx = 0
        coefficients: UDCTCoefficients = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        for scale_idx, decimation_ratios_scale in enumerate(self._decimation_ratios):
            coefficients.append([])
            num_directions = len(decimation_ratios_scale)

            for direction_idx in range(num_directions):
                coefficients[scale_idx].append([])
                window_direction_idx = min(
                    direction_idx, len(self._windows[scale_idx]) - 1
                )
                decimation_ratio_dir = decimation_ratios_scale[
                    min(direction_idx, len(decimation_ratios_scale) - 1), :
                ]

                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_decimated = (internal_shape // decimation_ratio_dir).tolist()
                    size = prod(shape_decimated)
                    end_idx = begin_idx + size
                    coeff = vector[begin_idx:end_idx].reshape(shape_decimated)
                    # Preserve dtype from forward transform (Option B)
                    if self._coefficient_dtype is not None:
                        coeff = coeff.to(self._coefficient_dtype)
                    coefficients[scale_idx][direction_idx].append(coeff)
                    begin_idx = end_idx
        return coefficients

    def _struct_complex(self, vector: torch.Tensor) -> UDCTCoefficients:
        """Private method for complex coefficient restructuring (no input validation)."""
        begin_idx = 0
        coefficients: UDCTCoefficients = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        for scale_idx, decimation_ratios_scale in enumerate(self._decimation_ratios):
            coefficients.append([])
            # In complex transform mode, we have 2*dim directions per scale (for scale > 0)
            # but decimation_ratios_scale only has dim rows, so we need to handle this
            if scale_idx > 0:
                num_directions = 2 * self._parameters.ndim
            else:
                num_directions = len(decimation_ratios_scale)

            for direction_idx in range(num_directions):
                coefficients[scale_idx].append([])
                # In complex transform mode, directions >= dim reuse windows and decimation ratios
                # from directions < dim. Negative frequency directions (dim..2*dim-1) use the same
                # windows and decimation ratios as positive frequency directions (0..dim-1)
                # For "wavelet" mode at highest scale, windows only has 1 direction, so use index 0
                if scale_idx > 0 and direction_idx >= self._parameters.ndim:
                    window_direction_idx = direction_idx % self._parameters.ndim
                    # Clamp to available windows (for "wavelet" mode at highest scale)
                    window_direction_idx = min(
                        window_direction_idx, len(self._windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(window_direction_idx, len(decimation_ratios_scale) - 1), :
                    ]
                else:
                    window_direction_idx = min(
                        direction_idx, len(self._windows[scale_idx]) - 1
                    )
                    decimation_ratio_dir = decimation_ratios_scale[
                        min(direction_idx, len(decimation_ratios_scale) - 1), :
                    ]

                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_decimated = (internal_shape // decimation_ratio_dir).tolist()
                    size = prod(shape_decimated)
                    end_idx = begin_idx + size
                    coeff = vector[begin_idx:end_idx].reshape(shape_decimated)
                    # Preserve dtype from forward transform (Option B)
                    if self._coefficient_dtype is not None:
                        coeff = coeff.to(self._coefficient_dtype)
                    coefficients[scale_idx][direction_idx].append(coeff)
                    begin_idx = end_idx
        return coefficients

    def _struct_monogenic(self, vector: torch.Tensor) -> UDCTCoefficients:
        """Private method for monogenic coefficient restructuring (no input validation)."""
        begin_idx = 0
        coefficients: UDCTCoefficients = []
        internal_shape = torch.tensor(self._parameters.shape, dtype=torch.int64)
        num_components = self._parameters.ndim + 1  # scalar + all Riesz components

        for scale_idx, decimation_ratios_scale in enumerate(self._decimation_ratios):
            coefficients.append([])
            num_directions = len(decimation_ratios_scale)

            for direction_idx in range(num_directions):
                coefficients[scale_idx].append([])
                window_direction_idx = min(
                    direction_idx, len(self._windows[scale_idx]) - 1
                )
                decimation_ratio_dir = decimation_ratios_scale[
                    min(direction_idx, len(decimation_ratios_scale) - 1), :
                ]

                for _ in self._windows[scale_idx][window_direction_idx]:
                    shape_decimated = (internal_shape // decimation_ratio_dir).tolist()
                    # For monogenic, each wedge has ndim+1 components
                    wedge_components: list[torch.Tensor] = []
                    for _ in range(num_components):
                        size = prod(shape_decimated)
                        end_idx = begin_idx + size
                        component = vector[begin_idx:end_idx].reshape(shape_decimated)
                        # Preserve dtype from forward transform (Option B)
                        if self._coefficient_dtype is not None:
                            component = component.to(self._coefficient_dtype)
                        wedge_components.append(component)
                        begin_idx = end_idx
                    coefficients[scale_idx][direction_idx].append(wedge_components)
        return coefficients

    def _from_sparse(
        self, arr_sparse: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert sparse window format to dense array.

        Parameters
        ----------
        arr_sparse : tuple[torch.Tensor, torch.Tensor]
            Sparse window format as a tuple of (indices, values).

        Returns
        -------
        torch.Tensor
            Dense tensor with the same shape as the transform input.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        >>> # Get a sparse window
        >>> sparse_window = transform.windows[0][0][0]
        >>> # Convert to dense
        >>> dense_window = transform._from_sparse(sparse_window)
        >>> dense_window.shape
        torch.Size([64, 64])
        """
        idx, val = from_sparse_new(arr_sparse)
        arr_full = torch.zeros(
            self._parameters.shape, dtype=val.dtype, device=val.device
        )
        arr_full.flatten()[idx.flatten()] = val.flatten()
        return arr_full

    def forward(self, image: torch.Tensor) -> UDCTCoefficients:
        """
        Apply forward curvelet transform.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.
            - For transform_kind="real" or "monogenic": must be real-valued
            - For transform_kind="complex": can be real-valued or complex-valued

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients organized by scale, direction, and wedge.
            For monogenic transforms, each coefficient tensor has shape
            (*wedge_shape, ndim+1) with channels stacked along the last axis.
        """
        # Validate input based on transform_kind
        if self._transform_kind in ("real", "monogenic") and image.is_complex():
            msg = (
                f"{self._transform_kind.capitalize()} transform requires real-valued input. "
                "Got complex tensor. Use transform_kind='complex' for complex inputs."
            )
            raise ValueError(msg)

        # Dispatch based on transform_kind
        if self._transform_kind == "real":
            return self._forward_real(image)
        if self._transform_kind == "complex":
            return self._forward_complex(image)
        # Must be "monogenic" (checked by type system)
        return self._forward_monogenic(image)

    def _forward_real(self, image: torch.Tensor) -> UDCTCoefficients:
        """Private method for real forward transform (no input validation)."""
        coeffs = _apply_forward_transform_real(
            image, self._parameters, self._windows, self._decimation_ratios
        )
        # Store coefficient dtype from first non-empty coefficient (Option B)
        if self._coefficient_dtype is None and len(coeffs) > 0:
            for scale in coeffs:
                found = False
                for direction in scale:
                    for wedge in direction:
                        if wedge.numel() > 0:
                            self._coefficient_dtype = wedge.dtype
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return coeffs

    def _forward_complex(self, image: torch.Tensor) -> UDCTCoefficients:
        """Private method for complex forward transform (no input validation)."""
        coeffs = _apply_forward_transform_complex(
            image, self._parameters, self._windows, self._decimation_ratios
        )
        # Store coefficient dtype from first non-empty coefficient (Option B)
        if self._coefficient_dtype is None and len(coeffs) > 0:
            for scale in coeffs:
                found = False
                for direction in scale:
                    for wedge in direction:
                        if wedge.numel() > 0:
                            self._coefficient_dtype = wedge.dtype
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return coeffs

    def _forward_monogenic(self, image: torch.Tensor) -> UDCTCoefficients:  # pylint: disable=unused-argument
        """Private method for monogenic forward transform (no input validation)."""
        # TODO: Implement monogenic forward transform for PyTorch
        # For now, raise NotImplementedError
        # When implemented, store dtype like in _forward_real and _forward_complex
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)

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
        # Dispatch based on transform_kind
        if self._transform_kind == "real":
            return self._backward_real(coefficients)
        if self._transform_kind == "complex":
            return self._backward_complex(coefficients)
        # Must be "monogenic" (checked by type system)
        return self._backward_monogenic(coefficients)

    def _backward_real(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """Private method for real backward transform (no input validation)."""
        return _apply_backward_transform(
            coefficients,
            self._parameters,
            self._windows,
            self._decimation_ratios,
            use_complex_transform=False,
        )

    def _backward_complex(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """Private method for complex backward transform (no input validation)."""
        return _apply_backward_transform(
            coefficients,
            self._parameters,
            self._windows,
            self._decimation_ratios,
            use_complex_transform=True,
        )

    def _backward_monogenic(self, coefficients: UDCTCoefficients) -> torch.Tensor:  # pylint: disable=unused-argument
        """Private method for monogenic backward transform (no input validation)."""
        # TODO: Implement monogenic backward transform for PyTorch
        # For now, raise NotImplementedError
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)

    def apply_to_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Apply a function to all internal tensors.

        This method is used for device and dtype transfers (e.g., when calling
        `model.to(device)` on a module containing this UDCT instance). It applies
        the given function to all internal tensor attributes including windows
        and decimation ratios.

        Parameters
        ----------
        fn : Callable[[torch.Tensor], torch.Tensor]
            Function to apply to each tensor. Typically this is a function like
            `lambda t: t.cuda()` or `lambda t: t.to(device)`.

        Examples
        --------
        >>> import torch
        >>> from curvelets.torch import UDCT
        >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
        >>> # Move all internal tensors to GPU (if available)
        >>> if torch.cuda.is_available():
        ...     transform.apply_to_tensors(lambda t: t.cuda())  # doctest: +SKIP
        """
        # Apply to windows (nested structure of (idx, val) tuples)
        for scale_idx, scale_windows in enumerate(self._windows):
            for dir_idx, dir_windows in enumerate(scale_windows):
                for wedge_idx, (idx, val) in enumerate(dir_windows):
                    self._windows[scale_idx][dir_idx][wedge_idx] = (fn(idx), fn(val))

        # Apply to decimation_ratios (list of tensors)
        for i, ratio in enumerate(self._decimation_ratios):
            self._decimation_ratios[i] = fn(ratio)
