"""Main UDCT class for PyTorch implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

from math import prod
from typing import Literal

import numpy as np
import torch

from ._backward_transform import _apply_backward_transform
from ._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_real,
)
from ._typing import UDCTCoefficients, UDCTWindows
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT


class UDCT:
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    angular_wedges_config : torch.Tensor
        Configuration array specifying the number of angular wedges per scale
        and dimension. Shape is (num_scales - 1, dimension), where num_scales
        includes the lowpass scale.
    window_overlap : float, optional
        Window overlap parameter controlling the smoothness of window transitions.
        Default is 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency parameters defining the frequency bands.
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-6.
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
    use_complex_transform : bool, optional
        Deprecated. Use transform_kind instead. Default is None.

    Attributes
    ----------
    high_frequency_mode : str
        High frequency mode.
    transform_kind : str
        Type of transform being used ("real", "complex", or "monogenic").

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import UDCT
    >>> # Create a 2D transform
    >>> transform = UDCT(shape=(64, 64), angular_wedges_config=torch.tensor([[3, 3], [6, 6]]))
    >>> data = torch.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> torch.allclose(data, recon, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor,
        window_overlap: float = 0.15,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-6,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
    ) -> None:
        if radial_frequency_params is None:
            radial_frequency_params = (
                np.pi / 3,
                2 * np.pi / 3,
                2 * np.pi / 3,
                4 * np.pi / 3,
            )

        # Validate transform_kind
        if transform_kind not in ("real", "complex", "monogenic"):
            msg = f"transform_kind must be 'real', 'complex', or 'monogenic', got {transform_kind!r}"
            raise ValueError(msg)

        self._parameters = ParamUDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
        )

        self._high_frequency_mode = high_frequency_mode
        self._transform_kind = transform_kind
        self._coefficient_dtype: torch.dtype | None = None

        # Compute windows
        window_computer = UDCTWindow(self._parameters, high_frequency_mode)
        self._windows, self._decimation_ratios, self._indices = (
            window_computer.compute()
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the transform."""
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
        """Curvelet windows in sparse format."""
        return self._windows

    @property
    def decimation_ratios(self) -> list[torch.Tensor]:
        """Decimation ratios for each scale."""
        return self._decimation_ratios

    @staticmethod
    def _compute_optimal_window_overlap(
        shape: tuple[int, ...], angular_wedges_config: torch.Tensor
    ) -> float:
        """Compute optimal window overlap for given configuration."""
        # Simple heuristic based on shape and config
        min_dim = min(shape)
        max_wedges = float(angular_wedges_config.max().item())
        return min(0.25, 0.1 + 0.01 * max_wedges / min_dim)

    @staticmethod
    def _compute_from_angular_wedges_config(
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
    ) -> UDCT:
        """Create UDCT from angular wedges configuration."""
        return UDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_kind,
        )

    @staticmethod
    def _compute_from_num_scales(
        shape: tuple[int, ...],
        num_scales: int,
        base_wedges: int = 3,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
    ) -> UDCT:
        """Create UDCT from number of scales."""
        ndim = len(shape)
        # Create config with exponentially increasing wedges per scale
        config_list = []
        for scale_idx in range(num_scales - 1):
            wedges = base_wedges * (2**scale_idx)
            config_list.append([wedges] * ndim)
        angular_wedges_config = torch.tensor(config_list, dtype=torch.int64)
        return UDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_kind,
        )

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
        self, windows: UDCTWindows | None = None
    ) -> list[list[list[torch.Tensor]]]:
        """
        Convert sparse windows to dense format.

        Parameters
        ----------
        windows : UDCTWindows, optional
            Sparse windows to convert. Uses self.windows if not provided.

        Returns
        -------
        list[list[list[torch.Tensor]]]
            Dense window arrays.
        """
        if windows is None:
            windows = self._windows

        dense_windows: list[list[list[torch.Tensor]]] = []
        for scale in windows:
            scale_dense: list[list[torch.Tensor]] = []
            for direction in scale:
                direction_dense: list[torch.Tensor] = []
                for idx, val in direction:
                    dense = torch.zeros(self.shape, dtype=val.dtype, device=val.device)
                    dense.flatten()[idx.flatten()] = val.flatten()
                    direction_dense.append(dense)
                scale_dense.append(direction_dense)
            dense_windows.append(scale_dense)
        return dense_windows

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
