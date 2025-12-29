"""Main UDCT class for PyTorch implementation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ._backward_transform import _apply_backward_transform
from ._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_real,
)
from ._meyerwavelet import MeyerWavelet
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
    shape : tuple[int, ...]
        Shape of the input data.
    high_frequency_mode : str
        High frequency mode.
    transform_kind : str
        Type of transform being used ("real", "complex", or "monogenic").
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list
        Decimation ratios for each scale/direction.

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

        # Compute windows
        window_computer = UDCTWindow(self._parameters, high_frequency_mode)
        self._windows, self._decimation_ratios, self._indices = (
            window_computer.compute()
        )

        # Precompute MeyerWavelet for wavelet mode
        if high_frequency_mode == "wavelet":
            self._meyerwavelet = MeyerWavelet(
                shape=shape,
                num_scales=self._parameters.num_scales,
                radial_frequency_params=radial_frequency_params,
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
        max_wedges = angular_wedges_config.max().item()
        return min(0.25, 0.1 + 0.01 * max_wedges / min_dim)

    @staticmethod
    def _compute_from_angular_wedges_config(
        shape: tuple[int, ...],
        angular_wedges_config: torch.Tensor,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex", "monogenic"] = "real",
    ) -> "UDCT":
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
    ) -> "UDCT":
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
        # Dispatch based on transform_kind
        if self._transform_kind == "monogenic":
            return self._struct_monogenic(vector, template)
        elif self._transform_kind == "complex":
            return self._struct_complex(vector, template)
        else:  # real
            return self._struct_real(vector, template)

    def _struct_real(
        self, vector: torch.Tensor, template: UDCTCoefficients
    ) -> UDCTCoefficients:
        """Private method for real coefficient restructuring (no input validation)."""
        result: UDCTCoefficients = []
        offset = 0
        for scale_idx, scale in enumerate(template):
            scale_coeffs: list[list[torch.Tensor]] = []
            for direction_idx, direction in enumerate(scale):
                direction_coeffs: list[torch.Tensor] = []
                for wedge_coeff in direction:
                    size = wedge_coeff.numel()
                    coeff = vector[offset : offset + size].reshape(wedge_coeff.shape)
                    direction_coeffs.append(coeff.to(wedge_coeff.dtype))
                    offset += size
                scale_coeffs.append(direction_coeffs)
            result.append(scale_coeffs)
        return result

    def _struct_complex(
        self, vector: torch.Tensor, template: UDCTCoefficients
    ) -> UDCTCoefficients:
        """Private method for complex coefficient restructuring (no input validation)."""
        # Complex transform uses same structure as real
        return self._struct_real(vector, template)

    def _struct_monogenic(
        self, vector: torch.Tensor, template: UDCTCoefficients
    ) -> UDCTCoefficients:
        """Private method for monogenic coefficient restructuring (no input validation)."""
        # TODO: Implement monogenic restructuring for PyTorch
        # For now, raise NotImplementedError
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)

    def from_sparse(
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
        if self._transform_kind in ("real", "monogenic"):
            if image.is_complex():
                msg = (
                    f"{self._transform_kind.capitalize()} transform requires real-valued input. "
                    "Got complex tensor. Use transform_kind='complex' for complex inputs."
                )
                raise ValueError(msg)

        # Dispatch based on transform_kind
        if self._transform_kind == "real":
            return self._forward_real(image)
        elif self._transform_kind == "complex":
            return self._forward_complex(image)
        elif self._transform_kind == "monogenic":
            return self._forward_monogenic(image)
        else:
            msg = f"Invalid transform_kind: {self._transform_kind!r}"
            raise ValueError(msg)

    def _forward_real(self, image: torch.Tensor) -> UDCTCoefficients:
        """Private method for real forward transform (no input validation)."""
        return _apply_forward_transform_real(
            image, self._parameters, self._windows, self._decimation_ratios
        )

    def _forward_complex(self, image: torch.Tensor) -> UDCTCoefficients:
        """Private method for complex forward transform (no input validation)."""
        return _apply_forward_transform_complex(
            image, self._parameters, self._windows, self._decimation_ratios
        )

    def _forward_monogenic(self, image: torch.Tensor) -> UDCTCoefficients:
        """Private method for monogenic forward transform (no input validation)."""
        # TODO: Implement monogenic forward transform for PyTorch
        # For now, raise NotImplementedError
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
        elif self._transform_kind == "complex":
            return self._backward_complex(coefficients)
        elif self._transform_kind == "monogenic":
            return self._backward_monogenic(coefficients)
        else:
            msg = f"Invalid transform_kind: {self._transform_kind!r}"
            raise ValueError(msg)

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

    def _backward_monogenic(self, coefficients: UDCTCoefficients) -> torch.Tensor:
        """Private method for monogenic backward transform (no input validation)."""
        # TODO: Implement monogenic backward transform for PyTorch
        # For now, raise NotImplementedError
        msg = "Monogenic transform not yet implemented for PyTorch"
        raise NotImplementedError(msg)
