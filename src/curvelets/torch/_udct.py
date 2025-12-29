"""Main UDCT class for PyTorch implementation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ._backward_transform import _apply_backward_transform
from ._backward_transform_monogenic import _apply_backward_transform_monogenic
from ._forward_transform import (
    _apply_forward_transform_complex,
    _apply_forward_transform_monogenic,
    _apply_forward_transform_real,
)
from ._meyerwavelet import MeyerWavelet
from ._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows
from ._udct_windows import UDCTWindow
from ._utils import ParamUDCT


class UDCT:
    """
    Uniform Discrete Curvelet Transform.

    Implements forward and backward curvelet transforms in N dimensions,
    providing multi-scale, multi-directional representations of signals.

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
        if radial_frequency_params is None:
            radial_frequency_params = (
                np.pi / 3,
                2 * np.pi / 3,
                2 * np.pi / 3,
                4 * np.pi / 3,
            )

        self._parameters = ParamUDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
        )

        self._high_frequency_mode = high_frequency_mode
        self._use_complex_transform = use_complex_transform

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
        use_complex_transform: bool = False,
    ) -> "UDCT":
        """Create UDCT from angular wedges configuration."""
        return UDCT(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            high_frequency_mode=high_frequency_mode,
            use_complex_transform=use_complex_transform,
        )

    @staticmethod
    def _compute_from_num_scales(
        shape: tuple[int, ...],
        num_scales: int,
        base_wedges: int = 3,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        use_complex_transform: bool = False,
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
            use_complex_transform=use_complex_transform,
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

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients organized by scale, direction, and wedge.
        """
        if self._use_complex_transform:
            return _apply_forward_transform_complex(
                image, self._parameters, self._windows, self._decimation_ratios
            )
        return _apply_forward_transform_real(
            image, self._parameters, self._windows, self._decimation_ratios
        )

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
        return _apply_backward_transform(
            coefficients,
            self._parameters,
            self._windows,
            self._decimation_ratios,
            self._use_complex_transform,
        )

    def forward_monogenic(self, image: torch.Tensor) -> MUDCTCoefficients:
        """
        Apply forward monogenic curvelet transform.

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.

        Returns
        -------
        MUDCTCoefficients
            Monogenic curvelet coefficients.
        """
        return _apply_forward_transform_monogenic(
            image, self._parameters, self._windows, self._decimation_ratios
        )

    def backward_monogenic(
        self, coefficients: MUDCTCoefficients
    ) -> tuple[torch.Tensor, ...]:
        """
        Apply backward (inverse) monogenic curvelet transform.

        Parameters
        ----------
        coefficients : MUDCTCoefficients
            Monogenic curvelet coefficients from forward_monogenic.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of (scalar, riesz_1, riesz_2, ..., riesz_ndim) components.
        """
        return _apply_backward_transform_monogenic(
            coefficients, self._parameters, self._windows, self._decimation_ratios
        )

    def monogenic(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Compute monogenic transform (scalar + Riesz components).

        Parameters
        ----------
        image : torch.Tensor
            Input image with shape matching self.shape.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of (scalar, riesz_1, riesz_2, ..., riesz_ndim) components.
        """
        coefficients = self.forward_monogenic(image)
        return self.backward_monogenic(coefficients)
