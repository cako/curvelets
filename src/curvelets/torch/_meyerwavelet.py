"""Meyer wavelet transform for PyTorch UDCT implementation."""

from __future__ import annotations

import numpy as np
import torch


class MeyerWavelet:
    """
    Multi-dimensional Meyer wavelet transform.

    Implements forward and backward Meyer wavelet transforms in multiple
    dimensions, using separable 1D wavelet transforms along each axis.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    num_scales : int
        Number of decomposition scales.
    radial_frequency_params : tuple[float, float, float, float], optional
        Four parameters defining radial frequency bands for Meyer wavelet
        decomposition: (transition_start, plateau_start, plateau_end, transition_end).
        These define the frequency ranges for the bandpass filters.
        Default is (:math:`\\pi/3`, :math:`2\\pi/3`, :math:`2\\pi/3`, :math:`4\\pi/3`).

    Examples
    --------
    >>> import torch
    >>> from curvelets.torch import MeyerWavelet
    >>> wavelet = MeyerWavelet(shape=(64, 64), num_scales=3)
    >>> signal = torch.randn(64, 64)
    >>> coefficients = wavelet.forward(signal)
    >>> len(coefficients)  # Number of scales
    3
    >>> reconstructed = wavelet.backward(coefficients)
    >>> torch.allclose(signal, reconstructed.real, atol=1e-10)
    True

    Notes
    -----
    This implementation uses separable 1D wavelet transforms along each axis,
    providing multi-scale, multi-dimensional decomposition of signals.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        num_scales: int,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.shape = shape
        self.num_scales = num_scales
        self.ndim = len(shape)

        if radial_frequency_params is None:
            radial_frequency_params = (
                np.pi / 3,
                2 * np.pi / 3,
                2 * np.pi / 3,
                4 * np.pi / 3,
            )
        self.radial_frequency_params = radial_frequency_params

        # Precompute 1D filters for each dimension
        self._filters: dict[int, dict[str, list[torch.Tensor]]] = {}
        for dim_idx, size in enumerate(shape):
            self._filters[dim_idx] = self._compute_single_filter(size)

    def _compute_single_filter(self, size: int) -> dict[str, list[torch.Tensor]]:
        """Compute Meyer wavelet filters for a single dimension."""
        # Use numpy for polyval since PyTorch doesn't have it
        frequency_grid = np.linspace(-1.5 * np.pi, 0.5 * np.pi, size + 1)[:-1]
        poly_coeffs = np.array([-20.0, 70.0, -84.0, 35.0, 0.0, 0.0, 0.0, 0.0])

        lowpass_filters: list[torch.Tensor] = []
        highpass_filters: list[torch.Tensor] = []

        for scale_idx in range(self.num_scales):
            # Compute radial boundaries for this scale
            scale_factor = 2 ** (self.num_scales - 1 - scale_idx)
            r0 = self.radial_frequency_params[0] / scale_factor
            r1 = self.radial_frequency_params[1] / scale_factor

            # Lowpass filter
            lowpass = np.zeros(size, dtype=np.float64)
            rising_mask = (np.abs(frequency_grid) >= -2) & (
                np.abs(frequency_grid) <= r1
            )
            if np.any(rising_mask):
                normalized_freq = (np.abs(frequency_grid[rising_mask]) + 2) / (r1 + 2)
                lowpass[rising_mask] = np.polyval(poly_coeffs, normalized_freq)
            lowpass[np.abs(frequency_grid) <= r0] = 1.0

            # Highpass filter
            highpass = np.zeros(size, dtype=np.float64)
            if scale_idx < self.num_scales - 1:
                next_scale_factor = 2 ** (self.num_scales - 2 - scale_idx)
                next_r0 = self.radial_frequency_params[0] / next_scale_factor
                next_r1 = self.radial_frequency_params[1] / next_scale_factor

                falling_mask = (np.abs(frequency_grid) >= next_r0) & (
                    np.abs(frequency_grid) <= next_r1
                )
                if np.any(falling_mask):
                    normalized_freq = (
                        np.abs(frequency_grid[falling_mask]) - next_r0
                    ) / (next_r1 - next_r0)
                    highpass[falling_mask] = np.polyval(poly_coeffs, normalized_freq)
                highpass[np.abs(frequency_grid) < next_r0] = 1.0

            lowpass_filters.append(torch.from_numpy(lowpass))
            highpass_filters.append(torch.from_numpy(highpass))

        return {"lowpass": lowpass_filters, "highpass": highpass_filters}

    def _forward_transform_1d(
        self,
        data_freq: torch.Tensor,
        dim_idx: int,
        scale_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D forward Meyer wavelet transform along specified dimension."""
        lowpass_filter = self._filters[dim_idx]["lowpass"][scale_idx].to(
            data_freq.device
        )
        highpass_filter = self._filters[dim_idx]["highpass"][scale_idx].to(
            data_freq.device
        )

        # Reshape filter for broadcasting
        filter_shape = [1] * data_freq.ndim
        filter_shape[dim_idx] = -1
        lowpass_filter = lowpass_filter.reshape(filter_shape)
        highpass_filter = highpass_filter.reshape(filter_shape)

        # Apply filters
        lowpass_band = data_freq * lowpass_filter.to(data_freq.dtype)
        highpass_band = data_freq * highpass_filter.to(data_freq.dtype)

        return lowpass_band, highpass_band

    def _inverse_transform_1d(
        self,
        lowpass_band: torch.Tensor,
        highpass_band: torch.Tensor,
        dim_idx: int,
        scale_idx: int,
    ) -> torch.Tensor:
        """Apply 1D inverse Meyer wavelet transform along specified dimension."""
        lowpass_filter = self._filters[dim_idx]["lowpass"][scale_idx].to(
            lowpass_band.device
        )
        highpass_filter = self._filters[dim_idx]["highpass"][scale_idx].to(
            lowpass_band.device
        )

        # Reshape filter for broadcasting
        filter_shape = [1] * lowpass_band.ndim
        filter_shape[dim_idx] = -1
        lowpass_filter = lowpass_filter.reshape(filter_shape)
        highpass_filter = highpass_filter.reshape(filter_shape)

        # Combine bands
        return lowpass_band * lowpass_filter.to(
            lowpass_band.dtype
        ) + highpass_band * highpass_filter.to(highpass_band.dtype)

    def forward(self, data: torch.Tensor) -> list[list[torch.Tensor]]:
        """
        Apply forward Meyer wavelet transform.

        Parameters
        ----------
        data : torch.Tensor
            Input data with shape matching self.shape.

        Returns
        -------
        list[list[torch.Tensor]]
            Wavelet coefficients organized as:
            coefficients[scale][band] where band indices encode
            dimension-wise decomposition.
        """
        # Transform to frequency domain
        data_freq = torch.fft.fftn(data)

        # Initialize coefficients
        coefficients: list[list[torch.Tensor]] = []

        current_freq = data_freq
        for scale_idx in range(self.num_scales - 1):
            scale_coeffs: list[torch.Tensor] = []

            # Process each dimension
            for dim_idx in range(self.ndim):
                lowpass_band, highpass_band = self._forward_transform_1d(
                    current_freq, dim_idx, scale_idx
                )
                # Store highpass for this dimension
                scale_coeffs.append(torch.fft.ifftn(highpass_band))
                current_freq = lowpass_band

            coefficients.append(scale_coeffs)

        # Store lowpass approximation at final scale
        final_coeffs = [torch.fft.ifftn(current_freq)]
        coefficients.append(final_coeffs)

        return coefficients

    def backward(self, coefficients: list[list[torch.Tensor]]) -> torch.Tensor:
        """
        Apply backward (inverse) Meyer wavelet transform.

        Parameters
        ----------
        coefficients : list[list[torch.Tensor]]
            Wavelet coefficients from forward transform.

        Returns
        -------
        torch.Tensor
            Reconstructed data with shape self.shape.
        """
        # Start with lowest scale approximation
        current_freq = torch.fft.fftn(coefficients[-1][0])

        # Reconstruct from coarse to fine
        for scale_idx in range(self.num_scales - 2, -1, -1):
            for dim_idx in range(self.ndim - 1, -1, -1):
                highpass_band = torch.fft.fftn(coefficients[scale_idx][dim_idx])
                current_freq = self._inverse_transform_1d(
                    current_freq, highpass_band, dim_idx, scale_idx
                )

        return torch.fft.ifftn(current_freq).real
