"""Forward transform functions for PyTorch UDCT implementation."""

from __future__ import annotations

import torch

from ._riesz import riesz_filters
from ._typing import MUDCTCoefficients, UDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, downsample, flip_fft_all_axes


def _process_wedge_real(
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    freq_band: torch.Tensor,
) -> torch.Tensor:
    """Process a single wedge for real transform mode."""
    # Clear the frequency band buffer for reuse
    freq_band.zero_()

    # Get the sparse window representation (indices and values)
    idx, val = window

    # Apply the window to the frequency domain
    idx_flat = idx.flatten()
    freq_band.flatten()[idx_flat] = image_frequency.flatten()[idx_flat] * val.flatten().to(freq_band.dtype)

    # Transform back to spatial domain using inverse FFT
    curvelet_band = torch.fft.ifftn(freq_band)

    # Downsample the curvelet band according to the decimation ratio
    coeff = downsample(curvelet_band, decimation_ratio)

    # Apply normalization factor
    coeff = coeff * torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    return coeff


def _process_wedge_complex(
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    parameters: ParamUDCT,
    flip_window: bool = False,
) -> torch.Tensor:
    """Process a single wedge for complex transform mode."""
    # Get the sparse window representation
    idx, val = window

    # Convert sparse window to dense
    subwindow = torch.zeros(parameters.shape, dtype=val.dtype, device=val.device)
    subwindow.flatten()[idx.flatten()] = val.flatten()

    # Optionally flip the window for negative frequency processing
    if flip_window:
        subwindow = flip_fft_all_axes(subwindow)

    # Apply window to frequency domain and transform to spatial domain
    band_filtered = torch.sqrt(torch.tensor(0.5)) * torch.fft.ifftn(
        image_frequency * subwindow.to(image_frequency.dtype)
    )

    # Downsample the curvelet band
    coeff = downsample(band_filtered, decimation_ratio)

    # Apply normalization factor
    coeff = coeff * torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    return coeff


def _apply_forward_transform_real(
    image: torch.Tensor,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> UDCTCoefficients:
    """Apply forward Uniform Discrete Curvelet Transform in real mode."""
    image_frequency = torch.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Allocate frequency_band once for reuse
    frequency_band = torch.zeros_like(image_frequency)

    # Low frequency band processing
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    frequency_band.flatten()[idx_flat] = image_frequency.flatten()[idx_flat] * val.flatten().to(complex_dtype)

    curvelet_band = torch.fft.ifftn(frequency_band)

    low_freq_coeff = downsample(curvelet_band, decimation_ratios[0][0])
    norm = torch.sqrt(
        torch.prod(
            torch.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2), dtype=torch.float64)
        )
    )
    low_freq_coeff = low_freq_coeff * norm

    # Build coefficients structure
    coefficients: UDCTCoefficients = [[[low_freq_coeff]]]

    for scale_idx in range(1, parameters.num_scales):
        scale_coeffs = []
        for direction_idx in range(len(windows[scale_idx])):
            direction_coeffs = []
            for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                window = windows[scale_idx][direction_idx][wedge_idx]
                if decimation_ratios[scale_idx].shape[0] == 1:
                    decimation_ratio = decimation_ratios[scale_idx][0, :]
                else:
                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                
                coeff = _process_wedge_real(
                    window,
                    decimation_ratio,
                    image_frequency,
                    frequency_band,
                )
                direction_coeffs.append(coeff)
            scale_coeffs.append(direction_coeffs)
        coefficients.append(scale_coeffs)

    return coefficients


def _apply_forward_transform_complex(
    image: torch.Tensor,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> UDCTCoefficients:
    """Apply forward Uniform Discrete Curvelet Transform in complex mode."""
    image_frequency = torch.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Low frequency band processing
    frequency_band = torch.zeros_like(image_frequency)
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    frequency_band.flatten()[idx_flat] = image_frequency.flatten()[idx_flat] * val.flatten().to(complex_dtype)

    curvelet_band = torch.fft.ifftn(frequency_band)

    low_freq_coeff = downsample(curvelet_band, decimation_ratios[0][0])
    norm = torch.sqrt(
        torch.prod(
            torch.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2), dtype=torch.float64)
        )
    )
    low_freq_coeff = low_freq_coeff * norm

    coefficients: UDCTCoefficients = [[[low_freq_coeff]]]

    for scale_idx in range(1, parameters.num_scales):
        scale_coeffs = []
        
        # Positive frequency bands (directions 0..dim-1)
        for direction_idx in range(parameters.ndim):
            direction_coeffs = []
            window_direction_idx = min(direction_idx, len(windows[scale_idx]) - 1)
            for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                if decimation_ratios[scale_idx].shape[0] == 1:
                    decimation_ratio = decimation_ratios[scale_idx][0, :]
                else:
                    decimation_ratio = decimation_ratios[scale_idx][window_direction_idx, :]
                
                coeff = _process_wedge_complex(
                    windows[scale_idx][window_direction_idx][wedge_idx],
                    decimation_ratio,
                    image_frequency,
                    parameters,
                    flip_window=False,
                )
                direction_coeffs.append(coeff)
            scale_coeffs.append(direction_coeffs)
        
        # Negative frequency bands (directions dim..2*dim-1)
        for direction_idx in range(parameters.ndim):
            direction_coeffs = []
            window_direction_idx = min(direction_idx, len(windows[scale_idx]) - 1)
            for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                if decimation_ratios[scale_idx].shape[0] == 1:
                    decimation_ratio = decimation_ratios[scale_idx][0, :]
                else:
                    decimation_ratio = decimation_ratios[scale_idx][window_direction_idx, :]
                
                coeff = _process_wedge_complex(
                    windows[scale_idx][window_direction_idx][wedge_idx],
                    decimation_ratio,
                    image_frequency,
                    parameters,
                    flip_window=True,
                )
                direction_coeffs.append(coeff)
            scale_coeffs.append(direction_coeffs)
        
        coefficients.append(scale_coeffs)

    return coefficients


def _process_wedge_monogenic(
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    riesz_filters_list: list[torch.Tensor],
    freq_band: torch.Tensor,
) -> list[torch.Tensor]:
    """Process a single wedge for monogenic transform."""
    # Scalar component
    freq_band.zero_()
    idx, val = window
    idx_flat = idx.flatten()
    freq_band.flatten()[idx_flat] = image_frequency.flatten()[idx_flat] * val.flatten().to(freq_band.dtype)
    curvelet_band_scalar = torch.fft.ifftn(freq_band)
    coeff_scalar = downsample(curvelet_band_scalar, decimation_ratio)
    coeff_scalar = coeff_scalar * torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    # Process all Riesz components
    riesz_coeffs: list[torch.Tensor] = []
    for riesz_filter in riesz_filters_list:
        freq_band.zero_()
        freq_band.flatten()[idx_flat] = (
            image_frequency.flatten()[idx_flat]
            * val.flatten().to(freq_band.dtype)
            * riesz_filter.flatten()[idx_flat]
        )
        curvelet_band_riesz = torch.fft.ifftn(freq_band)
        coeff_riesz = downsample(curvelet_band_riesz, decimation_ratio)
        coeff_riesz = coeff_riesz * torch.sqrt(2 * torch.prod(decimation_ratio.float()))
        riesz_coeffs.append(coeff_riesz.real)

    return [coeff_scalar, *riesz_coeffs]


def _apply_forward_transform_monogenic(
    image: torch.Tensor,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> MUDCTCoefficients:
    """Apply forward monogenic curvelet transform."""
    image_frequency = torch.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Compute Riesz filters once
    riesz_filters_list = riesz_filters(parameters.shape)

    frequency_band = torch.zeros_like(image_frequency)

    # Low frequency band processing
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    frequency_band.zero_()
    frequency_band.flatten()[idx_flat] = image_frequency.flatten()[idx_flat] * val.flatten().to(complex_dtype)

    curvelet_band_scalar = torch.fft.ifftn(frequency_band)
    low_freq_coeff_scalar = downsample(curvelet_band_scalar, decimation_ratios[0][0])
    norm = torch.sqrt(
        torch.prod(
            torch.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2), dtype=torch.float64)
        )
    )
    low_freq_coeff_scalar = low_freq_coeff_scalar * norm

    # Process all Riesz components for low frequency
    low_freq_riesz_coeffs: list[torch.Tensor] = []
    for riesz_filter in riesz_filters_list:
        frequency_band.zero_()
        frequency_band.flatten()[idx_flat] = (
            image_frequency.flatten()[idx_flat]
            * val.flatten().to(complex_dtype)
            * riesz_filter.flatten()[idx_flat]
        )
        curvelet_band_riesz = torch.fft.ifftn(frequency_band)
        low_freq_coeff_riesz = downsample(curvelet_band_riesz, decimation_ratios[0][0])
        low_freq_coeff_riesz = low_freq_coeff_riesz * norm
        low_freq_riesz_coeffs.append(low_freq_coeff_riesz.real)

    low_freq_coeff = [low_freq_coeff_scalar, *low_freq_riesz_coeffs]

    coefficients: MUDCTCoefficients = [[[low_freq_coeff]]]

    for scale_idx in range(1, parameters.num_scales):
        scale_coeffs = []
        for direction_idx in range(len(windows[scale_idx])):
            direction_coeffs = []
            for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                window = windows[scale_idx][direction_idx][wedge_idx]
                if decimation_ratios[scale_idx].shape[0] == 1:
                    decimation_ratio = decimation_ratios[scale_idx][0, :]
                else:
                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                
                coeffs = _process_wedge_monogenic(
                    window,
                    decimation_ratio,
                    image_frequency,
                    riesz_filters_list,
                    frequency_band,
                )
                direction_coeffs.append(coeffs)
            scale_coeffs.append(direction_coeffs)
        coefficients.append(scale_coeffs)

    return coefficients
