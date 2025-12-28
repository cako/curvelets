"""Backward monogenic transform functions for PyTorch UDCT implementation."""

from __future__ import annotations

import torch

from ._typing import MUDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, upsample


def _process_wedge_backward_monogenic(
    coefficients: list[torch.Tensor],
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
) -> list[torch.Tensor]:
    """Process a single wedge for monogenic backward transform."""
    coeff_scalar = coefficients[0]
    coeff_riesz_list = coefficients[1:]

    # Upsample scalar coefficient
    curvelet_band_scalar = upsample(coeff_scalar, decimation_ratio)

    # Undo normalization
    norm_factor = torch.sqrt(2 * torch.prod(decimation_ratio.float()))
    curvelet_band_scalar = curvelet_band_scalar / norm_factor

    # Transform to frequency domain
    curvelet_freq_scalar = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band_scalar)

    # Get window indices and values
    idx, val = window
    idx_flat = idx.flatten()
    window_values = val.flatten().to(curvelet_freq_scalar.dtype)

    # Initialize contribution for scalar
    contribution_scalar = torch.zeros(curvelet_freq_scalar.shape, dtype=curvelet_freq_scalar.dtype)
    contribution_scalar.flatten()[idx_flat] = curvelet_freq_scalar.flatten()[idx_flat] * window_values

    # Process all Riesz components
    contributions = [contribution_scalar]
    for coeff_riesz_k in coeff_riesz_list:
        curvelet_band_riesz = upsample(coeff_riesz_k.to(curvelet_freq_scalar.dtype), decimation_ratio)
        curvelet_band_riesz = curvelet_band_riesz / norm_factor
        curvelet_freq_riesz = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band_riesz)
        contribution_riesz = torch.zeros(curvelet_freq_riesz.shape, dtype=curvelet_freq_riesz.dtype)
        contribution_riesz.flatten()[idx_flat] = curvelet_freq_riesz.flatten()[idx_flat] * window_values
        contributions.append(contribution_riesz)

    return contributions


def _apply_backward_transform_monogenic(
    coefficients: MUDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Apply backward monogenic curvelet transform."""
    # Determine dtype from coefficients
    scalar_coeff = coefficients[0][0][0][0]
    complex_dtype = torch.complex64 if scalar_coeff.dtype == torch.float32 else torch.complex128
    real_dtype = torch.float32 if scalar_coeff.dtype == torch.float32 else torch.float64

    # Determine number of components
    num_components = len(coefficients[0][0][0])

    # Initialize frequency domain arrays
    image_frequencies = [
        torch.zeros(parameters.shape, dtype=complex_dtype) for _ in range(num_components)
    ]

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        image_frequencies_wavelet = [
            torch.zeros(parameters.shape, dtype=complex_dtype)
            for _ in range(num_components)
        ]
        image_frequencies_other = [
            torch.zeros(parameters.shape, dtype=complex_dtype)
            for _ in range(num_components)
        ]

        for scale_idx in range(1, parameters.num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    window = windows[scale_idx][direction_idx][wedge_idx]
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]

                    coeffs = coefficients[scale_idx][direction_idx][wedge_idx]
                    contributions = _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                    )

                    idx, _ = window
                    idx_flat = idx.flatten()
                    if scale_idx == highest_scale_idx:
                        for comp_idx, contrib in enumerate(contributions):
                            image_frequencies_wavelet[comp_idx].flatten()[idx_flat] += contrib.flatten()[idx_flat]
                    else:
                        for comp_idx, contrib in enumerate(contributions):
                            image_frequencies_other[comp_idx].flatten()[idx_flat] += contrib.flatten()[idx_flat]

        for comp_idx in range(num_components):
            image_frequencies[comp_idx] = (
                2 * image_frequencies_other[comp_idx]
                + image_frequencies_wavelet[comp_idx]
            )
    else:
        for scale_idx in range(1, parameters.num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    window = windows[scale_idx][direction_idx][wedge_idx]
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]

                    coeffs = coefficients[scale_idx][direction_idx][wedge_idx]
                    contributions = _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                    )

                    idx, _ = window
                    idx_flat = idx.flatten()
                    for comp_idx, contrib in enumerate(contributions):
                        image_frequencies[comp_idx].flatten()[idx_flat] += contrib.flatten()[idx_flat]

        for comp_idx in range(num_components):
            image_frequencies[comp_idx] = image_frequencies[comp_idx] * 2

    # Process low-frequency band
    decimation_ratio = decimation_ratios[0][0]
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    window_values = val.flatten().to(complex_dtype)

    low_coeffs = coefficients[0][0][0]
    low_coeff_scalar = low_coeffs[0]

    curvelet_band_scalar = upsample(low_coeff_scalar, decimation_ratio)
    curvelet_freq_scalar = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(curvelet_band_scalar)
    image_frequencies[0].flatten()[idx_flat] += curvelet_freq_scalar.flatten()[idx_flat] * window_values

    for comp_idx in range(1, num_components):
        low_coeff_riesz = low_coeffs[comp_idx]
        curvelet_band_riesz = upsample(low_coeff_riesz.to(complex_dtype), decimation_ratio)
        curvelet_freq_riesz = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(curvelet_band_riesz)
        image_frequencies[comp_idx].flatten()[idx_flat] += curvelet_freq_riesz.flatten()[idx_flat] * window_values

    # Transform back to spatial domain
    results = []
    scalar = torch.fft.ifftn(image_frequencies[0]).real.to(real_dtype)
    results.append(scalar)

    for comp_idx in range(1, num_components):
        riesz_k = -torch.fft.ifftn(image_frequencies[comp_idx]).real.to(real_dtype)
        results.append(riesz_k)

    return tuple(results)
