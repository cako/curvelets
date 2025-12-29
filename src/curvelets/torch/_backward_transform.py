"""Backward transform functions for PyTorch UDCT implementation."""

from __future__ import annotations

import torch

from ._typing import UDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, flip_fft_all_axes, upsample


def _process_wedge_backward_real(
    coefficient: torch.Tensor,
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
) -> torch.Tensor:
    """Process a single wedge for real backward transform mode."""
    # Upsample coefficient to full size
    curvelet_band = upsample(coefficient, decimation_ratio)

    # Undo normalization
    curvelet_band = curvelet_band / torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    # Transform to frequency domain
    curvelet_band = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band)

    # Get window indices and values
    idx, val = window
    idx_flat = idx.flatten()

    # Create sparse contribution array
    contribution = torch.zeros(
        curvelet_band.shape, dtype=curvelet_band.dtype, device=curvelet_band.device
    )
    contribution.flatten()[idx_flat] = curvelet_band.flatten()[
        idx_flat
    ] * val.flatten().to(curvelet_band.dtype)

    return contribution


def _process_wedge_backward_complex(
    coefficient: torch.Tensor,
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    parameters: ParamUDCT,
    flip_window: bool = False,
) -> torch.Tensor:
    """Process a single wedge for complex backward transform mode."""
    # Get window indices and values
    idx, val = window

    # Convert sparse window to dense
    subwindow = torch.zeros(parameters.shape, dtype=val.dtype, device=val.device)
    subwindow.flatten()[idx.flatten()] = val.flatten()

    # Optionally flip the window for negative frequency processing
    if flip_window:
        subwindow = flip_fft_all_axes(subwindow)

    # Upsample coefficient to full size
    curvelet_band = upsample(coefficient, decimation_ratio)

    # Undo normalization
    curvelet_band = curvelet_band / torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    # Transform to frequency domain
    curvelet_band = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band)

    # Apply window with sqrt(0.5) scaling for complex transform
    return (
        torch.sqrt(torch.tensor(0.5, device=curvelet_band.device))
        * curvelet_band
        * subwindow.to(curvelet_band.dtype)
    )


def _apply_backward_transform_real(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> torch.Tensor:
    """Apply backward Uniform Discrete Curvelet Transform in real mode."""
    # Determine dtype and device from coefficients
    complex_dtype = coefficients[0][0][0].dtype
    device = coefficients[0][0][0].device

    # Initialize frequency domain
    image_frequency = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        image_frequency_other_scales = torch.zeros(
            parameters.shape, dtype=complex_dtype
        )
        image_frequency_wavelet_scale = torch.zeros(
            parameters.shape, dtype=complex_dtype
        )

        for scale_idx in range(1, parameters.num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    window = windows[scale_idx][direction_idx][wedge_idx]
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            direction_idx, :
                        ]
                    contribution = _process_wedge_backward_real(
                        coefficients[scale_idx][direction_idx][wedge_idx],
                        window,
                        decimation_ratio,
                    )
                    idx, _ = window
                    idx_flat = idx.flatten()
                    if scale_idx == highest_scale_idx:
                        image_frequency_wavelet_scale.flatten()[idx_flat] += (
                            contribution.flatten()[idx_flat]
                        )
                    else:
                        image_frequency_other_scales.flatten()[idx_flat] += (
                            contribution.flatten()[idx_flat]
                        )
    else:
        for scale_idx in range(1, parameters.num_scales):
            for direction_idx in range(len(windows[scale_idx])):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    window = windows[scale_idx][direction_idx][wedge_idx]
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            direction_idx, :
                        ]
                    contribution = _process_wedge_backward_real(
                        coefficients[scale_idx][direction_idx][wedge_idx],
                        window,
                        decimation_ratio,
                    )
                    idx, _ = window
                    idx_flat = idx.flatten()
                    image_frequency.flatten()[idx_flat] += contribution.flatten()[
                        idx_flat
                    ]

    # Process low-frequency band
    image_frequency_low = torch.zeros(parameters.shape, dtype=complex_dtype)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(
        curvelet_band
    )
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    image_frequency_low.flatten()[idx_flat] += curvelet_band.flatten()[
        idx_flat
    ] * val.flatten().to(complex_dtype)

    # Combine
    if is_wavelet_mode_highest_scale:
        image_frequency = (
            2 * image_frequency_other_scales
            + image_frequency_wavelet_scale
            + image_frequency_low
        )
    else:
        image_frequency = 2 * image_frequency + image_frequency_low

    return torch.fft.ifftn(image_frequency).real


def _apply_backward_transform_complex(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> torch.Tensor:
    """Apply backward Uniform Discrete Curvelet Transform in complex mode."""
    complex_dtype = coefficients[0][0][0].dtype

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        image_frequency_other_scales = torch.zeros(
            parameters.shape, dtype=complex_dtype
        )
        image_frequency_wavelet_scale = torch.zeros(
            parameters.shape, dtype=complex_dtype
        )

        # Process positive frequency bands
        for scale_idx in range(1, parameters.num_scales):
            num_window_directions = len(windows[scale_idx])
            for direction_idx in range(parameters.ndim):
                window_direction_idx = min(direction_idx, num_window_directions - 1)
                for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            window_direction_idx, :
                        ]
                    contribution = _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx][wedge_idx],
                        windows[scale_idx][window_direction_idx][wedge_idx],
                        decimation_ratio,
                        parameters,
                        flip_window=False,
                    )
                    if scale_idx == highest_scale_idx:
                        if direction_idx == 0:
                            image_frequency_wavelet_scale += contribution
                    else:
                        image_frequency_other_scales += contribution

        # Process negative frequency bands
        for scale_idx in range(1, parameters.num_scales):
            num_window_directions = len(windows[scale_idx])
            for direction_idx in range(parameters.ndim):
                window_direction_idx = min(direction_idx, num_window_directions - 1)
                for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            window_direction_idx, :
                        ]
                    contribution = _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx + parameters.ndim][
                            wedge_idx
                        ],
                        windows[scale_idx][window_direction_idx][wedge_idx],
                        decimation_ratio,
                        parameters,
                        flip_window=True,
                    )
                    if scale_idx == highest_scale_idx:
                        if direction_idx == 0:
                            image_frequency_wavelet_scale += contribution
                    else:
                        image_frequency_other_scales += contribution
    else:
        image_frequency = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)

        # Process positive frequency bands
        for scale_idx in range(1, parameters.num_scales):
            num_window_directions = len(windows[scale_idx])
            for direction_idx in range(parameters.ndim):
                window_direction_idx = min(direction_idx, num_window_directions - 1)
                for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            window_direction_idx, :
                        ]
                    contribution = _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx][wedge_idx],
                        windows[scale_idx][window_direction_idx][wedge_idx],
                        decimation_ratio,
                        parameters,
                        flip_window=False,
                    )
                    image_frequency += contribution

        # Process negative frequency bands
        for scale_idx in range(1, parameters.num_scales):
            num_window_directions = len(windows[scale_idx])
            for direction_idx in range(parameters.ndim):
                window_direction_idx = min(direction_idx, num_window_directions - 1)
                for wedge_idx in range(len(windows[scale_idx][window_direction_idx])):
                    if decimation_ratios[scale_idx].shape[0] == 1:
                        decimation_ratio = decimation_ratios[scale_idx][0, :]
                    else:
                        decimation_ratio = decimation_ratios[scale_idx][
                            window_direction_idx, :
                        ]
                    contribution = _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx + parameters.ndim][
                            wedge_idx
                        ],
                        windows[scale_idx][window_direction_idx][wedge_idx],
                        decimation_ratio,
                        parameters,
                        flip_window=True,
                    )
                    image_frequency += contribution

    # Process low-frequency band
    image_frequency_low = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(
        curvelet_band
    )
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    image_frequency_low.flatten()[idx_flat] += curvelet_band.flatten()[
        idx_flat
    ] * val.flatten().to(complex_dtype)

    # Combine
    if is_wavelet_mode_highest_scale:
        image_frequency = (
            2 * image_frequency_other_scales
            + image_frequency_wavelet_scale
            + image_frequency_low
        )
    else:
        image_frequency = 2 * image_frequency + image_frequency_low

    return torch.fft.ifftn(image_frequency)


def _apply_backward_transform(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
    use_complex_transform: bool = False,
) -> torch.Tensor:
    """Apply backward Uniform Discrete Curvelet Transform (reconstruction)."""
    if use_complex_transform:
        return _apply_backward_transform_complex(
            coefficients, parameters, windows, decimation_ratios
        )
    return _apply_backward_transform_real(
        coefficients, parameters, windows, decimation_ratios
    )
