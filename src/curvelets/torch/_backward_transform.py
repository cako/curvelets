"""Backward transform functions for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import torch

from ._typing import UDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, flip_fft_all_axes, upsample


def _process_wedge_backward_real(
    coefficient: torch.Tensor,
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
) -> torch.Tensor:
    """
    Process a single wedge for real backward transform mode.

    This function upsamples a coefficient, transforms it to frequency domain,
    applies the window, and returns the frequency-domain contribution.

    Parameters
    ----------
    coefficient : torch.Tensor
        Downsampled coefficient array for this wedge.
    window : tuple[torch.Tensor, torch.Tensor]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).

    Returns
    -------
    torch.Tensor
        Frequency-domain contribution as sparse array (only non-zero at window indices).
        Same shape as the full image size.

    Notes
    -----
    The contribution is sparse - only non-zero at the window indices. This allows
    efficient accumulation using sparse indexing in the real transform mode.
    """
    # Upsample coefficient to full size
    curvelet_band = upsample(coefficient, decimation_ratio)

    # Undo normalization
    curvelet_band = curvelet_band / torch.sqrt(2 * torch.prod(decimation_ratio.float()))

    # Transform to frequency domain
    curvelet_band = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band)  # pylint: disable=not-callable

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
    """
    Process a single wedge for complex backward transform mode.

    This function upsamples a coefficient, transforms it to frequency domain,
    applies the window (optionally flipped for negative frequencies), and returns
    the frequency-domain contribution with :math:`\\sqrt{0.5}` scaling.

    Parameters
    ----------
    coefficient : torch.Tensor
        Downsampled coefficient array for this wedge.
    window : tuple[torch.Tensor, torch.Tensor]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    parameters : ParamUDCT
        UDCT parameters containing size information.
    flip_window : bool, optional
        If True, flip the window for negative frequency processing.
        Default is False.

    Returns
    -------
    torch.Tensor
        Full frequency-domain contribution array with :math:`\\sqrt{0.5}` scaling applied.

    Notes
    -----
    The contribution is a full array (not sparse) to allow efficient accumulation
    in complex transform mode. The :math:`\\sqrt{0.5}` scaling accounts for the separation
    of positive and negative frequencies.
    """
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
    curvelet_band = torch.prod(decimation_ratio.float()) * torch.fft.fftn(curvelet_band)  # pylint: disable=not-callable

    # Apply window with sqrt(0.5) scaling for complex transform
    return (
        torch.sqrt(torch.tensor(0.5, device=curvelet_band.device))
        * curvelet_band
        * subwindow.to(curvelet_band)
    )


def _apply_backward_transform_real(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> torch.Tensor:
    """
    Apply backward Uniform Discrete Curvelet Transform in real mode.

    This function reconstructs a real-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The real transform mode processes combined positive/negative
    frequency bands.

    Parameters
    ----------
    coefficients : UDCTCoefficients
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
    parameters : ParamUDCT
        UDCT parameters containing transform configuration.
    windows : UDCTWindows
        Curvelet windows in sparse format, must match those used in
        forward transform.
    decimation_ratios : list[torch.Tensor]
        Decimation ratios for each scale and direction, must match those
        used in forward transform.

    Returns
    -------
    torch.Tensor
        Reconstructed real-valued image or volume with shape `parameters.shape`.

    Notes
    -----
    The real transform combines positive and negative frequencies, resulting
    in real-valued output. Contributions are accumulated using sparse indexing
    for efficiency.
    """
    # Determine dtype and device from coefficients
    complex_dtype = coefficients[0][0][0].dtype
    device = coefficients[0][0][0].device

    # Initialize frequency domain
    image_frequency = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        image_frequency_other_scales = torch.zeros(
            parameters.shape, dtype=complex_dtype, device=device
        )
        image_frequency_wavelet_scale = torch.zeros(
            parameters.shape, dtype=complex_dtype, device=device
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
    image_frequency_low = torch.zeros(
        parameters.shape, dtype=complex_dtype, device=device
    )
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(  # pylint: disable=not-callable
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

    return torch.fft.ifftn(image_frequency).real  # pylint: disable=not-callable


def _apply_backward_transform_complex(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> torch.Tensor:
    """
    Apply backward Uniform Discrete Curvelet Transform in complex mode.

    This function reconstructs a complex-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The complex transform mode processes separate positive and negative
    frequency bands.

    Parameters
    ----------
    coefficients : UDCTCoefficients
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
    parameters : ParamUDCT
        UDCT parameters containing transform configuration.
    windows : UDCTWindows
        Curvelet windows in sparse format, must match those used in
        forward transform.
    decimation_ratios : list[torch.Tensor]
        Decimation ratios for each scale and direction, must match those
        used in forward transform.

    Returns
    -------
    torch.Tensor
        Reconstructed complex-valued image or volume with shape `parameters.shape`.

    Notes
    -----
    The complex transform separates positive and negative frequencies, resulting
    in complex-valued output. Contributions are accumulated using full array
    operations for efficiency.
    """
    complex_dtype = coefficients[0][0][0].dtype
    device = coefficients[0][0][0].device

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:  # pylint: disable=too-many-nested-blocks
        image_frequency_other_scales = torch.zeros(
            parameters.shape, dtype=complex_dtype, device=device
        )
        image_frequency_wavelet_scale = torch.zeros(
            parameters.shape, dtype=complex_dtype, device=device
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
        image_frequency = torch.zeros(
            parameters.shape, dtype=complex_dtype, device=device
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
    image_frequency_low = torch.zeros(
        parameters.shape, dtype=complex_dtype, device=device
    )
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = torch.sqrt(torch.prod(decimation_ratio.float())) * torch.fft.fftn(  # pylint: disable=not-callable
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

    return torch.fft.ifftn(image_frequency)  # pylint: disable=not-callable


def _apply_backward_transform(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
    use_complex_transform: bool = False,
) -> torch.Tensor:
    """
    Apply backward Uniform Discrete Curvelet Transform (reconstruction).

    This function reconstructs an image or volume from curvelet coefficients
    by upsampling, applying frequency-domain windows, and combining all bands.
    The transform can operate in two modes: real transform (default) or complex
    transform, matching the mode used in the forward transform.

    Parameters
    ----------
    coefficients : UDCTCoefficients
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands
          * Real mode: ndim directions per scale
          * Complex mode: 2*ndim directions per scale
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - num_scales : int
            Number of resolution scales
        - ndim : int
            Number of dimensions of the transform
        - shape : tuple[int, ...]
            Shape of the output data
    windows : UDCTWindows
        Curvelet windows in sparse format, must match those used in
        forward transform. Structure:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[torch.Tensor]
        Decimation ratios for each scale and direction, must match those
        used in forward transform. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0
    use_complex_transform : bool, optional
        Transform mode flag, must match forward transform:
        - False (default): Real transform mode. Reconstructs from combined
          positive/negative frequency bands. Returns real-valued output.
        - True: Complex transform mode. Reconstructs from separate positive
          and negative frequency bands. Directions 0..dim-1 are positive
          frequencies, directions dim..2*dim-1 are negative frequencies.
          Returns complex-valued output. Required for complex-valued inputs.

    Returns
    -------
    torch.Tensor
        Reconstructed image or volume with shape `parameters.shape`.
        - Real mode: Returns real-valued tensor (dtype matches input real part)
        - Complex mode: Returns complex-valued tensor

    Notes
    -----
    The backward transform process:

    1. **Upsampling**: Each coefficient band is upsampled to full size
       according to its decimation ratio.

    2. **FFT**: Each upsampled band is transformed to frequency domain.

    3. **Window application**: Frequency-domain windows are applied to
       each band. Windows are stored in sparse format for efficiency.

    4. **Combination**: All frequency bands are combined:
       - High-frequency bands are multiplied by 2 (to account for
         combined +/- frequencies in real mode, or separate processing
         in complex mode)
       - Low-frequency band is added separately
       - Final frequency-domain representation is obtained

    5. **IFFT**: Combined frequency representation is transformed back
       to spatial domain.

    For complex transform mode, positive and negative frequencies are
    processed separately and combined. The negative frequency windows
    are obtained by flipping the positive frequency windows using
    `flip_fft_all_axes`.

    The transform provides perfect reconstruction when used with the
    corresponding forward transform, due to the tight frame property
    of the curvelet windows.

    The normalization factors ensure energy preservation: coefficients
    are scaled by :math:`\\sqrt{2 \\prod d}` where :math:`d` is the decimation ratio in forward transform,
    and divided by the same factor in backward transform.
    """
    if use_complex_transform:
        return _apply_backward_transform_complex(
            coefficients, parameters, windows, decimation_ratios
        )
    return _apply_backward_transform_real(
        coefficients, parameters, windows, decimation_ratios
    )
