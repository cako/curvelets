"""Forward transform functions for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import torch

from ._typing import UDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, downsample, flip_fft_all_axes


def _process_wedge_real(
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    freq_band: torch.Tensor,
) -> torch.Tensor:
    """
    Process a single wedge for real transform mode.

    This function applies a frequency-domain window to extract a specific
    curvelet band, transforms it to spatial domain, downsamples it, and applies
    normalization.

    Parameters
    ----------
    window : tuple[torch.Tensor, torch.Tensor]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Input image in frequency domain (from FFT).
    freq_band : torch.Tensor
        Reusable frequency band buffer (will be cleared and filled).

    Returns
    -------
    torch.Tensor
        Downsampled and normalized coefficient array for this wedge.

    Notes
    -----
    The real transform combines positive and negative frequencies, so no
    :math:`\\sqrt{0.5}` scaling is applied. The normalization factor ensures proper
    energy preservation.
    """
    # Clear the frequency band buffer for reuse
    freq_band.zero_()

    # Get the sparse window representation (indices and values)
    idx, val = window

    # Apply the window to the frequency domain
    idx_flat = idx.flatten()
    freq_band.flatten()[idx_flat] = image_frequency.flatten()[
        idx_flat
    ] * val.flatten().to(freq_band.dtype)

    # Transform back to spatial domain using inverse FFT
    curvelet_band = torch.fft.ifftn(freq_band)  # pylint: disable=not-callable

    # Downsample the curvelet band according to the decimation ratio
    coeff = downsample(curvelet_band, decimation_ratio)

    # Apply normalization factor
    return coeff * torch.sqrt(2 * torch.prod(decimation_ratio.float()))


def _process_wedge_complex(
    window: tuple[torch.Tensor, torch.Tensor],
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    parameters: ParamUDCT,
    flip_window: bool = False,
) -> torch.Tensor:
    """
    Process a single wedge for complex transform mode.

    This function applies a frequency-domain window (optionally flipped for
    negative frequencies) to extract a specific curvelet band, transforms it
    to spatial domain with :math:`\\sqrt{0.5}` scaling, downsamples it, and applies
    normalization.

    Parameters
    ----------
    window : tuple[torch.Tensor, torch.Tensor]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Input image in frequency domain (from FFT).
    parameters : ParamUDCT
        UDCT parameters containing size information.
    flip_window : bool, optional
        If True, flip the window for negative frequency processing.
        Default is False.

    Returns
    -------
    torch.Tensor
        Downsampled and normalized coefficient array for this wedge.

    Notes
    -----
    The complex transform separates positive and negative frequencies, so
    :math:`\\sqrt{0.5}` scaling is applied to each band. The normalization factor ensures
    proper energy preservation.
    """
    # Get the sparse window representation
    idx, val = window

    # Convert sparse window to dense
    subwindow = torch.zeros(parameters.shape, dtype=val.dtype, device=val.device)
    subwindow.flatten()[idx.flatten()] = val.flatten()

    # Optionally flip the window for negative frequency processing
    if flip_window:
        subwindow = flip_fft_all_axes(subwindow)

    # Apply window to frequency domain and transform to spatial domain
    band_filtered = torch.sqrt(
        torch.tensor(0.5, device=image_frequency.device)
    ) * torch.fft.ifftn(image_frequency * subwindow.to(image_frequency))  # pylint: disable=not-callable

    # Downsample the curvelet band
    coeff = downsample(band_filtered, decimation_ratio)

    # Apply normalization factor
    return coeff * torch.sqrt(2 * torch.prod(decimation_ratio.float()))


def _apply_forward_transform_real(
    image: torch.Tensor,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[torch.Tensor],
) -> UDCTCoefficients:
    """
    Apply forward Uniform Discrete Curvelet Transform in real mode.

    This function decomposes an input image or volume into real-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Each
    curvelet band captures both positive and negative frequencies combined.

    Parameters
    ----------
    image : torch.Tensor
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be real-valued (floating point dtype).
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - num_scales : int
            Total number of scales (including lowpass scale)
        - ndim : int
            Number of dimensions of the transform
        - shape : tuple[int, ...]
            Shape of the input data
    windows : UDCTWindows
        Curvelet windows in sparse format, typically computed by
        `_udct_windows`. Structure is:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[torch.Tensor]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
        Each coefficient array has shape determined by decimation ratios.
        Coefficients are complex dtype matching the complex version of input dtype:
        - torch.float32 input -> torch.complex64 coefficients
        - torch.float64 input -> torch.complex128 coefficients

    Notes
    -----
    The real transform combines positive and negative frequencies, resulting
    in real-valued coefficients. This is suitable for real-valued inputs and
    provides a more compact representation.
    """
    image_frequency = torch.fft.fftn(image)  # pylint: disable=not-callable
    complex_dtype = image_frequency.dtype

    # Allocate frequency_band once for reuse
    frequency_band = torch.zeros_like(image_frequency)

    # Low frequency band processing
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    frequency_band.flatten()[idx_flat] = image_frequency.flatten()[
        idx_flat
    ] * val.flatten().to(complex_dtype)

    curvelet_band = torch.fft.ifftn(frequency_band)  # pylint: disable=not-callable

    low_freq_coeff = downsample(curvelet_band, decimation_ratios[0][0])
    norm = torch.sqrt(
        torch.prod(
            torch.full(
                (parameters.ndim,),
                fill_value=2 ** (parameters.num_scales - 2),
                dtype=torch.float64,
                device=image_frequency.device,
            )
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
    """
    Apply forward Uniform Discrete Curvelet Transform in complex mode.

    This function decomposes an input image or volume into complex-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Positive
    and negative frequency bands are separated into different directions.

    Parameters
    ----------
    image : torch.Tensor
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be complex-valued (complex floating point dtype).
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - num_scales : int
            Total number of scales (including lowpass scale)
        - ndim : int
            Number of dimensions of the transform
        - shape : tuple[int, ...]
            Shape of the input data
    windows : UDCTWindows
        Curvelet windows in sparse format, typically computed by
        `_udct_windows`. Structure is:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[torch.Tensor]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
        Each coefficient array has shape determined by decimation ratios.
        Coefficients have the same complex dtype as input.

    Notes
    -----
    The complex transform separates positive and negative frequencies into
    different directions. Each band is scaled by :math:`\\sqrt{0.5}` to maintain energy
    preservation. The negative frequency windows are obtained by flipping the
    positive frequency windows using `flip_fft_all_axes`.

    This mode is required for complex-valued inputs and provides full frequency
    information.
    """
    image_frequency = torch.fft.fftn(image)  # pylint: disable=not-callable
    complex_dtype = image_frequency.dtype

    # Low frequency band processing
    frequency_band = torch.zeros_like(image_frequency)
    idx, val = windows[0][0][0]
    idx_flat = idx.flatten()
    frequency_band.flatten()[idx_flat] = image_frequency.flatten()[
        idx_flat
    ] * val.flatten().to(complex_dtype)

    curvelet_band = torch.fft.ifftn(frequency_band)  # pylint: disable=not-callable

    low_freq_coeff = downsample(curvelet_band, decimation_ratios[0][0])
    norm = torch.sqrt(
        torch.prod(
            torch.full(
                (parameters.ndim,),
                fill_value=2 ** (parameters.num_scales - 2),
                dtype=torch.float64,
                device=image_frequency.device,
            )
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
                    decimation_ratio = decimation_ratios[scale_idx][
                        window_direction_idx, :
                    ]

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
                    decimation_ratio = decimation_ratios[scale_idx][
                        window_direction_idx, :
                    ]

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
