"""Forward transform functions for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import math

import torch

from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows


def _wedge_analyze_scale(
    decimation_ratio: torch.Tensor,
    *,
    complex_mode: bool = False,
) -> float:
    """Scale for periodized forward wedge (optional complex √0.5)."""
    prod_d = float(torch.prod(decimation_ratio.float()).item())
    scale = float(math.sqrt(2.0 * prod_d) / prod_d)
    if complex_mode:
        scale *= float(math.sqrt(0.5))
    return scale


def _process_wedge_real(
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
) -> torch.Tensor:
    """
    Process a single wedge for real transform mode.

    Uses a periodized (folded) sparse IFFT instead of a full-size IFFT
    followed by downsampling.

    Parameters
    ----------
    window : SparseWindow
        Sparse window representation (preferably with ``folded_indices``).
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Input image in frequency domain (from FFT).

    Returns
    -------
    torch.Tensor
        Downsampled and normalized coefficient array for this wedge.

    Notes
    -----
    The real transform combines positive and negative frequencies, so no
    :math:`\\sqrt{0.5}` scaling is applied. The normalization factor ensures proper
    energy preservation.

    Equivalence: ``downsample(ifftn(W·F), d) == ifftn(fold(W·F)) / prod(d)``.
    """
    return window.analyze(
        image_frequency,
        _wedge_analyze_scale(decimation_ratio),
        decimation=decimation_ratio,
    )


def _process_wedge_complex(
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    flip_window: bool = False,
) -> torch.Tensor:
    """
    Process a single wedge for complex transform mode.

    Uses a periodized sparse IFFT and optional index-space window flip
    (no dense ``to_dense`` / ``flip_fft_all_axes``).

    Parameters
    ----------
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Input image in frequency domain (from FFT).
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
    return window.analyze(
        image_frequency,
        _wedge_analyze_scale(decimation_ratio, complex_mode=True),
        flip=flip_window,
        decimation=decimation_ratio,
    )


def _forward_lowpass_periodized(
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    scale_norm: float,
) -> torch.Tensor:
    """Lowpass forward via folded sparse IFFT (no wedge ``sqrt(2)`` factor)."""
    scale = scale_norm / float(torch.prod(decimation_ratio.float()).item())
    return window.analyze(
        image_frequency,
        scale,
        decimation=decimation_ratio,
    )


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
        windows[scale][direction][wedge] = SparseWindow
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

    scale_norm = float(
        torch.sqrt(
            torch.prod(
                torch.full(
                    (parameters.ndim,),
                    fill_value=2 ** (parameters.num_scales - 2),
                    dtype=torch.float64,
                    device=image_frequency.device,
                )
            )
        ).item()
    )
    low_freq_coeff = _forward_lowpass_periodized(
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
        scale_norm,
    )

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
        windows[scale][direction][wedge] = SparseWindow
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

    scale_norm = float(
        torch.sqrt(
            torch.prod(
                torch.full(
                    (parameters.ndim,),
                    fill_value=2 ** (parameters.num_scales - 2),
                    dtype=torch.float64,
                    device=image_frequency.device,
                )
            )
        ).item()
    )
    low_freq_coeff = _forward_lowpass_periodized(
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
        scale_norm,
    )

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
                    flip_window=True,
                )
                direction_coeffs.append(coeff)
            scale_coeffs.append(direction_coeffs)

        coefficients.append(scale_coeffs)

    return coefficients
