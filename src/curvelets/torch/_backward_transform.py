"""Backward transform functions for PyTorch UDCT implementation."""

# pylint: disable=duplicate-code
# Duplicate code with numpy implementation is expected
from __future__ import annotations

import math

import torch

from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import UDCTCoefficients, UDCTWindows


def _wedge_synthesize_scale(
    decimation_ratio: torch.Tensor,
    *,
    complex_mode: bool = False,
    symmetric: bool = False,
) -> float:
    """Scale for periodized backward band (√D if symmetric, else 2√(D/2), optional complex √0.5)."""
    prod_d = float(torch.prod(decimation_ratio.float()).item())
    if symmetric:
        return float(math.sqrt(prod_d))
    scale = 2.0 * float(math.sqrt(prod_d / 2.0))
    if complex_mode:
        scale *= float(math.sqrt(0.5))
    return scale


def _process_wedge_backward_real(
    coefficient: torch.Tensor,
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    *,
    symmetric: bool = False,
) -> None:
    """
    Accumulate one real-mode wedge into ``image_frequency`` via tiled sparse FFT.

    Parameters
    ----------
    coefficient : torch.Tensor
        Downsampled coefficient array for this wedge.
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Full-size frequency accumulator (modified in-place).
    symmetric : bool, optional
        If True, use symmetric-band scaling ``sqrt(prod(d))`` (wavelet
        highest scale) instead of directional wedge scaling. Default is False.

    Notes
    -----
    Equivalence: ``fftn(upsample(c, d)) == tile(fftn(c), d)``.
    Scale is ``2 * sqrt(prod(d) / 2)`` for directional wedges and
    ``sqrt(prod(d))`` for symmetric wavelet bands.
    """
    window.synthesize(
        coefficient,
        image_frequency,
        _wedge_synthesize_scale(decimation_ratio, symmetric=symmetric),
        decimation=decimation_ratio,
    )


def _process_wedge_backward_complex(
    coefficient: torch.Tensor,
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
    flip_window: bool = False,
    *,
    symmetric: bool = False,
) -> None:
    """
    Accumulate one complex-mode wedge into ``image_frequency`` via tiled sparse FFT.

    Parameters
    ----------
    coefficient : torch.Tensor
        Downsampled coefficient array for this wedge.
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : torch.Tensor
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : torch.Tensor
        Full-size frequency accumulator (modified in-place).
    flip_window : bool, optional
        If True, use flipped indices for negative frequency processing.
        Default is False.
    symmetric : bool, optional
        If True, use symmetric-band scaling ``sqrt(prod(d))`` (wavelet
        highest scale) instead of directional wedge scaling. Default is False.

    Notes
    -----
    Scale includes :math:`\\sqrt{0.5}` for +/- frequency separation on
    directional wedges: ``2 * sqrt(0.5) * sqrt(prod(d) / 2)``.
    """
    window.synthesize(
        coefficient,
        image_frequency,
        _wedge_synthesize_scale(
            decimation_ratio, complex_mode=True, symmetric=symmetric
        ),
        flip=flip_window,
        decimation=decimation_ratio,
    )


def _backward_lowpass_periodized(
    coefficient: torch.Tensor,
    window: SparseWindow,
    decimation_ratio: torch.Tensor,
    image_frequency: torch.Tensor,
) -> None:
    """Accumulate lowpass band via small FFT + tiled sparse scatter."""
    prod_d = float(torch.prod(decimation_ratio.float()).item())
    window.synthesize(
        coefficient,
        image_frequency,
        float(math.sqrt(prod_d)),
        decimation=decimation_ratio,
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

    image_frequency = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)
    for scale_idx in range(1, parameters.num_scales):
        symmetric = len(windows[scale_idx]) == 1
        for direction_idx, direction_windows in enumerate(windows[scale_idx]):
            decimation_ratio = (
                decimation_ratios[scale_idx][0, :]
                if decimation_ratios[scale_idx].shape[0] == 1
                else decimation_ratios[scale_idx][direction_idx, :]
            )
            for wedge_idx, window in enumerate(direction_windows):
                _process_wedge_backward_real(
                    coefficients[scale_idx][direction_idx][wedge_idx],
                    window,
                    decimation_ratio,
                    image_frequency,
                    symmetric=symmetric,
                )

    _backward_lowpass_periodized(
        coefficients[0][0][0],
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
    )
    res: torch.Tensor = torch.fft.ifftn(image_frequency).real  # pylint: disable=not-callable
    return res


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
    frequency bands (1 direction at the wavelet-mode highest scale).

    Parameters
    ----------
    coefficients : UDCTCoefficients
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = torch.Tensor
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per
          curvelet scale; 1 direction at the wavelet-mode highest scale)
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

    image_frequency = torch.zeros(parameters.shape, dtype=complex_dtype, device=device)
    for scale_idx in range(1, parameters.num_scales):
        if len(windows[scale_idx]) == 1:
            decimation_ratio = decimation_ratios[scale_idx][0, :]
            for wedge_idx, window in enumerate(windows[scale_idx][0]):
                _process_wedge_backward_complex(
                    coefficients[scale_idx][0][wedge_idx],
                    window,
                    decimation_ratio,
                    image_frequency,
                    flip_window=False,
                    symmetric=True,
                )
            continue
        for flip_window in (False, True):
            coeff_offset = parameters.ndim if flip_window else 0
            for direction_idx in range(parameters.ndim):
                decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                for wedge_idx, window in enumerate(windows[scale_idx][direction_idx]):
                    _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx + coeff_offset][
                            wedge_idx
                        ],
                        window,
                        decimation_ratio,
                        image_frequency,
                        flip_window=flip_window,
                        symmetric=False,
                    )

    _backward_lowpass_periodized(
        coefficients[0][0][0],
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
    )
    res: torch.Tensor = torch.fft.ifftn(image_frequency)  # pylint: disable=not-callable
    return res


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
          * Real mode: ndim directions per curvelet scale (1 in wavelet mode)
          * Complex mode: 2*ndim directions per curvelet scale (1 in wavelet mode)
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
        windows[scale][direction][wedge] = SparseWindow
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
       - Directional curvelet bands are multiplied by 2 (to account for
         combined +/- frequencies in real mode, or separate processing
         in complex mode)
       - Low-frequency and symmetric wavelet-mode bands count once
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
