from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from ._typing import C, UDCTWindows, _to_complex_dtype, _to_real_dtype
from ._utils import ParamUDCT, flip_fft_all_axes, upsample


def _process_wedge_backward_real(
    coefficient: npt.NDArray[np.complexfloating],
    window: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]],
    decimation_ratio: npt.NDArray[np.int_],
    complex_dtype: npt.DTypeLike,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for real backward transform mode.

    This function upsamples a coefficient, transforms it to frequency domain,
    applies the window, and returns the frequency-domain contribution.

    Parameters
    ----------
    coefficient : npt.NDArray[np.complexfloating]
        Downsampled coefficient array for this wedge.
    window : tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    complex_dtype : npt.DTypeLike
        Complex dtype for output.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        Frequency-domain contribution as sparse array (only non-zero at window indices).
        Same shape as the full image size.

    Notes
    -----
    The contribution is sparse - only non-zero at the window indices. This allows
    efficient accumulation using sparse indexing in the real transform mode.
    """
    # Upsample coefficient to full size
    curvelet_band = upsample(coefficient, decimation_ratio)

    # Undo normalization: divide by sqrt(2 * prod(decimation_ratio))
    curvelet_band /= np.sqrt(2 * np.prod(decimation_ratio))

    # Transform to frequency domain
    curvelet_band = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band)

    # Get window indices and values
    idx, val = window

    # Create sparse contribution array (only non-zero at window indices)
    contribution = np.zeros(curvelet_band.shape, dtype=complex_dtype)
    contribution.flat[idx] = curvelet_band.flat[idx] * val.astype(complex_dtype)

    return contribution


def _process_wedge_backward_complex(
    coefficient: npt.NDArray[np.complexfloating],
    window: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]],
    decimation_ratio: npt.NDArray[np.int_],
    parameters: ParamUDCT,
    complex_dtype: npt.DTypeLike,
    flip_window: bool = False,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for complex backward transform mode.

    This function upsamples a coefficient, transforms it to frequency domain,
    applies the window (optionally flipped for negative frequencies), and returns
    the frequency-domain contribution with :math:`\\sqrt{0.5}` scaling.

    Parameters
    ----------
    coefficient : npt.NDArray[np.complexfloating]
        Downsampled coefficient array for this wedge.
    window : tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    parameters : ParamUDCT
        UDCT parameters containing size information.
    complex_dtype : npt.DTypeLike
        Complex dtype for output.
    flip_window : bool, optional
        If True, flip the window for negative frequency processing.
        Default is False.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        Full frequency-domain contribution array with sqrt(0.5) scaling applied.

    Notes
    -----
    The contribution is a full array (not sparse) to allow efficient accumulation
    in complex transform mode. The :math:`\\sqrt{0.5}` scaling accounts for the separation
    of positive and negative frequencies.
    """
    # pylint: disable=duplicate-code
    # Get window indices and values
    idx, val = window

    # Convert sparse window to dense for manipulation
    subwindow = np.zeros(parameters.shape, dtype=val.dtype)
    subwindow.flat[idx] = val

    # Optionally flip the window for negative frequency processing
    if flip_window:
        subwindow = flip_fft_all_axes(subwindow)

    # Upsample coefficient to full size
    curvelet_band = upsample(coefficient, decimation_ratio)

    # Undo normalization: divide by sqrt(2 * prod(decimation_ratio))
    curvelet_band /= np.sqrt(2 * np.prod(decimation_ratio))

    # Transform to frequency domain
    curvelet_band = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band)

    # Apply window with sqrt(0.5) scaling for complex transform
    return np.sqrt(0.5) * curvelet_band * subwindow.astype(complex_dtype)


def _apply_backward_transform_real(
    coefficients: list[list[list[npt.NDArray[C]]]],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform in real mode.

    This function reconstructs a real-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The real transform mode processes combined positive/negative
    frequency bands.

    Parameters
    ----------
    coefficients : list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
    parameters : ParamUDCT
        UDCT parameters containing transform configuration.
    windows : UDCTWindows
        Curvelet windows in sparse format, must match those used in
        forward transform.
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction, must match those
        used in forward transform.

    Returns
    -------
    np.ndarray
        Reconstructed real-valued image or volume with shape `parameters.shape`.

    Notes
    -----
    The real transform combines positive and negative frequencies, resulting
    in real-valued output. Contributions are accumulated using sparse indexing
    for efficiency.
    """
    # Determine dtype from coefficients
    real_dtype = _to_real_dtype(coefficients[0][0][0].dtype)
    complex_dtype = _to_complex_dtype(real_dtype)

    # Initialize frequency domain
    image_frequency = np.zeros(parameters.shape, dtype=complex_dtype)

    # Process high-frequency bands using loops
    # For "wavelet" mode at highest scale, we only have 1 window (ring-shaped, symmetric)
    # so we don't need the factor of 2 (the window already covers all frequencies)
    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        # For wavelet mode: process highest scale separately without factor of 2
        # Other scales use factor of 2 as normal
        image_frequency_other_scales = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_wavelet_scale = np.zeros(parameters.shape, dtype=complex_dtype)
        # pylint: disable=duplicate-code
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
                        complex_dtype,
                    )
                    idx, _ = window
                    if scale_idx == highest_scale_idx:
                        image_frequency_wavelet_scale.flat[idx] += contribution.flat[
                            idx
                        ]
                    else:
                        image_frequency_other_scales.flat[idx] += contribution.flat[idx]
    else:
        # Normal curvelet mode: process all scales together
        # pylint: disable=duplicate-code
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
                        complex_dtype,
                    )
                    idx, _ = window
                    image_frequency.flat[idx] += contribution.flat[idx]

    # Process low-frequency band
    image_frequency_low = np.zeros(parameters.shape, dtype=complex_dtype)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(curvelet_band)
    idx, val = windows[0][0][0]
    image_frequency_low.flat[idx] += curvelet_band.flat[idx] * val.astype(complex_dtype)

    # Combine: low frequency + high frequency contributions
    # For real transform mode, multiply high-frequency by 2 to account for combined +/- frequencies
    # Exception: For "wavelet" mode at highest scale, we don't multiply by 2
    if is_wavelet_mode_highest_scale:
        image_frequency = (
            2 * image_frequency_other_scales
            + image_frequency_wavelet_scale
            + image_frequency_low
        )
    else:
        image_frequency = 2 * image_frequency + image_frequency_low
    return np.fft.ifftn(image_frequency).real


def _apply_backward_transform_complex(
    coefficients: list[list[list[npt.NDArray[C]]]],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform in complex mode.

    This function reconstructs a complex-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The complex transform mode processes positive and negative
    frequency bands separately.

    Parameters
    ----------
    coefficients : list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
    parameters : ParamUDCT
        UDCT parameters containing transform configuration.
    windows : UDCTWindows
        Curvelet windows in sparse format, must match those used in
        forward transform.
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction, must match those
        used in forward transform.

    Returns
    -------
    np.ndarray
        Reconstructed complex-valued image or volume with shape `parameters.shape`.

    Notes
    -----
    The complex transform separates positive and negative frequencies, resulting
    in complex-valued output. Contributions are accumulated using full array
    operations for efficiency.
    """
    # Determine dtype from coefficients
    real_dtype = _to_real_dtype(coefficients[0][0][0].dtype)
    complex_dtype = _to_complex_dtype(real_dtype)

    # Initialize frequency domain
    # For "wavelet" mode at highest scale, we only have 1 window (ring-shaped, symmetric)
    # so we don't need the factor of 2 (the window already covers all frequencies)
    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    # pylint: disable=too-many-nested-blocks
    if is_wavelet_mode_highest_scale:
        # For wavelet mode: process highest scale separately without factor of 2
        # Other scales use factor of 2 as normal
        # Note: In wavelet mode, coefficients are identical for all directions,
        # so we only process direction 0 for positive and direction 0 for negative
        image_frequency_other_scales = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_wavelet_scale = np.zeros(parameters.shape, dtype=complex_dtype)

        # Process positive frequency bands (directions 0..dim-1)
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
                        complex_dtype,
                        flip_window=False,
                    )
                    if scale_idx == highest_scale_idx:
                        # For wavelet mode, only process direction 0 (coefficients are identical)
                        if direction_idx == 0:
                            image_frequency_wavelet_scale += contribution
                    else:
                        image_frequency_other_scales += contribution

        # Process negative frequency bands (directions dim..2*dim-1)
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
                        complex_dtype,
                        flip_window=True,
                    )
                    if scale_idx == highest_scale_idx:
                        # For wavelet mode, only process direction 0 (coefficients are identical)
                        if direction_idx == 0:
                            image_frequency_wavelet_scale += contribution
                    else:
                        image_frequency_other_scales += contribution
    else:
        # Normal curvelet mode: process all scales together
        image_frequency = np.zeros(parameters.shape, dtype=complex_dtype)
        # Process positive frequency bands (directions 0..dim-1)
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
                        complex_dtype,
                        flip_window=False,
                    )
                    image_frequency += contribution

        # Process negative frequency bands (directions dim..2*dim-1)
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
                        complex_dtype,
                        flip_window=True,
                    )
                    image_frequency += contribution

    # Process low-frequency band
    image_frequency_low = np.zeros(parameters.shape, dtype=complex_dtype)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsample(coefficients[0][0][0], decimation_ratio)
    curvelet_band = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(curvelet_band)
    idx, val = windows[0][0][0]
    image_frequency_low.flat[idx] += curvelet_band.flat[idx] * val.astype(complex_dtype)

    # Combine: low frequency + high frequency contributions
    # For complex transform mode, multiply high-frequency by 2 to account for separate +/- frequencies
    # Exception: For "wavelet" mode at highest scale, we don't multiply by 2
    if is_wavelet_mode_highest_scale:
        image_frequency = (
            2 * image_frequency_other_scales
            + image_frequency_wavelet_scale
            + image_frequency_low
        )
    else:
        image_frequency = 2 * image_frequency + image_frequency_low
    return np.fft.ifftn(image_frequency)


@overload
def _apply_backward_transform(
    coefficients: list[list[list[npt.NDArray[C]]]],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[True],
) -> np.ndarray: ...


@overload
def _apply_backward_transform(
    coefficients: list[list[list[npt.NDArray[C]]]],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[False] = False,
) -> np.ndarray: ...


def _apply_backward_transform(
    coefficients: list[list[list[npt.NDArray[C]]]],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform (reconstruction).

    This function reconstructs an image or volume from curvelet coefficients
    by upsampling, applying frequency-domain windows, and combining all bands.
    The transform can operate in two modes: real transform (default) or complex
    transform, matching the mode used in the forward transform.

    Parameters
    ----------
    coefficients : list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = np.ndarray
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
    decimation_ratios : list[npt.NDArray[np.int_]]
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
    np.ndarray
        Reconstructed image or volume with shape `parameters.shape`.
        - Real mode: Returns real-valued array (dtype matches input real part)
        - Complex mode: Returns complex-valued array

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
