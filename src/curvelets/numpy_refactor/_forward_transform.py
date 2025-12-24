from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from ._typing import (
    ComplexFloatingNDArray,
    FloatingNDArray,
    UDCTCoefficients,
    UDCTWindows,
    _is_complex_array,
    _is_floating_array,
)
from ._utils import ParamUDCT, downsample, flip_fft_all_axes


def _process_wedge_real(
    window: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]],
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    freq_band: npt.NDArray[np.complexfloating],
    complex_dtype: npt.DTypeLike,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for real transform mode.

    This function applies a frequency-domain window to extract a specific
    curvelet band, transforms it to spatial domain, downsamples it, and applies
    normalization.

    Parameters
    ----------
    window : tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : npt.NDArray[np.complexfloating]
        Input image in frequency domain (from FFT).
    freq_band : npt.NDArray[np.complexfloating]
        Reusable frequency band buffer (will be cleared and filled).
    complex_dtype : npt.DTypeLike
        Complex dtype matching image_frequency.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        Downsampled and normalized coefficient array for this wedge.

    Notes
    -----
    The real transform combines positive and negative frequencies, so no
    sqrt(0.5) scaling is applied. The normalization factor ensures proper
    energy preservation.
    """
    # Clear the frequency band buffer for reuse
    freq_band.fill(0)

    # Get the sparse window representation (indices and values)
    idx, val = window

    # Apply the window to the frequency domain: multiply image frequencies
    # by the window values at the specified indices
    freq_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)

    # Transform back to spatial domain using inverse FFT
    curvelet_band = np.fft.ifftn(freq_band)

    # Downsample the curvelet band according to the decimation ratio
    coeff = downsample(curvelet_band, decimation_ratio)

    # Apply normalization factor: sqrt(2 * product of decimation ratios)
    # This ensures proper energy preservation in the transform
    coeff *= np.sqrt(2 * np.prod(decimation_ratio))

    return coeff


def _process_wedge_complex(
    window: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]],
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    parameters: ParamUDCT,
    complex_dtype: npt.DTypeLike,
    flip_window: bool = False,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for complex transform mode.

    This function applies a frequency-domain window (optionally flipped for
    negative frequencies) to extract a specific curvelet band, transforms it
    to spatial domain with sqrt(0.5) scaling, downsamples it, and applies
    normalization.

    Parameters
    ----------
    window : tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : npt.NDArray[np.complexfloating]
        Input image in frequency domain (from FFT).
    parameters : ParamUDCT
        UDCT parameters containing size information.
    complex_dtype : npt.DTypeLike
        Complex dtype matching image_frequency.
    flip_window : bool, optional
        If True, flip the window for negative frequency processing.
        Default is False.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        Downsampled and normalized coefficient array for this wedge.

    Notes
    -----
    The complex transform separates positive and negative frequencies, so
    sqrt(0.5) scaling is applied to each band. The normalization factor ensures
    proper energy preservation.
    """
    # Get the sparse window representation (indices and values)
    idx, val = window

    # Convert sparse window to dense for manipulation
    subwindow = np.zeros(parameters.size, dtype=val.dtype)
    subwindow.flat[idx] = val

    # Optionally flip the window for negative frequency processing
    if flip_window:
        subwindow = flip_fft_all_axes(subwindow)

    # Apply window to frequency domain and transform to spatial domain
    # Apply sqrt(0.5) scaling for complex transform (separates +/- frequencies)
    band_filtered = np.sqrt(0.5) * np.fft.ifftn(
        image_frequency * subwindow.astype(complex_dtype)
    )

    # Downsample the curvelet band according to the decimation ratio
    coeff = downsample(band_filtered, decimation_ratio)

    # Apply normalization factor: sqrt(2 * product of decimation ratios)
    # This ensures proper energy preservation in the transform
    coeff *= np.sqrt(2 * np.prod(decimation_ratio))

    return coeff


def _apply_forward_transform_real(
    image: FloatingNDArray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> UDCTCoefficients:
    """
    Apply forward Uniform Discrete Curvelet Transform in real mode.

    This function decomposes an input image or volume into real-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Each
    curvelet band captures both positive and negative frequencies combined.

    Parameters
    ----------
    image : FloatingNDArray
        Input image or volume to decompose. Must have shape matching
        `parameters.size`. Must be real-valued (floating point dtype).
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - res : int
            Number of resolution scales
        - dim : int
            Dimensionality of the transform
        - size : tuple[int, ...]
            Shape of the input data
    windows : UDCTWindows
        Curvelet windows in sparse format, typically computed by
        `_udct_windows`. Structure is:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..res: High-frequency bands (dim directions per scale)
        Each coefficient array has shape determined by decimation ratios.
        All coefficients are real-valued.

    Notes
    -----
    The real transform combines positive and negative frequencies, resulting
    in real-valued coefficients. This is suitable for real-valued inputs and
    provides a more compact representation.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._utils import ParamUDCT
    >>> from curvelets.numpy_refactor._udct_windows import _udct_windows
    >>> from curvelets.numpy_refactor._forward_transform import _apply_forward_transform_real
    >>>
    >>> # Create parameters for 2D transform
    >>> params = ParamUDCT(
    ...     size=(64, 64),
    ...     res=3,
    ...     dim=2,
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5
    ... )
    >>>
    >>> # Compute windows (typically done once and reused)
    >>> windows, decimation_ratios, _ = _udct_windows(params)
    >>>
    >>> # Create test image
    >>> image = np.random.randn(64, 64)
    >>>
    >>> # Apply forward transform (real mode)
    >>> coeffs = _apply_forward_transform_real(image, params, windows, decimation_ratios)
    >>>
    >>> # Check structure
    >>> len(coeffs)  # Number of scales (0 + res)
    4
    >>> len(coeffs[0][0])  # Low-frequency: 1 wedge
    1
    >>> len(coeffs[1])  # First high-frequency scale: 2 directions (real mode)
    2
    >>> coeffs[0][0][0].shape  # Downsampled low-frequency coefficients
    (32, 32)
    >>> np.isrealobj(coeffs[0][0][0])  # Real coefficients
    True
    """
    image_frequency = np.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Allocate frequency_band once for reuse
    frequency_band = np.zeros_like(image_frequency)

    # Low frequency band processing
    idx, val = windows[0][0][0]
    frequency_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)

    # Real transform: take real part
    curvelet_band = np.fft.ifftn(frequency_band)

    low_freq_coeff = downsample(curvelet_band, decimation_ratios[0][0])
    norm = np.sqrt(
        np.prod(np.full((parameters.dim,), fill_value=2 ** (parameters.res - 1)))
    )
    low_freq_coeff *= norm

    # Real transform: combined +/- frequencies using nested list comprehensions
    # Build entire structure with list comprehensions
    coefficients: UDCTCoefficients = [
        [[low_freq_coeff]]  # Scale 0: 1 direction, 1 wedge
    ] + [
        [
            [
                _process_wedge_real(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    frequency_band,
                    complex_dtype,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(parameters.dim)
        ]
        for scale_idx in range(1, 1 + parameters.res)
    ]
    return coefficients


def _apply_forward_transform_complex(
    image: ComplexFloatingNDArray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> UDCTCoefficients:
    """
    Apply forward Uniform Discrete Curvelet Transform in complex mode.

    This function decomposes an input image or volume into complex-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Positive
    and negative frequency bands are separated into different directions.

    Parameters
    ----------
    image : ComplexFloatingNDArray
        Input image or volume to decompose. Must have shape matching
        `parameters.size`. Must be complex-valued (complex floating point dtype).
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - res : int
            Number of resolution scales
        - dim : int
            Dimensionality of the transform
        - size : tuple[int, ...]
            Shape of the input data
    windows : UDCTWindows
        Curvelet windows in sparse format, typically computed by
        `_udct_windows`. Structure is:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..res: High-frequency bands (2*dim directions per scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
        Each coefficient array has shape determined by decimation ratios.
        All coefficients are complex-valued.

    Notes
    -----
    The complex transform separates positive and negative frequencies into
    different directions. Each band is scaled by sqrt(0.5) to maintain energy
    preservation. The negative frequency windows are obtained by flipping the
    positive frequency windows using `flip_fft_all_axes`.

    This mode is required for complex-valued inputs and provides full frequency
    information.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._utils import ParamUDCT
    >>> from curvelets.numpy_refactor._udct_windows import _udct_windows
    >>> from curvelets.numpy_refactor._forward_transform import _apply_forward_transform_complex
    >>>
    >>> # Create parameters for 2D transform
    >>> params = ParamUDCT(
    ...     size=(64, 64),
    ...     res=3,
    ...     dim=2,
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5
    ... )
    >>>
    >>> # Compute windows (typically done once and reused)
    >>> windows, decimation_ratios, _ = _udct_windows(params)
    >>>
    >>> # Create test image
    >>> image = np.random.randn(64, 64)
    >>>
    >>> # Apply forward transform (complex mode)
    >>> coeffs_complex = _apply_forward_transform_complex(
    ...     image, params, windows, decimation_ratios
    ... )
    >>>
    >>> # Check structure
    >>> len(coeffs_complex)  # Number of scales (0 + res)
    4
    >>> len(coeffs_complex[1])  # Complex mode: 4 directions (2*dim)
    4
    >>> np.iscomplexobj(coeffs_complex[0][0][0])  # Complex coefficients
    True
    """
    image_frequency = np.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Low frequency band processing
    frequency_band = np.zeros_like(image_frequency)
    idx, val = windows[0][0][0]
    frequency_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)

    # Complex transform: keep complex low frequency
    curvelet_band = np.fft.ifftn(frequency_band)

    coefficients: UDCTCoefficients = [
        [[downsample(curvelet_band, decimation_ratios[0][0])]]
    ]
    norm = np.sqrt(
        np.prod(np.full((parameters.dim,), fill_value=2 ** (parameters.res - 1)))
    )
    coefficients[0][0][0] *= norm

    # Complex transform: separate +/- frequency bands using nested list comprehensions
    # Structure: [scale][direction][wedge]
    # Directions 0..dim-1 are positive frequencies
    # Directions dim..2*dim-1 are negative frequencies
    coefficients = coefficients + [
        [
            # Positive frequency bands (directions 0..dim-1)
            [
                _process_wedge_complex(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    parameters,
                    complex_dtype,
                    flip_window=False,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(parameters.dim)
        ]
        + [
            # Negative frequency bands (directions dim..2*dim-1)
            [
                _process_wedge_complex(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    parameters,
                    complex_dtype,
                    flip_window=True,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(parameters.dim)
        ]
        for scale_idx in range(1, 1 + parameters.res)
    ]
    return coefficients


@overload
def _apply_forward_transform(
    image: ComplexFloatingNDArray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[True],
) -> UDCTCoefficients: ...


@overload
def _apply_forward_transform(
    image: FloatingNDArray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[False] = False,
) -> UDCTCoefficients: ...


def _apply_forward_transform(
    image: FloatingNDArray | ComplexFloatingNDArray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
) -> UDCTCoefficients:
    """
    Apply forward Uniform Discrete Curvelet Transform (decomposition).

    This function decomposes an input image or volume into curvelet coefficients
    by applying frequency-domain windows and downsampling. The transform can
    operate in two modes: real transform (default) or complex transform.

    Parameters
    ----------
    image : FloatingNDArray | ComplexFloatingNDArray
        Input image or volume to decompose. Must have shape matching
        `parameters.size`. Must be either real-valued (FloatingNDArray) or
        complex-valued (ComplexFloatingNDArray).
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - res : int
            Number of resolution scales
        - dim : int
            Dimensionality of the transform
        - size : tuple[int, ...]
            Shape of the input data
    windows : UDCTWindows
        Curvelet windows in sparse format, typically computed by
        `_udct_windows`. Structure is:
        windows[scale][direction][wedge] = (indices, values) tuple
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0
    use_complex_transform : bool, optional
        Transform mode flag:
        - False (default): Real transform mode. Each curvelet band captures
          both positive and negative frequencies combined. Coefficients are
          real-valued. Suitable for real-valued inputs.
        - True: Complex transform mode. Positive and negative frequency bands
          are separated into different directions. Directions 0..dim-1 are
          positive frequencies, directions dim..2*dim-1 are negative frequencies.
          Each band is scaled by sqrt(0.5). Coefficients are complex-valued.
          Required for complex-valued inputs.

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..res: High-frequency bands
          * Real mode: dim directions per scale
          * Complex mode: 2*dim directions per scale
        Each coefficient array has shape determined by decimation ratios.

    Notes
    -----
    The forward transform process:

    1. **FFT**: Input is transformed to frequency domain using FFT.

    2. **Window application**: Frequency-domain windows are applied to
       extract different frequency bands and directions. Windows are stored
       in sparse format for efficiency.

    3. **IFFT**: Each windowed frequency band is transformed back to
       spatial domain.

    4. **Downsampling**: Each band is downsampled according to its
       decimation ratio, which depends on the scale and direction.

    5. **Normalization**: Coefficients are scaled to ensure proper energy
       preservation. Low-frequency band uses a different normalization than
       high-frequency bands.

    For complex transform mode, positive and negative frequencies are
    processed separately. The negative frequency windows are obtained by
    flipping the positive frequency windows using `flip_fft_all_axes`.

    The transform provides a tight frame, meaning perfect reconstruction
    is possible using the corresponding backward transform.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor._utils import ParamUDCT
    >>> from curvelets.numpy_refactor._udct_windows import _udct_windows
    >>> from curvelets.numpy_refactor._forward_transform import _apply_forward_transform
    >>>
    >>> # Create parameters for 2D transform
    >>> params = ParamUDCT(
    ...     size=(64, 64),
    ...     res=3,
    ...     dim=2,
    ...     angular_wedges_config=np.array([[3], [6], [12]]),
    ...     window_overlap=0.15,
    ...     window_threshold=1e-5
    ... )
    >>>
    >>> # Compute windows (typically done once and reused)
    >>> windows, decimation_ratios, _ = _udct_windows(params)
    >>>
    >>> # Create test image
    >>> image = np.random.randn(64, 64)
    >>>
    >>> # Apply forward transform (real mode)
    >>> coeffs = _apply_forward_transform(
    ...     image, params, windows, decimation_ratios, use_complex_transform=False
    ... )
    >>>
    >>> # Check structure
    >>> len(coeffs)  # Number of scales (0 + res)
    4
    >>> len(coeffs[0][0])  # Low-frequency: 1 wedge
    1
    >>> len(coeffs[1])  # First high-frequency scale: 2 directions (real mode)
    2
    >>> coeffs[0][0][0].shape  # Downsampled low-frequency coefficients
    (32, 32)
    >>>
    >>> # Apply forward transform (complex mode)
    >>> coeffs_complex = _apply_forward_transform(
    ...     image, params, windows, decimation_ratios, use_complex_transform=True
    ... )
    >>> len(coeffs_complex[1])  # Complex mode: 4 directions (2*dim)
    4
    >>> np.iscomplexobj(coeffs_complex[0][0][0])  # Complex coefficients
    True
    """
    if use_complex_transform:
        # Type guard narrows the type for the type checker
        # The overloads ensure type safety at call sites
        if _is_complex_array(image):
            return _apply_forward_transform_complex(
                image, parameters, windows, decimation_ratios
            )
        # Fall through if type guard doesn't match - try anyway for runtime flexibility
        # This handles edge cases where overloads can't determine type
        return _apply_forward_transform_complex(
            image,
            parameters,
            windows,
            decimation_ratios,  # type: ignore[arg-type]
        )

    # Real transform mode
    # Type guard narrows the type for the type checker
    # The overloads ensure type safety at call sites
    if _is_floating_array(image):
        return _apply_forward_transform_real(
            image, parameters, windows, decimation_ratios
        )

    # Fall through if type guard doesn't match - try anyway for runtime flexibility
    # This handles edge cases where overloads can't determine type
    return _apply_forward_transform_real(
        image,
        parameters,
        windows,
        decimation_ratios,  # type: ignore[arg-type]
    )
