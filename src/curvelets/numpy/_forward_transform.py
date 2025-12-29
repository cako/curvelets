from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from ._riesz import riesz_filters
from ._typing import (
    C,
    F,
    IntegerNDArray,
    IntpNDArray,
    MUDCTCoefficients,
    UDCTCoefficients,
    UDCTWindows,
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
    :math:`\\sqrt{0.5}` scaling is applied. The normalization factor ensures proper
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
    to spatial domain with :math:`\\sqrt{0.5}` scaling, downsamples it, and applies
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
    :math:`\\sqrt{0.5}` scaling is applied to each band. The normalization factor ensures
    proper energy preservation.
    """
    # pylint: disable=duplicate-code
    # Get the sparse window representation (indices and values)
    idx, val = window

    # Convert sparse window to dense for manipulation
    subwindow = np.zeros(parameters.shape, dtype=val.dtype)
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
    image: npt.NDArray[F],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> list[list[list[npt.NDArray[np.complexfloating]]]]:
    """
    Apply forward Uniform Discrete Curvelet Transform in real mode.

    This function decomposes an input image or volume into real-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Each
    curvelet band captures both positive and negative frequencies combined.

    Parameters
    ----------
    image : npt.NDArray[F]
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
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
        Each coefficient array has shape determined by decimation ratios.
        Coefficients are complex dtype matching the complex version of input dtype:
        - np.float32 input -> np.complex64 coefficients
        - np.float64 input -> np.complex128 coefficients

    Notes
    -----
    The real transform combines positive and negative frequencies, resulting
    in real-valued coefficients. This is suitable for real-valued inputs and
    provides a more compact representation.
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
        np.prod(
            np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
        )
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
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    frequency_band,
                    complex_dtype,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(len(windows[scale_idx]))
        ]
        for scale_idx in range(1, parameters.num_scales)
    ]
    return coefficients


def _apply_forward_transform_complex(
    image: npt.NDArray[C],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
) -> list[list[list[npt.NDArray[np.complexfloating]]]]:
    """
    Apply forward Uniform Discrete Curvelet Transform in complex mode.

    This function decomposes an input image or volume into complex-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Positive
    and negative frequency bands are separated into different directions.

    Parameters
    ----------
    image : npt.NDArray[C]
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
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
        Each coefficient array has shape determined by decimation ratios.
        Coefficients have the same complex dtype as input (C).

    Notes
    -----
    The complex transform separates positive and negative frequencies into
    different directions. Each band is scaled by :math:`\\sqrt{0.5}` to maintain energy
    preservation. The negative frequency windows are obtained by flipping the
    positive frequency windows using `flip_fft_all_axes`.

    This mode is required for complex-valued inputs and provides full frequency
    information.
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
        np.prod(
            np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
        )
    )
    coefficients[0][0][0] *= norm

    # Complex transform: separate +/- frequency bands using nested list comprehensions
    # Structure: [scale][direction][wedge]
    # Directions 0..dim-1 are positive frequencies
    # Directions dim..2*dim-1 are negative frequencies
    return coefficients + [
        [
            # Positive frequency bands (directions 0..dim-1)
            # For "wavelet" mode, reuse single window for all directions
            [
                _process_wedge_complex(
                    windows[scale_idx][min(direction_idx, len(windows[scale_idx]) - 1)][
                        wedge_idx
                    ],
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][
                        min(direction_idx, len(windows[scale_idx]) - 1), :
                    ],
                    image_frequency,
                    parameters,
                    complex_dtype,
                    flip_window=False,
                )
                for wedge_idx in range(
                    len(
                        windows[scale_idx][
                            min(direction_idx, len(windows[scale_idx]) - 1)
                        ]
                    )
                )
            ]
            for direction_idx in range(parameters.ndim)
        ]
        + [
            # Negative frequency bands (directions dim..2*dim-1)
            # For "wavelet" mode, reuse single window for all directions
            [
                _process_wedge_complex(
                    windows[scale_idx][min(direction_idx, len(windows[scale_idx]) - 1)][
                        wedge_idx
                    ],
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][
                        min(direction_idx, len(windows[scale_idx]) - 1), :
                    ],
                    image_frequency,
                    parameters,
                    complex_dtype,
                    flip_window=True,
                )
                for wedge_idx in range(
                    len(
                        windows[scale_idx][
                            min(direction_idx, len(windows[scale_idx]) - 1)
                        ]
                    )
                )
            ]
            for direction_idx in range(parameters.ndim)
        ]
        for scale_idx in range(1, parameters.num_scales)
    ]


def _process_wedge_monogenic(
    window: tuple[IntpNDArray, npt.NDArray[np.floating]],
    decimation_ratio: IntegerNDArray,
    image_frequency: npt.NDArray[np.complexfloating],
    riesz_filters_list: list[npt.NDArray[np.complexfloating]],
    freq_band: npt.NDArray[np.complexfloating],
    complex_dtype: npt.DTypeLike,
) -> list[npt.NDArray[np.complexfloating] | npt.NDArray[F]]:
    """
    Process a single wedge for monogenic transform.

    This function applies frequency-domain windows and Riesz filters to extract
    components: scalar (same as UDCT) plus all Riesz components (one per dimension).
    Each component is transformed to spatial domain, downsampled, and normalized.

    Parameters
    ----------
    window : tuple[IntpNDArray, npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
        Uses IntpNDArray type alias from _typing.py.
    decimation_ratio : IntegerNDArray
        Decimation ratio for this wedge (1D array with length equal to dimensions).
        Uses IntegerNDArray type alias from _typing.py.
    image_frequency : npt.NDArray[np.complexfloating]
        Input image in frequency domain (from FFT).
    riesz_filters_list : list[npt.NDArray[np.complexfloating]]
        Riesz transform filters R_1, R_2, ... R_ndim from riesz_filters().
        Each filter has shape matching image_frequency.
    freq_band : npt.NDArray[np.complexfloating]
        Reusable frequency band buffer (will be cleared and filled).
    complex_dtype : npt.DTypeLike
        Complex dtype matching image_frequency.

    Returns
    -------
    list[npt.NDArray[np.complexfloating] | npt.NDArray[F]]
        List with ndim+1 arrays: [scalar, riesz_1, riesz_2, ..., riesz_ndim].
        Scalar is complex (matches UDCT behavior), all Riesz components are real.
        Uses F TypeVar from _typing.py for real floating point types.
        Each array is downsampled and normalized.

    Notes
    -----
    The components are:
    - Scalar: IFFT(FFT(image) * window) - same as standard UDCT
    - Riesz_k: IFFT(FFT(image) * window * R_k_filter) for k = 1, 2, ..., ndim

    All components use the same decimation ratios and normalization factors
    as the standard UDCT transform.
    """
    # Scalar component (same as _process_wedge_real)
    freq_band.fill(0)
    idx, val = window
    freq_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)
    curvelet_band_scalar = np.fft.ifftn(freq_band)
    coeff_scalar = downsample(curvelet_band_scalar, decimation_ratio)
    coeff_scalar *= np.sqrt(2 * np.prod(decimation_ratio))

    # Convert to appropriate dtypes
    real_dtype = np.real(np.empty(0, dtype=complex_dtype)).dtype

    # Process all Riesz components
    riesz_coeffs: list[npt.NDArray[F]] = []
    for riesz_filter in riesz_filters_list:
        freq_band.fill(0)
        # Apply window and Riesz filter
        freq_band.flat[idx] = (
            image_frequency.flat[idx]
            * val.astype(complex_dtype)
            * riesz_filter.flat[idx]
        )
        curvelet_band_riesz = np.fft.ifftn(freq_band)
        coeff_riesz = downsample(curvelet_band_riesz, decimation_ratio)
        coeff_riesz *= np.sqrt(2 * np.prod(decimation_ratio))
        # Riesz components: take real part (Riesz transform of real function is real)
        riesz_coeffs.append(coeff_riesz.real.astype(real_dtype))

    # Return list: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
    # Scalar is kept as complex (matches UDCT behavior)
    return [coeff_scalar, *riesz_coeffs]


def _apply_forward_transform_monogenic(
    image: npt.NDArray[F],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[IntegerNDArray],
) -> MUDCTCoefficients:  # type: ignore[type-arg]
    """
    Apply forward monogenic curvelet transform.

    This function decomposes a real-valued input image or volume into monogenic
    curvelet coefficients by applying frequency-domain windows and Riesz transforms.
    Each coefficient band produces ndim+1 components: scalar (same as UDCT) plus
    all Riesz components (one per dimension).

    The monogenic curvelet transform was originally defined for 2D signals by
    Storath 2010 using quaternions, but this implementation extends it to arbitrary
    N-D signals by using all Riesz transform components. The reconstruction uses the
    discrete tight frame property of UDCT rather than quaternion multiplication,
    making the N-D extension straightforward.

    Parameters
    ----------
    image : npt.NDArray[F]
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be real-valued (floating point dtype).
        Uses the F TypeVar from _typing.py (np.float16, np.float32, np.float64, np.longdouble).
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
        Type alias from _typing.py.
    decimation_ratios : list[IntegerNDArray]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0
        Uses IntegerNDArray type alias from _typing.py.

    Returns
    -------
    MUDCTCoefficients
        Monogenic coefficients as nested list structure with lists of ndim+1 arrays.
        Type alias defined in _typing.py: list[list[list[list[npt.NDArray[np.complexfloating | F]]]]]
        Structure mirrors _apply_forward_transform_real() but returns lists.
        - scale 0: Low-frequency band (1 direction, 1 wedge) with ndim+1 components
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
          Each band has list of ndim+1 arrays: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
          where scalar is complex, all riesz components are real

    Notes
    -----
    The monogenic transform is mathematically defined only for real-valued functions.
    This function computes:
    - Scalar component: same as standard UDCT
    - Riesz_k component: applies :math:`R_k` filter :math:`(i \\xi_k / |\\xi|)` for :math:`k = 1, 2, \\ldots, \\text{ndim}`

    Structure mirrors _apply_forward_transform_real() but:
    - Computes Riesz filters once at the start
    - Processes each wedge to produce ndim+1 components (scalar + all Riesz)
    - Returns lists instead of complex arrays
    - Uses type aliases from _typing.py for consistency with rest of codebase

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy._utils import ParamUDCT
    >>> from curvelets.numpy._forward_transform import _apply_forward_transform_monogenic
    >>> from curvelets.numpy._udct_windows import UDCTWindow
    >>>
    >>> # Create parameters and windows
    >>> params = ParamUDCT(
    ...     shape=(64, 64),
    ...     angular_wedges_config=np.array([[3], [6]]),
    ...     window_overlap=0.15,
    ...     radial_frequency_params=(np.pi/3, 2*np.pi/3, 2*np.pi/3, 4*np.pi/3),
    ...     window_threshold=1e-5
    ... )
    >>> window_computer = UDCTWindow(params)
    >>> windows, decimation_ratios, _ = window_computer.compute()
    >>>
    >>> # Apply monogenic transform
    >>> image = np.random.randn(64, 64).astype(np.float64)
    >>> coeffs = _apply_forward_transform_monogenic(image, params, windows, decimation_ratios)
    >>> len(coeffs)  # Number of scales
    3
    >>> isinstance(coeffs[0][0][0], list)  # Each coefficient is a list
    True
    >>> len(coeffs[0][0][0])  # List has ndim+1 components (3 for 2D)
    3
    """
    image_frequency = np.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Compute Riesz filters once for the entire transform
    riesz_filters_list = riesz_filters(parameters.shape)

    # Allocate frequency_band once for reuse
    frequency_band = np.zeros_like(image_frequency)

    # Low frequency band processing (ndim+1 components: scalar + all Riesz)
    idx, val = windows[0][0][0]
    frequency_band.fill(0)
    frequency_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)

    # Scalar component
    curvelet_band_scalar = np.fft.ifftn(frequency_band)
    low_freq_coeff_scalar = downsample(curvelet_band_scalar, decimation_ratios[0][0])
    norm = np.sqrt(
        np.prod(
            np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
        )
    )
    low_freq_coeff_scalar *= norm

    # Convert to appropriate dtypes
    real_dtype = np.real(np.empty(0, dtype=complex_dtype)).dtype

    # Process all Riesz components for low frequency
    low_freq_riesz_coeffs: list[npt.NDArray[F]] = []
    for riesz_filter in riesz_filters_list:
        frequency_band.fill(0)
        frequency_band.flat[idx] = (
            image_frequency.flat[idx]
            * val.astype(complex_dtype)
            * riesz_filter.flat[idx]
        )
        curvelet_band_riesz = np.fft.ifftn(frequency_band)
        low_freq_coeff_riesz = downsample(curvelet_band_riesz, decimation_ratios[0][0])
        low_freq_coeff_riesz *= norm
        # Riesz components: take real part (Riesz transform of real function is real)
        low_freq_riesz_coeffs.append(low_freq_coeff_riesz.real.astype(real_dtype))

    # Build list: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
    # Scalar is kept as complex (matches UDCT behavior)
    low_freq_coeff = [low_freq_coeff_scalar, *low_freq_riesz_coeffs]

    # High-frequency bands using nested list comprehensions
    # Build entire structure with list comprehensions
    coefficients: MUDCTCoefficients = [  # type: ignore[type-arg]
        [[low_freq_coeff]]  # Scale 0: 1 direction, 1 wedge
    ] + [
        [
            [
                _process_wedge_monogenic(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    riesz_filters_list,
                    frequency_band,
                    complex_dtype,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(len(windows[scale_idx]))
        ]
        for scale_idx in range(1, parameters.num_scales)
    ]
    return coefficients


@overload
def _apply_forward_transform(
    image: npt.NDArray[np.float32],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[False] = False,
) -> list[list[list[npt.NDArray[np.complex64]]]]: ...


@overload
def _apply_forward_transform(
    image: npt.NDArray[np.complex64],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[True],
) -> list[list[list[npt.NDArray[np.complex64]]]]: ...


def _apply_forward_transform(
    image: npt.NDArray[F] | npt.NDArray[C],
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
) -> list[list[list[npt.NDArray[np.complexfloating]]]]:
    """
    Apply forward Uniform Discrete Curvelet Transform (decomposition).

    This function decomposes an input image or volume into curvelet coefficients
    by applying frequency-domain windows and downsampling. The transform can
    operate in two modes: real transform (default) or complex transform.

    Parameters
    ----------
    image : npt.NDArray[F] | npt.NDArray[C]
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be either real-valued (npt.NDArray[F]) or
        complex-valued (npt.NDArray[C]).
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
          Each band is scaled by :math:`\\sqrt{0.5}`. Coefficients are complex-valued.
          Required for complex-valued inputs.

    Returns
    -------
    list[list[list[npt.NDArray[C]]]]
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands
          * Real mode: dim directions per scale
          * Complex mode: 2*dim directions per scale
        Each coefficient array has shape determined by decimation ratios.
        Coefficients have complex dtype matching the input:
        - np.float32 input -> np.complex64 coefficients
        - np.float64 input -> np.complex128 coefficients
        - np.complex64 input -> np.complex64 coefficients
        - np.complex128 input -> np.complex128 coefficients

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
    """
    if use_complex_transform:
        # Runtime check for complex arrays
        # The overloads ensure type safety at call sites
        if np.iscomplexobj(image):
            return _apply_forward_transform_complex(
                image, parameters, windows, decimation_ratios
            )
        # Fall through if not complex - try anyway for runtime flexibility
        # This handles edge cases where overloads can't determine type
        return _apply_forward_transform_complex(
            image,
            parameters,
            windows,
            decimation_ratios,
        )

    # Real transform mode
    # Runtime check for real arrays
    # The overloads ensure type safety at call sites
    if not np.iscomplexobj(image):
        return _apply_forward_transform_real(
            image, parameters, windows, decimation_ratios
        )

    # Complex image passed to real transform - raise error
    # This enforces type safety: real transform requires real input
    error_msg = (
        "Real transform requires real-valued input. "
        "Got complex array. Use transform_kind='complex' for complex inputs."
    )
    raise ValueError(error_msg)
