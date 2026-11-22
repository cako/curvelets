from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from ._fft import FFTBackend, resolve_fft_engine
from ._periodized_fft import half_spectrum_source
from ._riesz import riesz_filters
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import (
    _C,
    _F,
    UDCTCoefficients,
    UDCTWindows,
    _IntegerNDArray,
)


def _wedge_analyze_scale(
    decimation_ratio: npt.NDArray[np.int_],
    *,
    complex_mode: bool = False,
    symmetric: bool = False,
) -> float:
    """Scale for periodized forward band (1/√D if symmetric, else √(2/D), optional complex √0.5)."""
    prod_d = float(np.prod(decimation_ratio))
    if symmetric:
        return float(1.0 / np.sqrt(prod_d))
    scale = float(np.sqrt(2.0 * prod_d) / prod_d)
    if complex_mode:
        scale *= float(np.sqrt(0.5))
    return scale


def _process_wedge_real(
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    fft: FFTBackend | None = None,
    *,
    symmetric: bool = False,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for real transform mode.

    Uses a periodized (folded) sparse IFFT instead of a full-size IFFT
    followed by downsampling.

    Parameters
    ----------
    window : SparseWindow
        Sparse window representation (preferably with ``folded_indices``).
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : npt.NDArray[np.complexfloating]
        Half-spectrum source of the input image (``half_spectrum_source`` of
        its ``rfftn``).
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.
    symmetric : bool, optional
        If True, use symmetric-band scaling ``1 / sqrt(prod(d))`` (wavelet
        highest scale) instead of directional wedge scaling. Default is False.

    Returns
    -------
    npt.NDArray[np.complexfloating]
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
        _wedge_analyze_scale(decimation_ratio, symmetric=symmetric),
        decimation=decimation_ratio,
        half=True,
        fft=fft,
    )


def _process_wedge_complex(
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    flip_window: bool = False,
    fft: FFTBackend | None = None,
    *,
    symmetric: bool = False,
) -> npt.NDArray[np.complexfloating]:
    """
    Process a single wedge for complex transform mode.

    Uses a periodized sparse IFFT and optional index-space window flip
    (no dense ``to_dense`` / ``flip_fft_all_axes``).

    Parameters
    ----------
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency : npt.NDArray[np.complexfloating]
        Input image in frequency domain (from FFT).
    flip_window : bool, optional
        If True, flip the window for negative frequency processing.
        Default is False.
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.
    symmetric : bool, optional
        If True, use symmetric-band scaling ``1 / sqrt(prod(d))`` (wavelet
        highest scale) instead of directional wedge scaling. Default is False.

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
    return window.analyze(
        image_frequency,
        _wedge_analyze_scale(decimation_ratio, complex_mode=True, symmetric=symmetric),
        flip=flip_window,
        decimation=decimation_ratio,
        fft=fft,
    )


def _forward_lowpass_periodized(
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    scale_norm: float,
    *,
    half: bool = False,
    fft: FFTBackend | None = None,
) -> npt.NDArray[np.complexfloating]:
    """Lowpass forward via folded sparse IFFT (no wedge ``sqrt(2)`` factor)."""
    scale = scale_norm / float(np.prod(decimation_ratio))
    return window.analyze(
        image_frequency,
        scale,
        decimation=decimation_ratio,
        half=half,
        fft=fft,
    )


def _apply_forward_transform_real(
    image: npt.NDArray[_F],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.complexfloating]:
    """
    Apply forward Uniform Discrete Curvelet Transform in real mode.

    This function decomposes an input image or volume into real-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Each
    curvelet band captures both positive and negative frequencies combined.

    Parameters
    ----------
    image : npt.NDArray[_F]
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
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients[_C]
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
    if fft is None:
        fft = resolve_fft_engine(None, parameters.shape)
    # Real input: gather from the rfftn half spectrum (Hermitian symmetry).
    image_frequency = half_spectrum_source(fft.rfftn(image))

    scale_norm = float(
        np.sqrt(
            np.prod(
                np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
            )
        )
    )
    low_freq_coeff = _forward_lowpass_periodized(
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
        scale_norm,
        half=True,
        fft=fft,
    )

    coefficients: UDCTCoefficients[np.complexfloating] = [[[low_freq_coeff]]] + [
        [
            [
                _process_wedge_real(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    fft,
                    symmetric=len(windows[scale_idx]) == 1,
                )
                for wedge_idx in range(len(windows[scale_idx][direction_idx]))
            ]
            for direction_idx in range(len(windows[scale_idx]))
        ]
        for scale_idx in range(1, parameters.num_scales)
    ]
    return coefficients


def _apply_forward_transform_complex(
    image: npt.NDArray[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.complexfloating]:
    """
    Apply forward Uniform Discrete Curvelet Transform in complex mode.

    This function decomposes an input image or volume into complex-valued curvelet
    coefficients by applying frequency-domain windows and downsampling. Positive
    and negative frequency bands are separated into different directions (except
    the symmetric wavelet-mode highest scale, which has 1 direction).

    Parameters
    ----------
    image : npt.NDArray[_C]
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
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0

    Returns
    -------
    UDCTCoefficients[_C]
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (2*ndim directions per
          curvelet scale; 1 direction at the wavelet-mode highest scale)
          * Directions 0..dim-1 are positive frequencies
          * Directions dim..2*dim-1 are negative frequencies
        Each coefficient array has shape determined by decimation ratios.
        Coefficients have the same complex dtype as input (C).

    Notes
    -----
    The complex transform separates positive and negative frequencies into
    different directions. Each curvelet band is scaled by :math:`\\sqrt{0.5}` to
    maintain energy preservation. The negative frequency windows are obtained by
    flipping indices via ``flip_fft_indices`` (equivalent to ``flip_fft_all_axes``).

    This mode is required for complex-valued inputs and provides full frequency
    information.
    """
    if fft is None:
        fft = resolve_fft_engine(None, parameters.shape)
    image_frequency = fft.fftn(image)

    scale_norm = float(
        np.sqrt(
            np.prod(
                np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
            )
        )
    )
    low_freq_coeff = _forward_lowpass_periodized(
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
        scale_norm,
        fft=fft,
    )

    coefficients: UDCTCoefficients[np.complexfloating] = [[[low_freq_coeff]]]
    for scale_idx in range(1, parameters.num_scales):
        if len(windows[scale_idx]) == 1:
            decimation_ratio = decimation_ratios[scale_idx][0, :]
            coefficients.append([
                [
                    _process_wedge_complex(
                        window,
                        decimation_ratio,
                        image_frequency,
                        flip_window=False,
                        fft=fft,
                        symmetric=True,
                    )
                    for window in windows[scale_idx][0]
                ]
            ])
            continue
        coefficients.append([
            [
                _process_wedge_complex(
                    window,
                    decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    flip_window=flip_window,
                    fft=fft,
                )
                for window in windows[scale_idx][direction_idx]
            ]
            for flip_window in (False, True)
            for direction_idx in range(parameters.ndim)
        ])
    return coefficients


def _monogenic_channels(
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    image_frequency: npt.NDArray[np.complexfloating],
    riesz_filters_list: list[npt.NDArray[np.complexfloating]],
    scale: float,
    *,
    half: bool,
    fft: FFTBackend | None,
) -> npt.NDArray[np.floating]:
    """Scalar (2 channels) and Riesz channels of one band, gathering once."""
    ifftn = np.fft.ifftn if fft is None else fft.ifftn
    samples = window.gather(
        image_frequency, scale=scale, half=half, decimation=decimation_ratio
    )
    real_dtype = samples.real.dtype
    coeff_scalar = ifftn(window.fold(samples, decimation=decimation_ratio))
    channels = [coeff_scalar.real, coeff_scalar.imag]
    for riesz_filter in riesz_filters_list:
        riesz_at_indices = riesz_filter.reshape(-1)[window.indices]
        folded = window.fold(
            samples * riesz_at_indices.astype(samples.dtype, copy=False),
            decimation=decimation_ratio,
        )
        channels.append(ifftn(folded).real)
    return np.stack(channels, axis=-1).astype(real_dtype, copy=False)


def _process_wedge_monogenic(
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    image_frequency: npt.NDArray[np.complexfloating],
    riesz_filters_list: list[npt.NDArray[np.complexfloating]],
    fft: FFTBackend | None = None,
    *,
    half: bool = False,
    symmetric: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Process a single wedge for monogenic transform.

    Uses periodized sparse IFFTs for the scalar and each Riesz channel.

    Parameters
    ----------
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : _IntegerNDArray
        Decimation ratio for this wedge (1D array with length equal to dimensions).
        Uses _IntegerNDArray type alias from typing.py.
    image_frequency : npt.NDArray[np.complexfloating]
        Input image in frequency domain (from FFT), or its half-spectrum
        source when ``half=True``.
    riesz_filters_list : list[npt.NDArray[np.complexfloating]]
        Riesz transform filters R_1, R_2, ... R_ndim from riesz_filters().
        Each filter has the full image shape.
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.
    half : bool, optional
        If True, ``image_frequency`` is a half-spectrum source.
    symmetric : bool, optional
        If True, use symmetric-band scaling ``1 / sqrt(prod(d))`` (wavelet
        highest scale) instead of directional wedge scaling. Default is False.

    Returns
    -------
    npt.NDArray[np.floating]
        Array with shape (*wedge_shape, ndim+2) containing stacked components.
        All values are real dtype:
        - Channel 0: scalar.real
        - Channel 1: scalar.imag
        - Channels 2..ndim+1: Riesz components (already real)
        Complex scalar can be reconstructed via .view(complex_dtype) on channels 0:2.

    Notes
    -----
    The components are:
    - Scalar: IFFT(FFT(image) * window) - same as standard UDCT (stored as 2 real channels)
    - Riesz_k: IFFT(FFT(image) * window * R_k_filter) for k = 1, 2, ..., ndim

    All components use the same decimation ratios and normalization factors
    as the standard UDCT transform.
    """
    return _monogenic_channels(
        window,
        decimation_ratio,
        image_frequency,
        riesz_filters_list,
        _wedge_analyze_scale(decimation_ratio, symmetric=symmetric),
        half=half,
        fft=fft,
    )


def _apply_forward_transform_monogenic(
    image: npt.NDArray[_F],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[_IntegerNDArray],
    riesz_filters_list: list[npt.NDArray[np.complexfloating]] | None = None,
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.floating]:
    """
    Apply forward monogenic curvelet transform.

    This function decomposes a real-valued input image or volume into monogenic
    curvelet coefficients by applying frequency-domain windows and Riesz transforms.
    Each coefficient band produces ndim+2 components stacked along the last axis:
    scalar (stored as 2 real channels) plus all Riesz components (one per dimension).

    The monogenic curvelet transform was originally defined for 2D signals by
    Storath 2010 using quaternions, but this implementation extends it to arbitrary
    N-D signals by using all Riesz transform components. The reconstruction uses the
    discrete tight frame property of UDCT rather than quaternion multiplication,
    making the N-D extension straightforward.

    Parameters
    ----------
    image : npt.NDArray[_F]
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be real-valued (floating point dtype).
        Uses the F TypeVar from typing.py (np.float16, np.float32, np.float64, np.longdouble).
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
        Type alias from typing.py.
    decimation_ratios : list[_IntegerNDArray]
        Decimation ratios for each scale and direction. Structure:
        - decimation_ratios[0]: shape (1, dim) for low-frequency band
        - decimation_ratios[scale]: shape (dim, dim) for scale > 0
        Uses _IntegerNDArray type alias from typing.py.
    riesz_filters_list : list of ndarray, optional
        Precomputed Riesz filters. If None, computed via ``riesz_filters``.

    Returns
    -------
    UDCTCoefficients[np.floating]
        Monogenic coefficients as nested list structure with arrays of shape
        (*wedge_shape, ndim+2). Same structure as standard UDCT but with extra
        channel dimension. All values are real dtype:
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands (ndim directions per scale)
        Each coefficient array has shape (*decimated_shape, ndim+2) where:
        - channel 0: scalar.real
        - channel 1: scalar.imag
        - channels 2..ndim+1: Riesz components (already real)
        Complex scalar can be reconstructed via .view(complex_dtype) on channels 0:2.

    Notes
    -----
    The monogenic transform is mathematically defined only for real-valued functions.
    This function computes:
    - Scalar component: same as standard UDCT (stored as 2 real channels)
    - Riesz_k component: applies :math:`R_k` filter :math:`(i \\xi_k / |\\xi|)` for :math:`k = 1, 2, \\ldots, \\text{ndim}`

    Structure mirrors _apply_forward_transform_real() but:
    - Computes Riesz filters once at the start
    - Processes each wedge to produce ndim+2 components stacked along last axis
    - Uses type aliases from typing.py for consistency with rest of codebase

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
    >>> coeffs[0][0][0].shape[-1]  # Last dimension has ndim+2 channels (4 for 2D)
    4
    """
    if fft is None:
        fft = resolve_fft_engine(None, parameters.shape)
    # Real input: gather from the rfftn half spectrum (Hermitian symmetry).
    image_frequency = half_spectrum_source(fft.rfftn(image))

    if riesz_filters_list is None:
        riesz_filters_list = riesz_filters(parameters.shape)

    scale_norm = float(
        np.sqrt(
            np.prod(
                np.full((parameters.ndim,), fill_value=2 ** (parameters.num_scales - 2))
            )
        )
    )
    dec0 = decimation_ratios[0][0]
    low_freq_coeff = _monogenic_channels(
        windows[0][0][0],
        dec0,
        image_frequency,
        riesz_filters_list,
        scale_norm / float(np.prod(dec0)),
        half=True,
        fft=fft,
    )

    coefficients: UDCTCoefficients[np.floating] = [[[low_freq_coeff]]] + [
        [
            [
                _process_wedge_monogenic(
                    windows[scale_idx][direction_idx][wedge_idx],
                    decimation_ratios[scale_idx][0, :]
                    if decimation_ratios[scale_idx].shape[0] == 1
                    else decimation_ratios[scale_idx][direction_idx, :],
                    image_frequency,
                    riesz_filters_list,
                    fft,
                    half=True,
                    symmetric=len(windows[scale_idx]) == 1,
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
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[False] = False,
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.complex64]: ...


@overload
def _apply_forward_transform(
    image: npt.NDArray[np.complex64],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[True],
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.complex64]: ...


def _apply_forward_transform(
    image: npt.NDArray[_F] | npt.NDArray[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
    fft: FFTBackend | None = None,
) -> UDCTCoefficients[np.complexfloating]:
    """
    Apply forward Uniform Discrete Curvelet Transform (decomposition).

    This function decomposes an input image or volume into curvelet coefficients
    by applying frequency-domain windows and downsampling. The transform can
    operate in two modes: real transform (default) or complex transform.

    Parameters
    ----------
    image : npt.NDArray[_F] | npt.NDArray[_C]
        Input image or volume to decompose. Must have shape matching
        `parameters.shape`. Must be either real-valued (npt.NDArray[_F]) or
        complex-valued (npt.NDArray[_C]).
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
    UDCTCoefficients[_C]
        Curvelet coefficients as nested list structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..(num_scales-1): High-frequency bands
          * Real mode: dim directions per curvelet scale (1 in wavelet mode)
          * Complex mode: 2*dim directions per curvelet scale (1 in wavelet mode)
        Each coefficient array has shape determined by decimation ratios.
        Coefficients have complex dtype matching the input:
        - np.float32 input -> np.complex64 coefficients
        - np.float64 input -> np.complex128 coefficients
        - np.complex64 input -> np.complex64 coefficients
        - np.complex128 input -> np.complex128 coefficients

    Notes
    -----
    Uses periodized sparse FFTs (fold + small IFFT) instead of full-size
    IFFTs followed by downsampling. Complex negative-frequency wedges use
    index remapping equivalent to ``flip_fft_all_axes``.
    """
    if use_complex_transform:
        if np.iscomplexobj(image):
            return _apply_forward_transform_complex(
                image, parameters, windows, decimation_ratios, fft
            )
        return _apply_forward_transform_complex(
            image, parameters, windows, decimation_ratios, fft
        )

    if not np.iscomplexobj(image):
        return _apply_forward_transform_real(
            image, parameters, windows, decimation_ratios, fft
        )

    error_msg = (
        "Real transform requires real-valued input. "
        "Got complex array. Use transform_kind='complex' for complex inputs."
    )
    raise ValueError(error_msg)
