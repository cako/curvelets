from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from ._fft import FFTBackend, resolve_fft_engine
from ._periodized_fft import half_spectrum_shape, hermitian_from_half
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import _C, UDCTCoefficients, UDCTWindows, _to_complex_dtype, _to_real_dtype


def _wedge_synthesize_scale(
    decimation_ratio: npt.NDArray[np.int_],
    *,
    complex_mode: bool = False,
) -> float:
    """Scale for periodized backward wedge (optional complex √0.5)."""
    scale = float(np.sqrt(np.prod(decimation_ratio) / 2.0))
    if complex_mode:
        scale *= float(np.sqrt(0.5))
    return scale


def _process_wedge_backward_real(
    coefficient: npt.NDArray[np.complexfloating],
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    image_frequency_mirror: npt.NDArray[np.complexfloating],
    factor: float = 1.0,
    fft: FFTBackend | None = None,
) -> None:
    """
    Accumulate one real-mode wedge into the half-spectrum accumulators.

    Parameters
    ----------
    coefficient : npt.NDArray[np.complexfloating]
        Downsampled coefficient array for this wedge.
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge (1D array with length equal to dimensions).
    image_frequency, image_frequency_mirror : npt.NDArray[np.complexfloating]
        Half-spectrum own / mirror accumulators (modified in-place).
    factor : float, optional
        Band combination factor (2 for curvelet bands, 1 for the wavelet band).
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.

    Notes
    -----
    Equivalence: ``fftn(upsample(c, d)) == tile(fftn(c), d)``.
    Scale is ``sqrt(prod(d) / 2)`` matching the former upsample / full-FFT path.
    """
    window.synthesize_half(
        coefficient,
        image_frequency,
        image_frequency_mirror,
        factor * _wedge_synthesize_scale(decimation_ratio),
        decimation=decimation_ratio,
        fft=fft,
    )


def _process_wedge_backward_complex(
    coefficient: npt.NDArray[np.complexfloating],
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    flip_window: bool = False,
    factor: float = 1.0,
    fft: FFTBackend | None = None,
) -> None:
    """
    Accumulate one complex-mode wedge into ``image_frequency``.

    Parameters
    ----------
    coefficient : npt.NDArray[np.complexfloating]
        Downsampled coefficient array for this wedge.
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : npt.NDArray[np.int_]
        Decimation ratio for this wedge.
    image_frequency : npt.NDArray[np.complexfloating]
        Full-size frequency accumulator (modified in-place).
    flip_window : bool, optional
        If True, use flipped (negative-frequency) window indices.
    factor : float, optional
        Band combination factor (2 for curvelet bands, 1 for the wavelet band).
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.
    """
    window.synthesize(
        coefficient,
        image_frequency,
        factor * _wedge_synthesize_scale(decimation_ratio, complex_mode=True),
        flip=flip_window,
        decimation=decimation_ratio,
        fft=fft,
    )


def _backward_lowpass_periodized(
    coefficient: npt.NDArray[np.complexfloating],
    window: SparseWindow,
    decimation_ratio: npt.NDArray[np.int_],
    image_frequency: npt.NDArray[np.complexfloating],
    fft: FFTBackend | None = None,
) -> None:
    """Accumulate lowpass band via small FFT + tiled sparse scatter."""
    window.synthesize(
        coefficient,
        image_frequency,
        float(np.sqrt(np.prod(decimation_ratio))),
        decimation=decimation_ratio,
        fft=fft,
    )


def _band_factor(
    scale_idx: int, highest_scale_idx: int, wavelet_highest: bool
) -> float:
    """
    Weight of a band in the reconstruction.

    Curvelet bands count twice (their windows cover one of each ``±k`` pair);
    the wavelet-mode highest band covers both and counts once.
    """
    return 1.0 if wavelet_highest and scale_idx == highest_scale_idx else 2.0


def _decimation_for(
    decimation_ratios: list[npt.NDArray[np.int_]], scale_idx: int, direction_idx: int
) -> npt.NDArray[np.int_]:
    ratios = decimation_ratios[scale_idx]
    return ratios[0, :] if ratios.shape[0] == 1 else ratios[direction_idx, :]


def _apply_backward_transform_real(
    coefficients: UDCTCoefficients[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    fft: FFTBackend | None = None,
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform in real mode.

    This function reconstructs a real-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The real transform mode processes combined positive/negative
    frequency bands.

    Parameters
    ----------
    coefficients : UDCTCoefficients[_C]
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
    in real-valued output. Contributions are accumulated on the ``rfftn`` half
    spectrum (its Hermitian part, see
    :func:`~curvelets.numpy._periodized_fft.hermitian_from_half`), and a single
    ``irfftn`` returns ``ifftn(F).real`` of the full-spectrum accumulation.
    """
    real_dtype = _to_real_dtype(coefficients[0][0][0].dtype)
    complex_dtype = _to_complex_dtype(real_dtype)
    if fft is None:
        fft = resolve_fft_engine("auto", parameters.shape)

    highest_scale_idx = parameters.num_scales - 1
    wavelet_highest = len(windows[highest_scale_idx]) == 1

    half_shape = half_spectrum_shape(parameters.shape)
    own = np.zeros(half_shape, dtype=complex_dtype)
    mirror = np.zeros(half_shape, dtype=complex_dtype)
    for scale_idx in range(1, parameters.num_scales):
        factor = _band_factor(scale_idx, highest_scale_idx, wavelet_highest)
        for direction_idx, direction_windows in enumerate(windows[scale_idx]):
            decimation_ratio = _decimation_for(
                decimation_ratios, scale_idx, direction_idx
            )
            for wedge_idx, window in enumerate(direction_windows):
                _process_wedge_backward_real(
                    coefficients[scale_idx][direction_idx][wedge_idx],
                    window,
                    decimation_ratio,
                    own,
                    mirror,
                    factor,
                    fft,
                )

    dec0 = decimation_ratios[0][0]
    windows[0][0][0].synthesize_half(
        coefficients[0][0][0],
        own,
        mirror,
        float(np.sqrt(np.prod(dec0))),
        decimation=dec0,
        fft=fft,
    )
    spectrum = hermitian_from_half(own, mirror)
    axes = tuple(range(parameters.ndim))
    return fft.irfftn(spectrum, s=parameters.shape, axes=axes)  # type: ignore[no-any-return]


def _apply_backward_transform_complex(
    coefficients: UDCTCoefficients[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    fft: FFTBackend | None = None,
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform in complex mode.

    This function reconstructs a complex-valued image or volume from curvelet
    coefficients by upsampling, applying frequency-domain windows, and combining
    all bands. The complex transform mode processes positive and negative
    frequency bands separately.

    Parameters
    ----------
    coefficients : UDCTCoefficients[_C]
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
    in complex-valued output. Contributions are accumulated using sparse tiled
    FFTs (no dense window conversion).
    """
    real_dtype = _to_real_dtype(coefficients[0][0][0].dtype)
    complex_dtype = _to_complex_dtype(real_dtype)
    if fft is None:
        fft = resolve_fft_engine("auto", parameters.shape)

    highest_scale_idx = parameters.num_scales - 1
    wavelet_highest = len(windows[highest_scale_idx]) == 1

    image_frequency = np.zeros(parameters.shape, dtype=complex_dtype)
    for flip_window in (False, True):
        coeff_offset = parameters.ndim if flip_window else 0
        for scale_idx in range(1, parameters.num_scales):
            factor = _band_factor(scale_idx, highest_scale_idx, wavelet_highest)
            num_window_directions = len(windows[scale_idx])
            for direction_idx in range(parameters.ndim):
                # The wavelet-mode highest band has one window, stored once per
                # flip in direction 0.
                if (
                    wavelet_highest
                    and scale_idx == highest_scale_idx
                    and direction_idx != 0
                ):
                    continue
                window_direction_idx = min(direction_idx, num_window_directions - 1)
                decimation_ratio = _decimation_for(
                    decimation_ratios, scale_idx, window_direction_idx
                )
                for wedge_idx, window in enumerate(
                    windows[scale_idx][window_direction_idx]
                ):
                    _process_wedge_backward_complex(
                        coefficients[scale_idx][direction_idx + coeff_offset][
                            wedge_idx
                        ],
                        window,
                        decimation_ratio,
                        image_frequency,
                        flip_window=flip_window,
                        factor=factor,
                        fft=fft,
                    )

    _backward_lowpass_periodized(
        coefficients[0][0][0],
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequency,
        fft=fft,
    )
    return fft.ifftn(image_frequency)  # type: ignore[no-any-return]


@overload
def _apply_backward_transform(
    coefficients: UDCTCoefficients[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[True],
    fft: FFTBackend | None = None,
) -> np.ndarray: ...


@overload
def _apply_backward_transform(
    coefficients: UDCTCoefficients[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: Literal[False] = False,
    fft: FFTBackend | None = None,
) -> np.ndarray: ...


def _apply_backward_transform(
    coefficients: UDCTCoefficients[_C],
    parameters: ParamUDCT,
    windows: UDCTWindows[np.floating],
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
    fft: FFTBackend | None = None,
) -> np.ndarray:
    """
    Apply backward Uniform Discrete Curvelet Transform (reconstruction).

    This function reconstructs an image or volume from curvelet coefficients
    by upsampling, applying frequency-domain windows, and combining all bands.
    The transform can operate in two modes: real transform (default) or complex
    transform, matching the mode used in the forward transform.

    Parameters
    ----------
    coefficients : UDCTCoefficients[_C]
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
        windows[scale][direction][wedge] = SparseWindow
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
    Uses periodized sparse FFTs (small FFT + tiled sparse scatter) instead of
    full-size upsample / FFT. Complex negative-frequency wedges use index
    remapping equivalent to ``flip_fft_all_axes``.
    """
    if use_complex_transform:
        return _apply_backward_transform_complex(
            coefficients, parameters, windows, decimation_ratios, fft
        )
    return _apply_backward_transform_real(
        coefficients, parameters, windows, decimation_ratios, fft
    )
