from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
import numpy as np
import numpy.typing as npt

from ._fft import FFTBackend, resolve_fft_engine
from ._periodized_fft import half_spectrum_shape, hermitian_from_half
from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import (
    _F,
    UDCTCoefficients,
    UDCTWindows,
    _IntegerNDArray,
    _to_complex_dtype,
)

_HalfAccumulator = tuple[
    npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]
]


def _synthesize_monogenic_band(
    coefficients: npt.NDArray[np.floating],
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    accumulators: list[_HalfAccumulator],
    scale: float,
    fft: FFTBackend | None,
) -> None:
    """Accumulate the scalar and Riesz channels of one band (half spectrum)."""
    complex_dtype = accumulators[0][0].dtype
    scalar_2ch = np.ascontiguousarray(coefficients[..., :2])
    channels = [scalar_2ch.view(complex_dtype).squeeze(-1)] + [
        coefficients[..., 2 + riesz_idx].astype(complex_dtype)
        for riesz_idx in range(coefficients.shape[-1] - 2)
    ]
    for channel, (own, mirror) in zip(channels, accumulators, strict=True):
        window.synthesize_half(
            channel, own, mirror, scale, decimation=decimation_ratio, fft=fft
        )


def _process_wedge_backward_monogenic(
    coefficients: npt.NDArray[np.floating],
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    accumulators: list[_HalfAccumulator],
    factor: float = 1.0,
    fft: FFTBackend | None = None,
) -> None:
    """
    Accumulate one monogenic wedge into component frequency buffers.

    Uses the discrete tight frame property of UDCT with periodized sparse FFTs:
    - scalar: c₀ · W → reconstructs f
    - riesz_k: cₖ · W → reconstructs Rₖf (will be negated in caller for -Rₖf)

    Parameters
    ----------
    coefficients : npt.NDArray[np.floating]
        Coefficient array with shape (*wedge_shape, ndim+2). All real dtype:
        - Channel 0: scalar.real
        - Channel 1: scalar.imag
        - Channels 2..ndim+1: Riesz components
        Complex scalar is reconstructed via .view(complex_dtype) on channels 0:2.
    window : SparseWindow
        Sparse window representation.
    decimation_ratio : _IntegerNDArray
        Decimation ratio for this wedge.
    accumulators : list of (ndarray, ndarray)
        Half-spectrum (own, mirror) accumulators for
        [scalar, riesz_1, ..., riesz_ndim] (modified in-place).
    factor : float, optional
        Band combination factor (2 for curvelet bands, 1 for the wavelet band).
    fft : FFTBackend, optional
        FFT implementation; defaults to ``numpy.fft``.
    """
    scale = factor * float(np.sqrt(np.prod(decimation_ratio) / 2.0))
    _synthesize_monogenic_band(
        coefficients, window, decimation_ratio, accumulators, scale, fft
    )


def _apply_backward_transform_monogenic(
    coefficients: UDCTCoefficients[_F],
    parameters: ParamUDCT,
    windows: UDCTWindows[_F],
    decimation_ratios: list[_IntegerNDArray],
    fft: FFTBackend | None = None,
) -> tuple[npt.NDArray[_F], ...]:
    """
    Apply backward monogenic curvelet transform.

    This uses the discrete tight frame property of UDCT rather than the
    continuous quaternion formula from Storath 2010. The result satisfies:
    backward(forward(f)) with transform_kind="monogenic" ≈ monogenic(f)

    Where monogenic(f) = (f, -R₁f, -R₂f, ..., -Rₙf) for N-D signals.

    The reconstruction uses the partition of unity property:
    - scalar: ∑ c₀ · W = f
    - riesz_k: -∑ cₖ · W = -Rₖf for k = 1, 2, ..., ndim

    The monogenic curvelet transform was originally defined for 2D signals by
    Storath 2010 using quaternions, but this implementation extends it to arbitrary
    N-D signals by using all Riesz transform components.

    Parameters
    ----------
    coefficients : UDCTCoefficients[np.floating]
        Monogenic curvelet coefficients from forward_monogenic().
        Each coefficient array has shape (*wedge_shape, ndim+2) with real dtype:
        - Channel 0: scalar.real
        - Channel 1: scalar.imag
        - Channels 2..ndim+1: Riesz components
        Complex scalar is reconstructed via .view(complex_dtype) on channels 0:2.
    parameters : ParamUDCT
        UDCT parameters.
    windows : UDCTWindows[np.floating]
        Curvelet windows in sparse format.
    decimation_ratios : list[_IntegerNDArray]
        Decimation ratios for each scale and direction.

    Returns
    -------
    tuple[npt.NDArray[_F], ...]
        Reconstructed components: (scalar, riesz1, riesz2, ..., riesz_ndim)
        - scalar: Original input :math:`f`
        - riesz_k: :math:`-R_k f` for :math:`k = 1, 2, \\ldots, \\text{ndim}`
    """
    first_coeff = coefficients[0][0][0]
    real_dtype = first_coeff.dtype
    complex_dtype = _to_complex_dtype(real_dtype)
    num_components = first_coeff.shape[-1] - 1
    if fft is None:
        fft = resolve_fft_engine(None, parameters.shape)

    half_shape = half_spectrum_shape(parameters.shape)
    accumulators: list[_HalfAccumulator] = [
        (
            np.zeros(half_shape, dtype=complex_dtype),
            np.zeros(half_shape, dtype=complex_dtype),
        )
        for _ in range(num_components)
    ]

    highest_scale_idx = parameters.num_scales - 1
    wavelet_highest = len(windows[highest_scale_idx]) == 1
    for scale_idx in range(1, parameters.num_scales):
        # Curvelet bands count twice; the wavelet-mode highest band once.
        factor = 1.0 if wavelet_highest and scale_idx == highest_scale_idx else 2.0
        for direction_idx, direction_windows in enumerate(windows[scale_idx]):
            if decimation_ratios[scale_idx].shape[0] == 1:
                decimation_ratio = decimation_ratios[scale_idx][0, :]
            else:
                decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
            for wedge_idx, window in enumerate(direction_windows):
                _process_wedge_backward_monogenic(
                    coefficients[scale_idx][direction_idx][wedge_idx],
                    window,
                    decimation_ratio,
                    accumulators,
                    factor,
                    fft,
                )

    dec0 = decimation_ratios[0][0]
    _synthesize_monogenic_band(
        coefficients[0][0][0],
        windows[0][0][0],
        dec0,
        accumulators,
        float(np.sqrt(np.prod(dec0))),
        fft,
    )

    axes = tuple(range(parameters.ndim))
    results: list[npt.NDArray[_F]] = []
    for comp_idx, (own, mirror) in enumerate(accumulators):
        component = fft.irfftn(
            hermitian_from_half(own, mirror), s=parameters.shape, axes=axes
        )
        if comp_idx > 0:
            component = -component  # Riesz components are -R_k f
        results.append(component.astype(real_dtype, copy=False))
    return tuple(results)
