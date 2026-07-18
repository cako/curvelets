from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
import numpy as np
import numpy.typing as npt

from ._sparse_window import SparseWindow
from ._utils import ParamUDCT
from .typing import (
    _F,
    UDCTCoefficients,
    UDCTWindows,
    _IntegerNDArray,
    _to_complex_dtype,
)


def _process_wedge_backward_monogenic(
    coefficients: npt.NDArray[np.floating],
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    image_frequencies: list[npt.NDArray[np.complexfloating]],
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
    image_frequencies : list of ndarray
        Frequency accumulators for [scalar, riesz_1, ..., riesz_ndim]
        (modified in-place).
    """
    num_channels = coefficients.shape[-1]
    num_riesz = num_channels - 2
    complex_dtype = image_frequencies[0].dtype

    scalar_2ch = np.ascontiguousarray(coefficients[..., :2])
    coeff_scalar = scalar_2ch.view(complex_dtype).squeeze(-1)
    scale = float(np.sqrt(np.prod(decimation_ratio) / 2.0))

    window.synthesize(
        coeff_scalar,
        image_frequencies[0],
        scale,
        decimation=decimation_ratio,
    )

    for riesz_idx in range(num_riesz):
        coeff_riesz_k = coefficients[..., 2 + riesz_idx].astype(complex_dtype)
        window.synthesize(
            coeff_riesz_k,
            image_frequencies[1 + riesz_idx],
            scale,
            decimation=decimation_ratio,
        )


def _backward_lowpass_monogenic_periodized(
    low_coeffs: npt.NDArray[np.floating],
    window: SparseWindow,
    decimation_ratio: _IntegerNDArray,
    image_frequencies: list[npt.NDArray[np.complexfloating]],
) -> None:
    """Accumulate monogenic lowpass (scalar + Riesz) via tiled sparse FFT."""
    complex_dtype = image_frequencies[0].dtype
    num_riesz = low_coeffs.shape[-1] - 2
    scale = float(np.sqrt(np.prod(decimation_ratio)))

    low_scalar_2ch = np.ascontiguousarray(low_coeffs[..., :2])
    low_coeff_scalar = low_scalar_2ch.view(complex_dtype).squeeze(-1)
    window.synthesize(
        low_coeff_scalar,
        image_frequencies[0],
        scale,
        decimation=decimation_ratio,
    )

    for riesz_idx in range(num_riesz):
        low_coeff_riesz = low_coeffs[..., 2 + riesz_idx].astype(complex_dtype)
        window.synthesize(
            low_coeff_riesz,
            image_frequencies[1 + riesz_idx],
            scale,
            decimation=decimation_ratio,
        )


def _apply_backward_transform_monogenic(
    coefficients: UDCTCoefficients[_F],
    parameters: ParamUDCT,
    windows: UDCTWindows[_F],
    decimation_ratios: list[_IntegerNDArray],
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

    num_channels = first_coeff.shape[-1]
    num_components = num_channels - 1

    image_frequencies = [
        np.zeros(parameters.shape, dtype=complex_dtype) for _ in range(num_components)
    ]

    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:  # pylint: disable=too-many-nested-blocks
        image_frequencies_wavelet = [
            np.zeros(parameters.shape, dtype=complex_dtype)
            for _ in range(num_components)
        ]
        image_frequencies_other = [
            np.zeros(parameters.shape, dtype=complex_dtype)
            for _ in range(num_components)
        ]

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

                    coeffs = coefficients[scale_idx][direction_idx][wedge_idx]
                    targets = (
                        image_frequencies_wavelet
                        if scale_idx == highest_scale_idx
                        else image_frequencies_other
                    )
                    _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                        targets,
                    )

        for comp_idx in range(num_components):
            image_frequencies[comp_idx] = (
                2 * image_frequencies_other[comp_idx]
                + image_frequencies_wavelet[comp_idx]
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

                    coeffs = coefficients[scale_idx][direction_idx][wedge_idx]
                    _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                        image_frequencies,
                    )

        for comp_idx in range(num_components):
            image_frequencies[comp_idx] *= 2

    _backward_lowpass_monogenic_periodized(
        coefficients[0][0][0],
        windows[0][0][0],
        decimation_ratios[0][0],
        image_frequencies,
    )

    results = []
    scalar: npt.NDArray[_F] = np.fft.ifftn(image_frequencies[0]).real.astype(real_dtype)
    results.append(scalar)

    for comp_idx in range(1, num_components):
        riesz_k: npt.NDArray[_F] = -np.fft.ifftn(
            image_frequencies[comp_idx]
        ).real.astype(real_dtype)
        results.append(riesz_k)

    return tuple(results)
