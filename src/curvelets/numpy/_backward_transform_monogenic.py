from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
import numpy as np
import numpy.typing as npt

from ._typing import (
    F,
    IntegerNDArray,
    IntpNDArray,
    UDCTCoefficients,
    UDCTWindows,
    _to_complex_dtype,
)
from ._utils import ParamUDCT, upsample


def _process_wedge_backward_monogenic(
    coefficients: npt.NDArray[np.floating],
    window: tuple[IntpNDArray, npt.NDArray[np.floating]],
    decimation_ratio: IntegerNDArray,
    complex_dtype: npt.DTypeLike,
) -> list[npt.NDArray[np.complexfloating]]:
    """
    Process a single wedge for monogenic backward transform.

    Uses the discrete tight frame property of UDCT:
    - scalar: c₀ · W → reconstructs f
    - riesz_k: cₖ · W → reconstructs Rₖf (will be negated in caller for -Rₖf)

    This is simpler than the continuous quaternion formula because the UDCT
    windows satisfy partition of unity ∑|W|² = 1, making each component
    independently reconstructable.

    Parameters
    ----------
    coefficients : npt.NDArray[np.floating]
        Coefficient array with shape (*wedge_shape, ndim+2). All real dtype:
        - Channel 0: scalar.real
        - Channel 1: scalar.imag
        - Channels 2..ndim+1: Riesz components
        Complex scalar is reconstructed via .view(complex_dtype) on channels 0:2.
    window : tuple[IntpNDArray, npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : IntegerNDArray
        Decimation ratio for this wedge.
    complex_dtype : npt.DTypeLike
        Complex dtype for output.

    Returns
    -------
    list[npt.NDArray[np.complexfloating]]
        List of frequency-domain contributions: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
        Each is sparse (only non-zero at window indices).
    """
    # ndim+2 channels: [scalar.real, scalar.imag, riesz_1, ..., riesz_ndim]
    # Number of Riesz components = ndim = coefficients.shape[-1] - 2
    num_channels = coefficients.shape[-1]
    num_riesz = num_channels - 2  # ndim

    # Reconstruct complex scalar from first 2 channels via .view()
    scalar_2ch = np.ascontiguousarray(coefficients[..., :2])
    coeff_scalar = scalar_2ch.view(complex_dtype).squeeze(-1)

    # Upsample scalar coefficient to full size
    curvelet_band_scalar = upsample(coeff_scalar, decimation_ratio)

    # Undo normalization: divide by sqrt(2 * prod(decimation_ratio))
    # This matches the standard backward transform normalization
    norm_factor = np.sqrt(2 * np.prod(decimation_ratio))
    curvelet_band_scalar /= norm_factor

    # Transform to frequency domain
    curvelet_freq_scalar = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band_scalar)

    # Get window indices and values
    idx, val = window
    window_values = val.astype(complex_dtype)

    # Initialize contribution array for scalar
    contribution_scalar = np.zeros(curvelet_freq_scalar.shape, dtype=complex_dtype)
    contribution_scalar.flat[idx] = curvelet_freq_scalar.flat[idx] * window_values

    # Process all Riesz components (channels 2 onwards)
    contributions = [contribution_scalar]
    for riesz_idx in range(num_riesz):
        coeff_riesz_k = coefficients[..., 2 + riesz_idx]
        # Upsample Riesz coefficient to full size
        curvelet_band_riesz = upsample(
            coeff_riesz_k.astype(complex_dtype), decimation_ratio
        )
        # Undo normalization
        curvelet_band_riesz /= norm_factor
        # Transform to frequency domain
        curvelet_freq_riesz = np.prod(decimation_ratio) * np.fft.fftn(
            curvelet_band_riesz
        )
        # Initialize contribution array and apply window
        contribution_riesz = np.zeros(curvelet_freq_riesz.shape, dtype=complex_dtype)
        contribution_riesz.flat[idx] = curvelet_freq_riesz.flat[idx] * window_values
        contributions.append(contribution_riesz)

    return contributions


def _apply_backward_transform_monogenic(
    coefficients: UDCTCoefficients[F],
    parameters: ParamUDCT,
    windows: UDCTWindows[F],
    decimation_ratios: list[IntegerNDArray],
) -> tuple[npt.NDArray[F], ...]:
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
    decimation_ratios : list[IntegerNDArray]
        Decimation ratios for each scale and direction.

    Returns
    -------
    tuple[npt.NDArray[F], ...]
        Reconstructed components: (scalar, riesz1, riesz2, ..., riesz_ndim)
        - scalar: Original input :math:`f`
        - riesz_k: :math:`-R_k f` for :math:`k = 1, 2, \\ldots, \\text{ndim}`
    """
    # Determine dtype from coefficients - access first wedge array
    first_coeff = coefficients[0][0][0]  # Shape: (*wedge_shape, ndim+2)
    real_dtype = first_coeff.dtype
    complex_dtype = _to_complex_dtype(real_dtype)

    # Determine number of output components (ndim+1): scalar + riesz_1..riesz_ndim
    # Input has ndim+2 channels: [scalar.real, scalar.imag, riesz_1, ..., riesz_ndim]
    num_channels = first_coeff.shape[-1]
    num_components = num_channels - 1  # ndim+1 (scalar + ndim Riesz)

    # Initialize frequency domain arrays for all components dynamically
    image_frequencies = [
        np.zeros(parameters.shape, dtype=complex_dtype) for _ in range(num_components)
    ]

    # Process high-frequency bands
    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:  # pylint: disable=too-many-nested-blocks
        # Separate handling for wavelet mode at highest scale
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
                    contributions = _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                        complex_dtype,
                    )

                    idx, _ = window
                    if scale_idx == highest_scale_idx:
                        for comp_idx, contrib in enumerate(contributions):
                            image_frequencies_wavelet[comp_idx].flat[idx] += (
                                contrib.flat[idx]
                            )
                    else:
                        for comp_idx, contrib in enumerate(contributions):
                            image_frequencies_other[comp_idx].flat[idx] += contrib.flat[
                                idx
                            ]

        # Combine with factor of 2 for real transform mode
        for comp_idx in range(num_components):
            image_frequencies[comp_idx] = (
                2 * image_frequencies_other[comp_idx]
                + image_frequencies_wavelet[comp_idx]
            )
    else:
        # Normal curvelet mode
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

                    coeffs = coefficients[scale_idx][direction_idx][wedge_idx]
                    contributions = _process_wedge_backward_monogenic(
                        coeffs,
                        window,
                        decimation_ratio,
                        complex_dtype,
                    )

                    idx, _ = window
                    for comp_idx, contrib in enumerate(contributions):
                        image_frequencies[comp_idx].flat[idx] += contrib.flat[idx]

        # Multiply by 2 for real transform mode
        for comp_idx in range(num_components):
            image_frequencies[comp_idx] *= 2

    # Process low-frequency band
    # Use same structure as standard backward transform for consistency
    decimation_ratio = decimation_ratios[0][0]
    idx, val = windows[0][0][0]
    window_values = val.astype(complex_dtype)

    # Get low-frequency coefficients - shape (*wedge_shape, ndim+2)
    low_coeffs = coefficients[0][0][0]

    # Reconstruct complex scalar from first 2 channels via .view()
    low_scalar_2ch = np.ascontiguousarray(low_coeffs[..., :2])
    low_coeff_scalar = low_scalar_2ch.view(complex_dtype).squeeze(-1)

    # Process scalar component
    curvelet_band_scalar = upsample(low_coeff_scalar, decimation_ratio)
    curvelet_freq_scalar = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
        curvelet_band_scalar
    )
    image_frequencies[0].flat[idx] += curvelet_freq_scalar.flat[idx] * window_values

    # Process all Riesz components for low frequency (channels 2 onwards)
    num_riesz = num_channels - 2  # ndim
    for riesz_idx in range(num_riesz):
        low_coeff_riesz = low_coeffs[..., 2 + riesz_idx]  # Access via last dimension
        curvelet_band_riesz = upsample(
            low_coeff_riesz.astype(complex_dtype), decimation_ratio
        )
        curvelet_freq_riesz = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
            curvelet_band_riesz
        )
        # Output component index: scalar=0, riesz_1=1, riesz_2=2, etc.
        image_frequencies[1 + riesz_idx].flat[idx] += (
            curvelet_freq_riesz.flat[idx] * window_values
        )

    # Transform back to spatial domain and take real part
    results = []
    scalar: npt.NDArray[F] = np.fft.ifftn(image_frequencies[0]).real.astype(real_dtype)
    results.append(scalar)

    # Negate Riesz components: forward computes Rₖf, we want -Rₖf
    for comp_idx in range(1, num_components):
        riesz_k: npt.NDArray[F] = -np.fft.ifftn(
            image_frequencies[comp_idx]
        ).real.astype(real_dtype)
        results.append(riesz_k)

    return tuple(results)
