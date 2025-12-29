from __future__ import annotations

# pylint: disable=duplicate-code
# Duplicate code with torch implementation is expected
import numpy as np
import numpy.typing as npt

from ._typing import F, IntegerNDArray, IntpNDArray, MUDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, upsample


def _process_wedge_backward_monogenic(
    coefficients: list[npt.NDArray[np.complexfloating] | npt.NDArray[F]],
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
    coefficients : list
        List of coefficient arrays: [scalar, riesz_1, riesz_2, ..., riesz_ndim]
        - scalar: Complex array (from forward with transform_kind="monogenic")
        - riesz_k: Real arrays for k = 1, 2, ..., ndim
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
    coeff_scalar = coefficients[0]
    coeff_riesz_list = coefficients[1:]  # All Riesz components

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

    # Process all Riesz components
    contributions = [contribution_scalar]
    for coeff_riesz_k in coeff_riesz_list:
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
    coefficients: MUDCTCoefficients,  # type: ignore[type-arg]
    parameters: ParamUDCT,
    windows: UDCTWindows,
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
    coefficients : MUDCTCoefficients
        Monogenic curvelet coefficients from forward_monogenic().
    parameters : ParamUDCT
        UDCT parameters.
    windows : UDCTWindows
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
    # Determine dtype from coefficients
    scalar_coeff = coefficients[0][0][0][0]  # First scalar component
    real_dtype = np.real(np.empty(0, dtype=scalar_coeff.dtype)).dtype
    complex_dtype = np.result_type(real_dtype, 1j)

    # Determine number of components (ndim+1)
    num_components = len(coefficients[0][0][0])

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

    # Get low-frequency coefficients
    low_coeffs = coefficients[0][0][0]
    low_coeff_scalar = low_coeffs[0]

    # Process scalar component
    curvelet_band_scalar = upsample(low_coeff_scalar, decimation_ratio)
    curvelet_freq_scalar = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
        curvelet_band_scalar
    )
    image_frequencies[0].flat[idx] += curvelet_freq_scalar.flat[idx] * window_values

    # Process all Riesz components for low frequency
    for comp_idx in range(1, num_components):
        low_coeff_riesz = low_coeffs[comp_idx]
        curvelet_band_riesz = upsample(
            low_coeff_riesz.astype(complex_dtype), decimation_ratio
        )
        curvelet_freq_riesz = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
            curvelet_band_riesz
        )
        image_frequencies[comp_idx].flat[idx] += (
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
