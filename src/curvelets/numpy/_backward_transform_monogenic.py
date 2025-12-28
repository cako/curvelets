from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._typing import F, IntegerNDArray, IntpNDArray, MUDCTCoefficients, UDCTWindows
from ._utils import ParamUDCT, upsample


def _process_wedge_backward_monogenic(
    coefficients: tuple[
        npt.NDArray[np.complexfloating], npt.NDArray[F], npt.NDArray[F]
    ],
    window: tuple[IntpNDArray, npt.NDArray[np.floating]],
    decimation_ratio: IntegerNDArray,
    complex_dtype: npt.DTypeLike,
) -> tuple[
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.complexfloating],
]:
    """
    Process a single wedge for monogenic backward transform.

    Uses the discrete tight frame property of UDCT:
    - scalar: c₀ · W → reconstructs f
    - riesz1: c₁ · W → reconstructs R₁f (will be negated in caller for -R₁f)
    - riesz2: c₂ · W → reconstructs R₂f (will be negated in caller for -R₂f)

    This is simpler than the continuous quaternion formula because the UDCT
    windows satisfy partition of unity ∑|W|² = 1, making each component
    independently reconstructable.

    Parameters
    ----------
    coefficients : tuple
        Three coefficient arrays: (scalar, riesz1, riesz2)
        - scalar: Complex array (from forward_monogenic)
        - riesz1, riesz2: Real arrays
    window : tuple[IntpNDArray, npt.NDArray[np.floating]]
        Sparse window representation as (indices, values) tuple.
    decimation_ratio : IntegerNDArray
        Decimation ratio for this wedge.
    complex_dtype : npt.DTypeLike
        Complex dtype for output.

    Returns
    -------
    tuple[npt.NDArray[np.complexfloating], ...]
        Three frequency-domain contributions: (scalar, riesz1, riesz2)
        Each is sparse (only non-zero at window indices).
    """
    coeff_scalar, coeff_riesz1, coeff_riesz2 = coefficients

    # Upsample all coefficients to full size
    curvelet_band_scalar = upsample(coeff_scalar, decimation_ratio)
    curvelet_band_riesz1 = upsample(
        coeff_riesz1.astype(complex_dtype), decimation_ratio
    )
    curvelet_band_riesz2 = upsample(
        coeff_riesz2.astype(complex_dtype), decimation_ratio
    )

    # Undo normalization: divide by sqrt(2 * prod(decimation_ratio))
    # This matches the standard backward transform normalization
    norm_factor = np.sqrt(2 * np.prod(decimation_ratio))
    curvelet_band_scalar /= norm_factor
    curvelet_band_riesz1 /= norm_factor
    curvelet_band_riesz2 /= norm_factor

    # Transform to frequency domain
    curvelet_freq_scalar = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band_scalar)
    curvelet_freq_riesz1 = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band_riesz1)
    curvelet_freq_riesz2 = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band_riesz2)

    # Get window indices and values
    idx, val = window
    window_values = val.astype(complex_dtype)

    # Initialize contribution arrays
    contribution_scalar = np.zeros(curvelet_freq_scalar.shape, dtype=complex_dtype)
    contribution_riesz1 = np.zeros(curvelet_freq_scalar.shape, dtype=complex_dtype)
    contribution_riesz2 = np.zeros(curvelet_freq_scalar.shape, dtype=complex_dtype)

    # Apply simple tight frame formula: each component uses the same window W
    # scalar: c₀ · W → f
    # riesz1: c₁ · W → R₁f
    # riesz2: c₂ · W → R₂f
    contribution_scalar.flat[idx] = curvelet_freq_scalar.flat[idx] * window_values
    contribution_riesz1.flat[idx] = curvelet_freq_riesz1.flat[idx] * window_values
    contribution_riesz2.flat[idx] = curvelet_freq_riesz2.flat[idx] * window_values

    return (contribution_scalar, contribution_riesz1, contribution_riesz2)


def _apply_backward_transform_monogenic(
    coefficients: MUDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[IntegerNDArray],
) -> tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]:
    """
    Apply backward monogenic curvelet transform.

    This uses the discrete tight frame property of UDCT rather than the
    continuous quaternion formula from Storath 2010. The result satisfies:
    backward_monogenic(forward_monogenic(f)) ≈ monogenic(f)

    Where monogenic(f) = (f, -R₁f, -R₂f)

    The reconstruction uses the partition of unity property:
    - scalar: ∑ c₀ · W = f
    - riesz1: -∑ c₁ · W = -R₁f
    - riesz2: -∑ c₂ · W = -R₂f

    .. note::
        **2D Limitation**: The monogenic curvelet transform is mathematically
        defined only for 2D signals according to Storath 2010. While this
        implementation accepts arbitrary dimensions, only the first two Riesz
        components (R_1 and R_2) are used in the reconstruction, which is
        correct only for 2D inputs.

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
    tuple[npt.NDArray[F], npt.NDArray[F], npt.NDArray[F]]
        Three reconstructed components: (scalar, riesz1, riesz2)
        - scalar: Original input f
        - riesz1: -R_1 f
        - riesz2: -R_2 f
    """
    # Determine dtype from coefficients
    scalar_coeff = coefficients[0][0][0][0]  # First scalar component
    real_dtype = np.real(np.empty(0, dtype=scalar_coeff.dtype)).dtype
    complex_dtype = np.result_type(real_dtype, 1j)

    # Initialize frequency domain arrays for all three components
    image_frequency_scalar = np.zeros(parameters.shape, dtype=complex_dtype)
    image_frequency_riesz1 = np.zeros(parameters.shape, dtype=complex_dtype)
    image_frequency_riesz2 = np.zeros(parameters.shape, dtype=complex_dtype)

    # Process high-frequency bands
    highest_scale_idx = parameters.num_scales - 1
    is_wavelet_mode_highest_scale = len(windows[highest_scale_idx]) == 1

    if is_wavelet_mode_highest_scale:
        # Separate handling for wavelet mode at highest scale
        image_frequency_scalar_wavelet = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_riesz1_wavelet = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_riesz2_wavelet = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_scalar_other = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_riesz1_other = np.zeros(parameters.shape, dtype=complex_dtype)
        image_frequency_riesz2_other = np.zeros(parameters.shape, dtype=complex_dtype)

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
                    contrib_scalar, contrib_riesz1, contrib_riesz2 = (
                        _process_wedge_backward_monogenic(
                            coeffs,
                            window,
                            decimation_ratio,
                            complex_dtype,
                        )
                    )

                    idx, _ = window
                    if scale_idx == highest_scale_idx:
                        image_frequency_scalar_wavelet.flat[idx] += contrib_scalar.flat[
                            idx
                        ]
                        image_frequency_riesz1_wavelet.flat[idx] += contrib_riesz1.flat[
                            idx
                        ]
                        image_frequency_riesz2_wavelet.flat[idx] += contrib_riesz2.flat[
                            idx
                        ]
                    else:
                        image_frequency_scalar_other.flat[idx] += contrib_scalar.flat[
                            idx
                        ]
                        image_frequency_riesz1_other.flat[idx] += contrib_riesz1.flat[
                            idx
                        ]
                        image_frequency_riesz2_other.flat[idx] += contrib_riesz2.flat[
                            idx
                        ]

        # Combine with factor of 2 for real transform mode
        image_frequency_scalar = (
            2 * image_frequency_scalar_other + image_frequency_scalar_wavelet
        )
        image_frequency_riesz1 = (
            2 * image_frequency_riesz1_other + image_frequency_riesz1_wavelet
        )
        image_frequency_riesz2 = (
            2 * image_frequency_riesz2_other + image_frequency_riesz2_wavelet
        )
    else:
        # Normal curvelet mode
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
                    contrib_scalar, contrib_riesz1, contrib_riesz2 = (
                        _process_wedge_backward_monogenic(
                            coeffs,
                            window,
                            decimation_ratio,
                            complex_dtype,
                        )
                    )

                    idx, _ = window
                    image_frequency_scalar.flat[idx] += contrib_scalar.flat[idx]
                    image_frequency_riesz1.flat[idx] += contrib_riesz1.flat[idx]
                    image_frequency_riesz2.flat[idx] += contrib_riesz2.flat[idx]

        # Multiply by 2 for real transform mode
        image_frequency_scalar *= 2
        image_frequency_riesz1 *= 2
        image_frequency_riesz2 *= 2

    # Process low-frequency band
    # Use same structure as standard backward transform for consistency
    decimation_ratio = decimation_ratios[0][0]
    idx, val = windows[0][0][0]
    window_values = val.astype(complex_dtype)

    # Get low-frequency coefficients
    low_coeff_scalar = coefficients[0][0][0][0]
    low_coeff_riesz1 = coefficients[0][0][0][1]
    low_coeff_riesz2 = coefficients[0][0][0][2]

    # Upsample coefficients to full size
    curvelet_band_scalar = upsample(low_coeff_scalar, decimation_ratio)
    curvelet_band_riesz1 = upsample(
        low_coeff_riesz1.astype(complex_dtype), decimation_ratio
    )
    curvelet_band_riesz2 = upsample(
        low_coeff_riesz2.astype(complex_dtype), decimation_ratio
    )

    # Transform to frequency domain (same normalization as standard backward)
    curvelet_freq_scalar = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
        curvelet_band_scalar
    )
    curvelet_freq_riesz1 = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
        curvelet_band_riesz1
    )
    curvelet_freq_riesz2 = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(
        curvelet_band_riesz2
    )

    # Apply tight frame formula: each component uses the same window W
    image_frequency_scalar.flat[idx] += curvelet_freq_scalar.flat[idx] * window_values
    image_frequency_riesz1.flat[idx] += curvelet_freq_riesz1.flat[idx] * window_values
    image_frequency_riesz2.flat[idx] += curvelet_freq_riesz2.flat[idx] * window_values

    # Transform back to spatial domain and take real part
    scalar: npt.NDArray[F] = np.fft.ifftn(image_frequency_scalar).real.astype(
        real_dtype
    )  # type: ignore[assignment]
    # Negate Riesz components: forward computes R₁f and R₂f, we want -R₁f and -R₂f
    riesz1: npt.NDArray[F] = -np.fft.ifftn(image_frequency_riesz1).real.astype(
        real_dtype
    )  # type: ignore[assignment]
    riesz2: npt.NDArray[F] = -np.fft.ifftn(image_frequency_riesz2).real.astype(
        real_dtype
    )  # type: ignore[assignment]

    return (scalar, riesz1, riesz2)
