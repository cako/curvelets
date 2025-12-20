from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .typing import UDCTCoefficients, UDCTWindows
from .utils import ParamUDCT, _fftflip_all_axes, upsamp


def _apply_backward_transform(
    coefficients: UDCTCoefficients,
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
    coefficients : UDCTCoefficients
        Curvelet coefficients from forward transform. Structure:
        coefficients[scale][direction][wedge] = np.ndarray
        - scale 0: Low-frequency band (1 direction, 1 wedge)
        - scale 1..res: High-frequency bands
          * Real mode: dim directions per scale
          * Complex mode: 2*dim directions per scale
    parameters : ParamUDCT
        UDCT parameters containing transform configuration:
        - res : int
            Number of resolution scales
        - dim : int
            Dimensionality of the transform
        - size : tuple[int, ...]
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
        Reconstructed image or volume with shape `parameters.size`.
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
    `_fftflip_all_axes`.

    The transform provides perfect reconstruction when used with the
    corresponding forward transform, due to the tight frame property
    of the curvelet windows.

    The normalization factors ensure energy preservation: coefficients
    are scaled by sqrt(2 * prod(decimation_ratio)) in forward transform,
    and divided by the same factor in backward transform.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor.utils import ParamUDCT
    >>> from curvelets.numpy_refactor._udct_windows import _udct_windows
    >>> from curvelets.numpy_refactor._forward_transform import _apply_forward_transform
    >>> from curvelets.numpy_refactor._backward_transform import _apply_backward_transform
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
    >>> # Forward transform (real mode)
    >>> coeffs = _apply_forward_transform(
    ...     image, params, windows, decimation_ratios, use_complex_transform=False
    ... )
    >>>
    >>> # Backward transform (real mode)
    >>> recon = _apply_backward_transform(
    ...     coeffs, params, windows, decimation_ratios, use_complex_transform=False
    ... )
    >>>
    >>> # Check reconstruction accuracy
    >>> np.allclose(image, recon, atol=1e-10)
    >>> True
    >>> recon.shape
    >>> (64, 64)
    >>> np.isrealobj(recon)  # Real output in real mode
    >>> True
    >>>
    >>> # Forward transform (complex mode)
    >>> coeffs_complex = _apply_forward_transform(
    ...     image, params, windows, decimation_ratios, use_complex_transform=True
    ... )
    >>>
    >>> # Backward transform (complex mode)
    >>> recon_complex = _apply_backward_transform(
    ...     coeffs_complex, params, windows, decimation_ratios, use_complex_transform=True
    ... )
    >>>
    >>> # Check reconstruction accuracy
    >>> np.allclose(image, recon_complex.real, atol=1e-10)
    >>> True
    >>> np.iscomplexobj(recon_complex)  # Complex output in complex mode
    >>> True
    """
    real_dtype = coefficients[0][0][0].real.dtype
    complex_dtype = (
        np.ones(1, dtype=real_dtype) + 1j * np.ones(1, dtype=real_dtype)
    ).dtype
    image_frequency = np.zeros(parameters.size, dtype=complex_dtype)

    if use_complex_transform:
        # Complex transform: reconstruct from separate +/- frequency bands
        for scale_idx in range(1, 1 + parameters.res):
            # Process positive frequency bands (directions 0..dim-1)
            for direction_idx in range(parameters.dim):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    # Convert sparse window to dense
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    subwindow = np.zeros(parameters.size, dtype=val.dtype)
                    subwindow.flat[idx] = val

                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                    curvelet_band = upsamp(
                        coefficients[scale_idx][direction_idx][wedge_idx],
                        decimation_ratio,
                    )
                    curvelet_band /= np.sqrt(2 * np.prod(decimation_ratio))
                    curvelet_band = np.prod(decimation_ratio) * np.fft.fftn(
                        curvelet_band
                    )

                    # Apply window
                    image_frequency += (
                        np.sqrt(0.5) * curvelet_band * subwindow.astype(complex_dtype)
                    )

            # Process negative frequency bands (directions dim..2*dim-1)
            for direction_idx in range(parameters.dim):
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    # Convert sparse window to dense
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    subwindow = np.zeros(parameters.size, dtype=val.dtype)
                    subwindow.flat[idx] = val

                    # Apply fftflip to get negative frequency window
                    subwindow_flipped = _fftflip_all_axes(subwindow)

                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                    curvelet_band = upsamp(
                        coefficients[scale_idx][direction_idx + parameters.dim][
                            wedge_idx
                        ],
                        decimation_ratio,
                    )
                    curvelet_band /= np.sqrt(2 * np.prod(decimation_ratio))
                    curvelet_band = np.prod(decimation_ratio) * np.fft.fftn(
                        curvelet_band
                    )

                    # Apply flipped window
                    image_frequency += (
                        np.sqrt(0.5)
                        * curvelet_band
                        * subwindow_flipped.astype(complex_dtype)
                    )

        # Low frequency band
        image_frequency_low = np.zeros(parameters.size, dtype=complex_dtype)
        decimation_ratio = decimation_ratios[0][0]
        curvelet_band = upsamp(coefficients[0][0][0], decimation_ratio)
        curvelet_band = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(curvelet_band)
        idx, val = windows[0][0][0]
        image_frequency_low.flat[idx] += curvelet_band.flat[idx] * val.astype(
            complex_dtype
        )

        # Combine: low frequency + high frequency contributions
        image_frequency = 2 * image_frequency + image_frequency_low
        # Complex transform: preserve complex output for complex inputs
        return np.fft.ifftn(image_frequency)

    # Real transform: combined +/- frequencies
    for scale_idx in range(1, 1 + parameters.res):
        for direction_idx in range(parameters.dim):
            for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                curvelet_band = upsamp(
                    coefficients[scale_idx][direction_idx][wedge_idx], decimation_ratio
                )
                curvelet_band /= np.sqrt(2 * np.prod(decimation_ratio))
                curvelet_band = np.prod(decimation_ratio) * np.fft.fftn(curvelet_band)
                idx, val = windows[scale_idx][direction_idx][wedge_idx]
                image_frequency.flat[idx] += curvelet_band.flat[idx] * val.astype(
                    complex_dtype
                )

    image_frequency_low = np.zeros(parameters.size, dtype=complex_dtype)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsamp(coefficients[0][0][0], decimation_ratio)
    curvelet_band = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(curvelet_band)
    idx, val = windows[0][0][0]
    image_frequency_low.flat[idx] += curvelet_band.flat[idx] * val.astype(complex_dtype)
    image_frequency = 2 * image_frequency + image_frequency_low
    return np.fft.ifftn(image_frequency).real
