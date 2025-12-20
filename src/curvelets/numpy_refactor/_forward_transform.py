from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .typing import UDCTCoefficients, UDCTWindows
from .utils import ParamUDCT, _fftflip_all_axes, downsamp


def _apply_forward_transform(
    image: np.ndarray,
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
    image : np.ndarray
        Input image or volume to decompose. Must have shape matching
        `parameters.size`. Can be real or complex-valued.
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
    flipping the positive frequency windows using `_fftflip_all_axes`.

    The transform provides a tight frame, meaning perfect reconstruction
    is possible using the corresponding backward transform.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor.utils import ParamUDCT
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
    >>> 4
    >>> len(coeffs[0][0])  # Low-frequency: 1 wedge
    >>> 1
    >>> len(coeffs[1])  # First high-frequency scale: 2 directions (real mode)
    >>> 2
    >>> coeffs[0][0][0].shape  # Downsampled low-frequency coefficients
    >>> (32, 32)
    >>>
    >>> # Apply forward transform (complex mode)
    >>> coeffs_complex = _apply_forward_transform(
    ...     image, params, windows, decimation_ratios, use_complex_transform=True
    ... )
    >>> len(coeffs_complex[1])  # Complex mode: 4 directions (2*dim)
    >>> 4
    >>> np.iscomplexobj(coeffs_complex[0][0][0])  # Complex coefficients
    >>> True
    """
    image_frequency = np.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Low frequency band processing
    frequency_band = np.zeros_like(image_frequency)
    idx, val = windows[0][0][0]
    frequency_band.flat[idx] = image_frequency.flat[idx] * val.astype(complex_dtype)

    if use_complex_transform:
        # Complex transform: keep complex low frequency
        curvelet_band = np.fft.ifftn(frequency_band)
    else:
        # Real transform: take real part
        curvelet_band = np.fft.ifftn(frequency_band)

    coefficients: UDCTCoefficients = [
        [[downsamp(curvelet_band, decimation_ratios[0][0])]]
    ]
    norm = np.sqrt(
        np.prod(np.full((parameters.dim,), fill_value=2 ** (parameters.res - 1)))
    )
    coefficients[0][0][0] *= norm

    if use_complex_transform:
        # Complex transform: separate +/- frequency bands
        # Structure: [scale][direction][wedge]
        # Directions 0..dim-1 are positive frequencies
        # Directions dim..2*dim-1 are negative frequencies
        for scale_idx in range(1, 1 + parameters.res):
            coefficients.append([])
            # Positive frequency bands (directions 0..dim-1)
            for direction_idx in range(parameters.dim):
                coefficients[scale_idx].append([])
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    # Convert sparse window to dense for manipulation
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    subwindow = np.zeros(parameters.size, dtype=val.dtype)
                    subwindow.flat[idx] = val

                    # Apply window to frequency domain
                    band_filtered = np.sqrt(0.5) * np.fft.ifftn(
                        image_frequency * subwindow.astype(complex_dtype)
                    )

                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                    coefficients[scale_idx][direction_idx].append(
                        downsamp(band_filtered, decimation_ratio)
                    )
                    coefficients[scale_idx][direction_idx][wedge_idx] *= np.sqrt(
                        2 * np.prod(decimation_ratio)
                    )

            # Negative frequency bands (directions dim..2*dim-1)
            for direction_idx in range(parameters.dim):
                coefficients[scale_idx].append([])
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    # Convert sparse window to dense for manipulation
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    subwindow = np.zeros(parameters.size, dtype=val.dtype)
                    subwindow.flat[idx] = val

                    # Apply fftflip to get negative frequency window
                    subwindow_flipped = _fftflip_all_axes(subwindow)

                    # Apply flipped window to frequency domain
                    band_filtered = np.sqrt(0.5) * np.fft.ifftn(
                        image_frequency * subwindow_flipped.astype(complex_dtype)
                    )

                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                    coefficients[scale_idx][direction_idx + parameters.dim].append(
                        downsamp(band_filtered, decimation_ratio)
                    )
                    coefficients[scale_idx][direction_idx + parameters.dim][
                        wedge_idx
                    ] *= np.sqrt(2 * np.prod(decimation_ratio))
    else:
        # Real transform: combined +/- frequencies
        for scale_idx in range(1, 1 + parameters.res):
            coefficients.append([])
            for direction_idx in range(parameters.dim):
                coefficients[scale_idx].append([])
                for wedge_idx in range(len(windows[scale_idx][direction_idx])):
                    frequency_band = np.zeros_like(image_frequency)
                    idx, val = windows[scale_idx][direction_idx][wedge_idx]
                    frequency_band.flat[idx] = image_frequency.flat[idx] * val.astype(
                        complex_dtype
                    )

                    curvelet_band = np.fft.ifftn(frequency_band)
                    decimation_ratio = decimation_ratios[scale_idx][direction_idx, :]
                    coefficients[scale_idx][direction_idx].append(
                        downsamp(curvelet_band, decimation_ratio)
                    )
                    coefficients[scale_idx][direction_idx][wedge_idx] *= np.sqrt(
                        2 * np.prod(decimation_ratio)
                    )
    return coefficients
