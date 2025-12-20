from __future__ import annotations

import logging
from math import prod
from typing import Literal

import numpy as np
import numpy.typing as npt

from .meyerwavelet import meyerfwdmd, meyerinvmd
from .typing import UDCTCoefficients, UDCTWindows
from .udctmdwin import udctmdwin
from .utils import ParamUDCT, circshift, downsamp, from_sparse_new, upsamp


def _fftflip_all_axes(F: np.ndarray) -> np.ndarray:
    """
    Apply fftflip to all axes of an array.

    This produces X(-omega) from X(omega) in FFT representation.
    After flipping, the array is circshifted by 1 in each dimension
    to maintain proper frequency alignment.

    Parameters
    ----------
    F : np.ndarray
        Input array in FFT representation.

    Returns
    -------
    np.ndarray
        Flipped array representing negative frequencies.
    """
    Fc = F.copy()
    for axis in range(F.ndim):
        Fc = np.flip(Fc, axis)
    shiftvec = tuple(1 for _ in range(F.ndim))
    return circshift(Fc, shiftvec)


def udctmddec(
    image: np.ndarray,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
) -> UDCTCoefficients:
    """
    Apply UDCT decomposition (forward transform).

    Parameters
    ----------
    image : np.ndarray
        Input image/volume.
    parameters : ParamUDCT
        UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale/direction.
    use_complex_transform : bool, optional
        If True, use complex transform (separate +/- frequency bands).
        If False, use real transform (combined +/- frequencies). Default is False.

    Returns
    -------
    UDCTCoefficients
        Curvelet coefficients. When use_complex_transform=True, directions are doubled
        (first dim directions for positive frequencies, next dim for negative).

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor import udctmddec, ParamUDCT
    >>> # This function is typically called internally by UDCT.forward()
    >>> pass
    """
    image_frequency = np.fft.fftn(image)
    complex_dtype = image_frequency.dtype

    # Low frequency band processing
    frequency_band = np.zeros_like(image_frequency)
    idx, val = from_sparse_new(windows[0][0][0])
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
                    idx, val = from_sparse_new(
                        windows[scale_idx][direction_idx][wedge_idx]
                    )
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
                    idx, val = from_sparse_new(
                        windows[scale_idx][direction_idx][wedge_idx]
                    )
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
                    idx, val = from_sparse_new(
                        windows[scale_idx][direction_idx][wedge_idx]
                    )
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


def udctmdrec(
    coefficients: UDCTCoefficients,
    parameters: ParamUDCT,
    windows: UDCTWindows,
    decimation_ratios: list[npt.NDArray[np.int_]],
    use_complex_transform: bool = False,
) -> np.ndarray:
    """
    Apply UDCT reconstruction (backward transform).

    Parameters
    ----------
    coefficients : UDCTCoefficients
        Curvelet coefficients.
    parameters : ParamUDCT
        UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list[npt.NDArray[np.int_]]
        Decimation ratios for each scale/direction.
    use_complex_transform : bool, optional
        If True, use complex transform (separate +/- frequency bands) and
        return complex output. This is required for complex-valued inputs.
        If False, use real transform (combined +/- frequencies) and return
        real output. Default is False.

    Returns
    -------
    np.ndarray
        Reconstructed image/volume. Returns complex array when use_complex_transform=True,
        real array when use_complex_transform=False.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor import udctmdrec, ParamUDCT
    >>> # This function is typically called internally by UDCT.backward()
    >>> pass
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
                    idx, val = from_sparse_new(
                        windows[scale_idx][direction_idx][wedge_idx]
                    )
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
                    idx, val = from_sparse_new(
                        windows[scale_idx][direction_idx][wedge_idx]
                    )
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
        idx, val = from_sparse_new(windows[0][0][0])
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
                idx, val = from_sparse_new(windows[scale_idx][direction_idx][wedge_idx])
                image_frequency.flat[idx] += curvelet_band.flat[idx] * val.astype(
                    complex_dtype
                )

    image_frequency_low = np.zeros(parameters.size, dtype=complex_dtype)
    decimation_ratio = decimation_ratios[0][0]
    curvelet_band = upsamp(coefficients[0][0][0], decimation_ratio)
    curvelet_band = np.sqrt(np.prod(decimation_ratio)) * np.fft.fftn(curvelet_band)
    idx, val = from_sparse_new(windows[0][0][0])
    image_frequency_low.flat[idx] += curvelet_band.flat[idx] * val.astype(complex_dtype)
    image_frequency = 2 * image_frequency + image_frequency_low
    return np.fft.ifftn(image_frequency).real


class UDCT:
    """
    Uniform Discrete Curvelet Transform (UDCT) implementation.

    This class provides forward and backward curvelet transforms with support
    for both real and complex transforms, as well as optional Meyer wavelet
    decomposition at the highest scale.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    angular_wedges_config : np.ndarray, optional
        Configuration array specifying the number of angular wedges per scale
        and dimension. Shape is (num_scales, dimension). If provided, cannot
        be used together with num_scales/wedges_per_direction. Default is None.
    num_scales : int, optional
        Number of scales. Must be > 1. Used when angular_wedges_config is not
        provided. Default is 3.
    wedges_per_direction : int, optional
        Number of angular wedges per direction at the coarsest scale.
        The number of wedges doubles at each finer scale. Must be >= 3.
        Used when angular_wedges_config is not provided. Default is 3.
    window_overlap : float, optional
        Window overlap parameter controlling the smoothness of window transitions.
        If None and using num_scales/wedges_per_direction, automatically chosen
        based on wedges_per_direction. Default is None (auto) or 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency parameters defining the frequency bands.
        Default is (pi/3, 2*pi/3, 2*pi/3, 4*pi/3).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-5.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" applies Meyer wavelet decomposition at the highest scale.
        Default is "curvelet".
    use_complex_transform : bool, optional
        If True, use complex transform which separates positive and negative
        frequency components into different bands. Each band is scaled by
        sqrt(0.5). If False (default), use real transform where each band
        captures both +/- frequencies combined.

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the input data.
    high_frequency_mode : str
        High frequency mode.
    use_complex_transform : bool
        Whether complex transform is enabled.
    parameters : ParamUDCT
        Internal UDCT parameters.
    windows : UDCTWindows
        Curvelet windows in sparse format.
    decimation_ratios : list
        Decimation ratios for each scale/direction.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy_refactor import UDCT
    >>> # Create a 2D transform using num_scales (simplified interface)
    >>> transform = UDCT(shape=(64, 64), num_scales=3, wedges_per_direction=3)
    >>> data = np.random.randn(64, 64)
    >>> coeffs = transform.forward(data)
    >>> recon = transform.backward(coeffs)
    >>> np.allclose(data, recon, atol=1e-4)
    True
    >>> # Create using angular_wedges_config (advanced interface)
    >>> cfg = np.array([[3, 3], [6, 6]])
    >>> transform2 = UDCT(shape=(64, 64), angular_wedges_config=cfg)
    >>> coeffs2 = transform2.forward(data)
    >>> recon2 = transform2.backward(coeffs2)
    >>> np.allclose(data, recon2, atol=1e-4)
    True
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: np.ndarray | None = None,
        num_scales: int | None = None,
        wedges_per_direction: int | None = None,
        window_overlap: float | None = None,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-5,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        use_complex_transform: bool = False,
    ) -> None:
        # Store basic attributes
        self.shape = shape
        self.high_frequency_mode = high_frequency_mode
        self.use_complex_transform = use_complex_transform

        # Calculate necessary parameters
        params_dict = self._calculate_necessary_parameters(
            shape=shape,
            angular_wedges_config=angular_wedges_config,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
        )

        # Create ParamUDCT object
        self.parameters = ParamUDCT(
            dim=params_dict["dimension"],
            size=params_dict["internal_shape"],
            angular_wedges_config=params_dict["angular_wedges_config"],
            window_overlap=params_dict["window_overlap"],
            radial_frequency_params=params_dict["radial_frequency_params"],
            window_threshold=params_dict["window_threshold"],
        )

        # Calculate windows
        self.windows, self.decimation_ratios, self.indices = self._calculate_windows()

        # Initialize state
        self._wavelet_bands: list[np.ndarray] = []

    def _calculate_necessary_parameters(
        self,
        shape: tuple[int, ...],
        angular_wedges_config: np.ndarray | None,
        num_scales: int | None,
        wedges_per_direction: int | None,
        window_overlap: float | None,
        radial_frequency_params: tuple[float, float, float, float] | None,
        window_threshold: float,
        high_frequency_mode: Literal["curvelet", "wavelet"],
    ) -> dict:
        """
        Calculate all necessary parameters for UDCT initialization.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the input data.
        angular_wedges_config : np.ndarray | None
            Configuration array, or None to use num_scales/wedges_per_direction.
        num_scales : int | None
            Number of scales (used when angular_wedges_config is None).
        wedges_per_direction : int | None
            Wedges per direction (used when angular_wedges_config is None).
        window_overlap : float | None
            Window overlap parameter.
        radial_frequency_params : tuple[float, float, float, float] | None
            Radial frequency parameters.
        window_threshold : float
            Window threshold.
        high_frequency_mode : str
            High frequency mode.

        Returns
        -------
        dict
            Dictionary containing all calculated parameters.
        """
        dimension = len(shape)

        # Determine which initialization style to use
        if angular_wedges_config is not None:
            if num_scales is not None or wedges_per_direction is not None:
                msg = "Cannot specify both angular_wedges_config and num_scales/wedges_per_direction"
                raise ValueError(msg)
            # Use provided angular_wedges_config directly
            computed_angular_wedges_config = angular_wedges_config
            num_scales_computed = len(angular_wedges_config)
            # Use provided window_overlap or default
            computed_window_overlap = (
                window_overlap if window_overlap is not None else 0.15
            )
        else:
            # Use num_scales/wedges_per_direction (SimpleUDCT style)
            if num_scales is None:
                num_scales = 3
            if wedges_per_direction is None:
                wedges_per_direction = 3

            if num_scales <= 1:
                msg = "num_scales must be > 1"
                raise ValueError(msg)
            if wedges_per_direction < 3:
                msg = "wedges_per_direction must be >= 3"
                raise ValueError(msg)

            # Convert to angular_wedges_config
            wedges_per_scale: npt.NDArray[np.int_] = (
                wedges_per_direction * 2 ** np.arange(num_scales - 1)
            ).astype(int)
            computed_angular_wedges_config = np.tile(
                wedges_per_scale[:, None], dimension
            )
            num_scales_computed = num_scales

            # Auto-select window_overlap if not provided
            if window_overlap is None:
                if wedges_per_direction == 3:
                    computed_window_overlap = 0.15
                elif wedges_per_direction == 4:
                    computed_window_overlap = 0.3
                elif wedges_per_direction == 5:
                    computed_window_overlap = 0.5
                else:
                    computed_window_overlap = 0.5
            else:
                computed_window_overlap = window_overlap

            # Validate window_overlap
            for scale_idx, num_wedges in enumerate(wedges_per_scale, start=1):
                const = (
                    2 ** (scale_idx / num_wedges)
                    * (1 + 2 * computed_window_overlap)
                    * (1 + computed_window_overlap)
                )
                if const >= num_wedges:
                    msg = (
                        f"window_overlap={computed_window_overlap:.3f} does not respect the relationship "
                        f"(2^{scale_idx}/{num_wedges})(1+2α)(1+α) = {const:.3f} < 1 for scale {scale_idx + 1}"
                    )
                    logging.warning(msg)

        # Validate wavelet mode requirements
        # For wavelet mode, we need at least 2 scales total, which means
        # at least 2 rows in angular_wedges_config
        if high_frequency_mode == "wavelet" and len(computed_angular_wedges_config) < 2:
            msg = "Wavelet mode requires at least 2 scales (num_scales >= 2)"
            raise ValueError(msg)

        # Calculate internal shape (wavelet mode halves the size)
        if high_frequency_mode == "wavelet":
            internal_shape = tuple(s // 2 for s in shape)
        else:
            internal_shape = shape

        # Set default radial_frequency_params if not provided
        if radial_frequency_params is None:
            computed_radial_frequency_params: tuple[float, float, float, float] = tuple(
                np.array([1.0, 2.0, 2.0, 4.0]) * np.pi / 3
            )
        else:
            computed_radial_frequency_params = radial_frequency_params

        return {
            "dimension": dimension,
            "internal_shape": internal_shape,
            "angular_wedges_config": computed_angular_wedges_config,
            "window_overlap": computed_window_overlap,
            "radial_frequency_params": computed_radial_frequency_params,
            "window_threshold": window_threshold,
        }

    def _calculate_windows(
        self,
    ) -> tuple[
        UDCTWindows, list[npt.NDArray[np.int_]], dict[int, dict[int, np.ndarray]]
    ]:
        """
        Calculate curvelet windows, decimation ratios, and indices.

        Returns
        -------
        tuple
            (windows, decimation_ratios, indices)
        """
        return udctmdwin(self.parameters)

    def from_sparse(
        self, arr_sparse: tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """
        Convert sparse array representation to dense array.

        Parameters
        ----------
        arr_sparse : tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]
            Sparse array as (indices, values) tuple.

        Returns
        -------
        np.ndarray
            Dense array representation.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> sparse = (np.array([0, 100]), np.array([1.0, 2.0]))
        >>> dense = transform.from_sparse(sparse)
        >>> dense.shape
        (64, 64)
        """
        idx, val = from_sparse_new(arr_sparse)
        arr_full = np.zeros(self.parameters.size, dtype=val.dtype)
        arr_full.flat[idx] += val
        return arr_full

    def vect(self, coefficients: UDCTCoefficients) -> npt.NDArray[np.complexfloating]:
        """
        Convert structured coefficients to vector representation.

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Structured curvelet coefficients.

        Returns
        -------
        np.ndarray
            Flattened vector of all coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> vec.shape
        (4096,)
        """
        coefficients_vec = []
        for scale_coeffs in coefficients:
            for direction_coeffs in scale_coeffs:
                for wedge_coeffs in direction_coeffs:
                    coefficients_vec.append(wedge_coeffs.ravel())
        return np.concatenate(coefficients_vec)

    def struct(
        self, coefficients_vec: npt.NDArray[np.complexfloating]
    ) -> UDCTCoefficients:
        """
        Convert vector representation to structured coefficients.

        Parameters
        ----------
        coefficients_vec : np.ndarray
            Flattened vector of coefficients.

        Returns
        -------
        UDCTCoefficients
            Structured curvelet coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> vec = transform.vect(coeffs)
        >>> coeffs_recon = transform.struct(vec)
        >>> len(coeffs_recon) == len(coeffs)
        True
        """
        begin_idx = 0
        coefficients: UDCTCoefficients = []
        internal_shape = np.array(self.parameters.size)
        for scale_idx, decimation_ratios_scale in enumerate(self.decimation_ratios):
            coefficients.append([])
            for direction_idx, decimation_ratio_dir in enumerate(
                decimation_ratios_scale
            ):
                coefficients[scale_idx].append([])
                for _ in self.windows[scale_idx][direction_idx]:
                    shape_decimated = internal_shape // decimation_ratio_dir
                    end_idx = begin_idx + prod(shape_decimated)
                    wedge = coefficients_vec[begin_idx:end_idx].reshape(shape_decimated)
                    coefficients[scale_idx][direction_idx].append(wedge)
                    begin_idx = end_idx
        return coefficients

    def forward(self, image: np.ndarray) -> UDCTCoefficients:
        """
        Apply forward curvelet transform.

        Parameters
        ----------
        image : np.ndarray
            Input data with shape matching self.shape.

        Returns
        -------
        UDCTCoefficients
            Curvelet coefficients as nested list structure.
            When use_complex_transform=True, directions are doubled (first dim directions
            for positive frequencies, next dim for negative).

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> len(coeffs)  # Number of scales
        4
        """
        np.testing.assert_equal(self.shape, image.shape)

        if self.high_frequency_mode == "wavelet":
            # Apply Meyer wavelet decomposition
            # meyerfwdmd returns 2^dim bands: first is lowpass, rest are highpass
            bands = meyerfwdmd(image)
            lowpass = bands[0]
            self._wavelet_bands = bands[1:]  # Store highpass bands for backward

            # Apply curvelet transform to lowpass only
            return udctmddec(
                lowpass,
                self.parameters,
                self.windows,
                self.decimation_ratios,
                use_complex_transform=self.use_complex_transform,
            )

        return udctmddec(
            image,
            self.parameters,
            self.windows,
            self.decimation_ratios,
            use_complex_transform=self.use_complex_transform,
        )

    def backward(self, coefficients: UDCTCoefficients) -> np.ndarray:
        """
        Apply backward curvelet transform (reconstruction).

        Parameters
        ----------
        coefficients : UDCTCoefficients
            Curvelet coefficients from forward transform.

        Returns
        -------
        np.ndarray
            Reconstructed data with shape matching self.shape.
            Returns complex array when use_complex_transform=True (required for complex inputs),
            real array when use_complex_transform=False.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy_refactor import UDCT
        >>> transform = UDCT(shape=(64, 64))
        >>> data = np.random.randn(64, 64)
        >>> coeffs = transform.forward(data)
        >>> recon = transform.backward(coeffs)
        >>> np.allclose(data, recon, atol=1e-4)
        True
        """
        if self.high_frequency_mode == "wavelet":
            # Reconstruct lowpass from curvelet coefficients
            lowpass_recon = udctmdrec(
                coefficients,
                self.parameters,
                self.windows,
                self.decimation_ratios,
                use_complex_transform=self.use_complex_transform,
            )

            # Combine with wavelet highpass bands and apply Meyer inverse
            all_bands = [lowpass_recon, *self._wavelet_bands]
            return meyerinvmd(all_bands)

        return udctmdrec(
            coefficients,
            self.parameters,
            self.windows,
            self.decimation_ratios,
            use_complex_transform=self.use_complex_transform,
        )
