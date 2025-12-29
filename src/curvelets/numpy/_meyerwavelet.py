from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._typing import C, F
from ._utils import circular_shift, meyer_window


class MeyerWavelet:
    """
    Multi-dimensional Meyer wavelet transform with pre-computed filters.

    This class provides forward and backward Meyer wavelet transforms with
    filters pre-computed during initialization for improved performance. The
    forward transform returns all subbands in a nested list structure, and the
    backward transform accepts this full structure for reconstruction.

    The filter computation uses the same method as UDCT's `_create_bandpass_windows()`
    for num_scales=2, ensuring compatibility when used with UDCT in wavelet mode
    with num_scales=2 and high_frequency_mode="meyer".

    Parameters
    ----------
    shape : tuple[int, ...]
        Expected shape of input signals. Used for validation and to determine
        the number of dimensions. All dimensions must be even (divisible by 2).

    Attributes
    ----------
    shape : tuple[int, ...]
        Expected signal shape (all dimensions are even).
    dimension : int
        Number of dimensions.

    Raises
    ------
    ValueError
        If any dimension in shape is odd (not divisible by 2).

    Notes
    -----
    This implementation requires all dimensions to be even. Odd-length signals
    are not supported due to the downsampling strategy used in the transform,
    which produces mismatched subband sizes that cannot be correctly
    reconstructed.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelets.numpy import MeyerWavelet
    >>> wavelet = MeyerWavelet(shape=(64, 64))
    >>> signal = np.random.randn(64, 64)
    >>> coefficients = wavelet.forward(signal)
    >>> len(coefficients)  # 2 subband groups
    2
    >>> coefficients[0][0].shape  # Lowpass subband
    (32, 32)
    >>> len(coefficients[1])  # Highpass subbands
    3
    >>> reconstructed = wavelet.backward(coefficients)
    >>> np.allclose(signal, reconstructed, atol=1e-10)
    True
    >>> # Odd dimensions raise an error
    >>> try:
    ...     MeyerWavelet(shape=(65, 65))
    ... except ValueError:
    ...     print("Odd dimensions not supported")
    Odd dimensions not supported
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        """
        Initialize Meyer wavelet transform.

        All required filters are pre-computed during initialization based on
        the input shape. This ensures optimal performance during forward and
        backward transforms.

        Parameters
        ----------
        shape : tuple[int, ...]
            Expected shape of input signals. Used for validation and to
            determine which filters to pre-compute. All dimensions must be
            even (divisible by 2).

        Raises
        ------
        ValueError
            If any dimension in shape is odd (not divisible by 2).

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> wavelet.shape
        (64, 64)
        >>> wavelet.dimension
        2
        >>> len(wavelet._filters)
        1
        >>> # Odd dimensions raise an error
        >>> try:
        ...     MeyerWavelet(shape=(65, 65))
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: All dimensions must be even, got shape (65, 65) with odd dimensions at indices [0, 1]
        """
        # Validate that all dimensions are even
        odd_dimensions = [i for i, dim in enumerate(shape) if dim % 2 != 0]
        if odd_dimensions:
            error_msg = (
                f"All dimensions must be even, got shape {shape} "
                f"with odd dimensions at indices {odd_dimensions}"
            )
            raise ValueError(error_msg)

        self.shape = shape
        self.dimension = len(shape)
        self._filters: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}

        # Pre-compute all required filters
        self._initialize_filters()

    def _initialize_filters(self) -> None:
        """
        Pre-compute all filters needed for the transform.

        Determines unique filter sizes from the input shape and pre-computes
        all required filters. Filters are stored in `_filters` for
        direct access during forward and backward transforms.
        """
        # Determine unique filter sizes (one per unique dimension size)
        required_filter_sizes = set(self.shape)

        # Pre-compute all filters
        for signal_length in required_filter_sizes:
            self._filters[signal_length] = self._compute_single_filter(signal_length)

    def _compute_single_filter(
        self, signal_length: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Compute a single Meyer wavelet filter pair for given signal length.

        Uses the same computation method as UDCT's `_create_bandpass_windows()` for
        num_scales=2 compatibility. The lowpass filter is computed using:
        - Frequency grid: :math:`\\text{np.linspace}(-1.5\\pi, 0.5\\pi, \\text{signal\\_length}, \\text{endpoint=False})`
        - Meyer window parameters: :math:`[-2, -1, \\pi/3, 2\\pi/3]`
        - Special case for num_scales=2: adds :math:`\\text{meyer\\_window}(|\\text{freq} + 2\\pi|, ...)`
        - Circular shift by :math:`\\text{signal\\_length} // 4` before square root

        The highpass filter is computed as the complement of the lowpass to ensure
        perfect reconstruction: `|lowpass|² + |highpass|² = 1`.

        Parameters
        ----------
        signal_length : int
            Length of the signal along the transform dimension.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            Lowpass and highpass filters as 1D arrays of length signal_length.
            The filters satisfy perfect reconstruction: `|lowpass|² + |highpass|² = 1`.
        """
        # Compute frequency grid (matching UDCT)
        frequency_grid = np.linspace(
            -1.5 * np.pi, 0.5 * np.pi, signal_length, endpoint=False
        )

        # Meyer window parameters (matching UDCT for num_scales=2)
        meyer_params = np.array([-2, -1, np.pi / 3, 2 * np.pi / 3])
        abs_frequency_grid = np.abs(frequency_grid)

        # Compute Meyer window function
        window_values = meyer_window(abs_frequency_grid, *meyer_params)

        # Add num_scales=2 special case (matching UDCT)
        window_values += meyer_window(np.abs(frequency_grid + 2 * np.pi), *meyer_params)

        # Lowpass filter: circular-shifted and square-rooted (matching UDCT)
        lowpass_window_sq_shifted = circular_shift(window_values, (signal_length // 4,))
        lowpass_filter = np.sqrt(lowpass_window_sq_shifted)

        # Highpass filter: complement of lowpass for perfect reconstruction
        # Compute as complement of the shifted lowpass window to ensure |lowpass|² + |highpass|² = 1
        highpass_filter = np.sqrt(1.0 - lowpass_window_sq_shifted)

        return lowpass_filter, highpass_filter

    def _forward_transform_1d(
        self, signal: npt.NDArray, axis_index: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Apply 1D Meyer wavelet forward transform along specified axis.

        Parameters
        ----------
        signal : npt.NDArray
            Input array (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            Lowpass and highpass subbands. Output dtype matches input:
            real input produces real output, complex input produces complex output.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> lowpass.shape
        (32, 64)
        >>> highpass.shape
        (32, 64)
        """
        last_axis_index = signal.ndim - 1
        signal = np.swapaxes(signal, axis_index, last_axis_index)
        signal_shape = signal.shape
        signal_length = signal_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Reshape filters for broadcasting
        lowpass_filter = np.reshape(lowpass_filter, (1, signal_length))
        highpass_filter = np.reshape(highpass_filter, (1, signal_length))

        # Preserve input dtype
        input_dtype = signal.dtype
        is_complex_input = np.iscomplexobj(signal)

        # Transform to frequency domain
        signal_frequency_domain = np.fft.fft(signal, axis=last_axis_index)

        # Apply filters and transform back
        lowpass_full_resolution = np.fft.ifft(
            lowpass_filter * signal_frequency_domain, axis=last_axis_index
        )
        highpass_full_resolution = np.fft.ifft(
            highpass_filter * signal_frequency_domain, axis=last_axis_index
        )

        # Preserve complex values for complex input, take real for real input
        # Preserve the original dtype when taking real part or casting complex
        if not is_complex_input:
            # Take real part and cast back to original real dtype
            lowpass_full_resolution = lowpass_full_resolution.real
            highpass_full_resolution = highpass_full_resolution.real
        lowpass_full_resolution = lowpass_full_resolution.astype(input_dtype)
        highpass_full_resolution = highpass_full_resolution.astype(input_dtype)

        # Downsample by factor of 2 (take every other sample)
        lowpass_subband = lowpass_full_resolution[..., ::2]
        highpass_subband = highpass_full_resolution[..., 1::2]

        # Swap axes back to original order
        lowpass_subband = np.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = np.swapaxes(highpass_subband, axis_index, last_axis_index)

        return lowpass_subband, highpass_subband

    def _inverse_transform_1d(
        self,
        lowpass_subband: npt.NDArray,
        highpass_subband: npt.NDArray,
        axis_index: int,
    ) -> npt.NDArray:
        """
        Apply 1D Meyer wavelet inverse transform along specified axis.

        Parameters
        ----------
        lowpass_subband : npt.NDArray
            Lowpass subband (real or complex).
        highpass_subband : npt.NDArray
            Highpass subband (real or complex).
        axis_index : int
            Axis along which to apply the transform.

        Returns
        -------
        npt.NDArray
            Reconstructed array. Output dtype matches input: real input produces
            real output, complex input produces complex output.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> lowpass, highpass = wavelet._forward_transform_1d(signal, 0)
        >>> reconstructed = wavelet._inverse_transform_1d(lowpass, highpass, 0)
        >>> np.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        last_axis_index = lowpass_subband.ndim - 1
        lowpass_subband = np.swapaxes(lowpass_subband, axis_index, last_axis_index)
        highpass_subband = np.swapaxes(highpass_subband, axis_index, last_axis_index)

        # Compute upsampled shape
        upsampled_shape = list(lowpass_subband.shape)
        upsampled_shape[-1] = 2 * upsampled_shape[-1]

        # Determine dtype based on input - preserve the original dtype
        is_complex = np.iscomplexobj(lowpass_subband) or np.iscomplexobj(
            highpass_subband
        )

        # Determine the appropriate dtype for upsampled arrays
        # If either subband is complex, we need complex dtype for both arrays
        # Use the dtype of the complex subband, or promote to complex if needed
        if is_complex:
            # If both are complex, use the dtype of lowpass (or highpass if lowpass is real)
            if np.iscomplexobj(lowpass_subband):
                dtype = lowpass_subband.dtype
            elif np.iscomplexobj(highpass_subband):
                dtype = highpass_subband.dtype
            else:
                # Should not happen if is_complex is True, but handle gracefully
                dtype = np.result_type(lowpass_subband, highpass_subband)
        else:
            # Both are real, use lowpass dtype as reference
            dtype = lowpass_subband.dtype

        # Preserve the input dtype for final output casting
        input_dtype = lowpass_subband.dtype

        # Pre-allocate upsampled arrays
        lowpass_upsampled = np.zeros(upsampled_shape, dtype=dtype)
        highpass_upsampled = np.zeros(upsampled_shape, dtype=dtype)

        # Interleave subbands (upsample by inserting zeros)
        lowpass_upsampled[..., ::2] = lowpass_subband
        highpass_upsampled[..., 1::2] = highpass_subband

        signal_length = upsampled_shape[-1]

        # Get pre-computed filters
        lowpass_filter, highpass_filter = self._filters[signal_length]

        # Reshape filters for broadcasting
        lowpass_filter = np.reshape(lowpass_filter, (1, signal_length))
        highpass_filter = np.reshape(highpass_filter, (1, signal_length))

        # Transform to frequency domain and combine
        combined_frequency_domain = lowpass_filter * np.fft.fft(
            lowpass_upsampled, axis=last_axis_index
        ) + highpass_filter * np.fft.fft(highpass_upsampled, axis=last_axis_index)

        # Transform back to spatial domain
        reconstructed_full_resolution = np.fft.ifft(
            combined_frequency_domain, axis=last_axis_index
        )

        # Preserve complex values for complex input, take real for real input
        # Preserve the appropriate dtype based on whether input was complex
        if is_complex:
            # Cast back to appropriate complex dtype (FFT may promote complex64 to complex128)
            # Use the dtype we determined earlier (which matches the complex subband)
            reconstructed_signal = 2 * reconstructed_full_resolution
            reconstructed_signal = reconstructed_signal.astype(dtype)
        else:
            # Take real part and cast back to original real dtype
            reconstructed_signal = 2 * reconstructed_full_resolution.real
            reconstructed_signal = reconstructed_signal.astype(input_dtype)

        return np.swapaxes(reconstructed_signal, axis_index, last_axis_index)

    def forward(
        self, signal: npt.NDArray[F] | npt.NDArray[C]
    ) -> list[list[npt.NDArray[F] | npt.NDArray[C]]]:
        """
        Apply multi-dimensional Meyer wavelet forward transform.

        Decomposes the input signal into 2^dimension subbands organized into 2
        subband groups. Returns all subbands in a nested list structure similar
        to UDCT.

        Parameters
        ----------
        signal : ``npt.NDArray[F]`` | ``npt.NDArray[C]``
            Input array (real or complex). Must match the shape specified
            during initialization.

        Returns
        -------
        list[list[``npt.NDArray[F]`` | ``npt.NDArray[C]``]]
            All subbands organized into 2 subband groups:

            - coefficients[0]: [lowpass] - single lowpass subband (1 subband)
            - coefficients[1]: [highpass_0, highpass_1, ...] - all highpass
              subbands (2^dimension - 1 subbands)

            Each subband has shape approximately half the input shape in each
            dimension.

        Raises
        ------
        ValueError
            If signal shape does not match expected shape.

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> coefficients = wavelet.forward(signal)
        >>> len(coefficients)  # 2 subband groups
        2
        >>> coefficients[0][0].shape  # Lowpass subband
        (32, 32)
        >>> len(coefficients[1])  # Highpass subbands
        3
        """
        # Validate input shape
        if signal.shape != self.shape:
            error_msg = f"Signal shape {signal.shape} does not match expected shape {self.shape}"
            raise ValueError(error_msg)

        # Start with the full signal
        current_bands = [signal]

        # Apply 1D transform along each dimension
        for dimension_index in range(self.dimension):
            new_bands: list[npt.NDArray] = []
            for band in current_bands:
                lowpass_subband, highpass_subband = self._forward_transform_1d(
                    band, dimension_index
                )
                new_bands.append(lowpass_subband)
                new_bands.append(highpass_subband)
            current_bands = new_bands

        # Return structure: [[lowpass], [highpass_0, highpass_1, ...]]
        # len(result) == 2 (num_subbands)
        return [[current_bands[0]], current_bands[1:]]

    def backward(
        self, coefficients: list[list[npt.NDArray[F] | npt.NDArray[C]]]
    ) -> npt.NDArray[F] | npt.NDArray[C]:
        """
        Apply multi-dimensional Meyer wavelet inverse transform.

        Reconstructs the original signal from the full coefficient structure
        returned by forward().

        Parameters
        ----------
        coefficients : list[list[``npt.NDArray[F]`` | ``npt.NDArray[C]``]]
            Full coefficient structure from forward() with 2 subband groups:

            - coefficients[0]: [lowpass] - single lowpass subband
            - coefficients[1]: [highpass_0, highpass_1, ...] - all highpass
              subbands

        Returns
        -------
        ``npt.NDArray[F]`` | ``npt.NDArray[C]``
            Reconstructed signal with shape matching the original input.

        Raises
        ------
        ValueError
            If coefficients structure is invalid (must have 2 subband groups).

        Examples
        --------
        >>> import numpy as np
        >>> from curvelets.numpy import MeyerWavelet
        >>> wavelet = MeyerWavelet(shape=(64, 64))
        >>> signal = np.random.randn(64, 64)
        >>> coefficients = wavelet.forward(signal)
        >>> reconstructed = wavelet.backward(coefficients)
        >>> np.allclose(signal, reconstructed, atol=1e-10)
        True
        """
        # Validate coefficient structure
        if len(coefficients) != 2:
            error_msg = (
                f"coefficients must have 2 subband groups, got {len(coefficients)}"
            )
            raise ValueError(error_msg)

        # Extract lowpass and highpass bands
        lowpass_subband = coefficients[0][0]
        highpass_bands: list[npt.NDArray[F] | npt.NDArray[C]] = coefficients[1]

        # Combine lowpass with highpass bands
        all_bands = [lowpass_subband, *highpass_bands]

        # Apply inverse transform along each dimension in reverse order
        current_bands = all_bands
        for dimension_index in range(self.dimension - 1, -1, -1):
            new_bands: list[npt.NDArray] = []
            for band_index in range(len(current_bands) // 2):
                reconstructed = self._inverse_transform_1d(
                    current_bands[2 * band_index],
                    current_bands[2 * band_index + 1],
                    dimension_index,
                )
                new_bands.append(reconstructed)
            current_bands = new_bands

        # Return the fully reconstructed signal
        return current_bands[0]
